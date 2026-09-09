"""A bounded, validation-selected GRU pilot, separate from qecgen's decoder registry.

The caller supplies only detector features (including boundary masks) and logical
observable targets. This module cannot inspect circuits, calibration, or final-test data.
Time limits are cooperative: they are checked around each synchronized minibatch, so
one running native operation cannot be preempted. Budget-limited runs are recorded as
such and must not be called converged decoder-transfer experiments.
Training reserves up to one second within the same budget for checkpoint restoration
and prediction. An operation that overruns this reserve still cannot extend that budget.
Calls in one process must not overlap: Torch's RNG and backend settings are global.
Use separate processes for concurrent experiment arms.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:
    raise ImportError(
        "The isolated realism decoder pilot needs its optional PyTorch environment; "
        "qecgen's core generation dependencies do not include PyTorch."
    ) from exc

BitArray = npt.NDArray[np.bool_]
InputArray = npt.NDArray[Any]
_RUNTIME_LOCK = threading.Lock()
# A long pilot may keep this module loaded while another agent edits its source.
_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class DecoderBudgetExceededError(TimeoutError):
    """No partial inference array is returned when the caller's deadline expires."""


class DecoderConcurrencyError(RuntimeError):
    """Overlapping calls cannot safely restore process-global Torch settings."""


@dataclass(frozen=True)
class DecoderConfig:
    hidden_size: int = 64
    layers: int = 1
    batch_size: int = 512
    max_epochs: int = 60
    patience: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    min_delta: float = 0.0001
    gradient_clip: float = 1.0
    max_seconds: float = 600.0
    deadline_monotonic: float | None = None
    device: Literal["auto", "cpu", "cuda"] = "auto"
    cpu_threads: int = 1
    init_checkpoint: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "layers",
            "batch_size",
            "max_epochs",
            "patience",
            "cpu_threads",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "gradient_clip", "max_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "min_delta"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.deadline_monotonic is not None and not math.isfinite(self.deadline_monotonic):
            raise ValueError("deadline_monotonic must be finite")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")


class _Network(nn.Module):
    def __init__(self, features: int, hidden_size: int, layers: int) -> None:
        super().__init__()
        self.gru = nn.GRU(features, hidden_size, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        _, hidden = self.gru(x)
        return cast(Tensor, self.head(hidden[-1])).squeeze(-1)


@dataclass
class FittedDecoder:
    network: _Network
    config: DecoderConfig
    input_shape: tuple[int, int]
    device: torch.device
    deadline_monotonic: float


@dataclass
class FitResult:
    model: FittedDecoder
    summary: dict[str, Any]
    checkpoint_path: Path


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise DecoderBudgetExceededError("The decoder's absolute time budget has expired")


def _features(x: InputArray, name: str, *, allow_empty: bool = False) -> BitArray:
    if not isinstance(x, np.ndarray) or x.ndim != 3:
        raise ValueError(f"{name} must have shape (shots, rounds, features)")
    if x.shape[1] < 1 or x.shape[2] < 1 or (not allow_empty and x.shape[0] < 1):
        raise ValueError(f"{name} must have nonempty shot, round and feature dimensions")
    if x.dtype.kind not in "buif" or not np.all((x == 0) | (x == 1)):
        raise ValueError(f"{name} must contain only binary detector bits and validity masks")
    return x.astype(np.bool_, copy=False)


def _targets(y: InputArray, shots: int, name: str) -> BitArray:
    if not isinstance(y, np.ndarray) or y.shape not in ((shots,), (shots, 1)):
        raise ValueError(f"{name} must have shape (shots,) or (shots, 1)")
    if y.dtype.kind not in "buif" or not np.all((y == 0) | (y == 1)):
        raise ValueError(f"{name} must contain one binary logical-observable target per shot")
    return y.reshape(shots).astype(np.bool_, copy=False)


def _array_hash(array: BitArray, deadline: float) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": array.shape, "dtype": str(array.dtype)}).encode())
    for start in range(0, len(array), 4096):
        _check_deadline(deadline)
        digest.update(np.ascontiguousarray(array[start : start + 4096]).tobytes())
    return digest.hexdigest()


def _resolve_device(config: DecoderConfig) -> torch.device:
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was explicitly requested but is unavailable")
    use_cuda = config.device != "cpu" and torch.cuda.is_available()
    if use_cuda:
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace not in (":4096:8", ":16:8"):
            raise ValueError("Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    return torch.device("cuda:0" if use_cuda else "cpu")


@contextmanager
def _deterministic_run(device: torch.device, cpu_threads: int, seed: int | None) -> Iterator[None]:
    """Restore process RNG and backend settings rather than perturbing another experiment."""
    if not _RUNTIME_LOCK.acquire(blocking=False):
        raise DecoderConcurrencyError("Concurrent decoder calls require separate processes")
    try:
        with _torch_state(device, cpu_threads, seed):
            yield
    finally:
        _RUNTIME_LOCK.release()


@contextmanager
def _torch_state(device: torch.device, cpu_threads: int, seed: int | None) -> Iterator[None]:
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_precision = torch.get_float32_matmul_precision()
    devices = [device.index or 0] if device.type == "cuda" else []
    try:
        torch.set_num_threads(cpu_threads)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        with (
            torch.random.fork_rng(devices=devices),
            torch.backends.cudnn.flags(
                enabled=True, deterministic=True, benchmark=False, allow_tf32=False
            ),
        ):
            if seed is not None:
                torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
                if devices:
                    torch.cuda.default_generators[devices[0]].manual_seed(seed)
            yield
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_determinism, warn_only=previous_warn_only)
        torch.set_float32_matmul_precision(previous_precision)


def _tensor(array: BitArray, device: torch.device) -> Tensor:
    # Conversion happens per batch; full hardware arrays need not fit in GPU memory.
    return torch.from_numpy(np.array(array, dtype=np.float32, copy=True)).to(device)


def _evaluate(
    network: _Network,
    x: BitArray,
    y: BitArray,
    batch_size: int,
    device: torch.device,
    deadline: float,
) -> tuple[float, int]:
    network.eval()
    total_loss = 0.0
    errors = 0
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            _check_deadline(deadline)
            stop = start + batch_size
            logits = network(_tensor(x[start:stop], device))
            target = _tensor(y[start:stop], device)
            total_loss += float(
                nn.functional.binary_cross_entropy_with_logits(
                    logits, target, reduction="sum"
                ).item()
            )
            errors += int(((logits >= 0) != target.bool()).sum().item())
            _check_deadline(deadline)
    return total_loss / len(x), errors


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _save_checkpoint(
    path: Path,
    network: _Network,
    config: DecoderConfig,
    input_shape: tuple[int, int],
    seed: int,
    epoch: int,
) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "format_version": 1,
            "input_shape": input_shape,
            "hidden_size": config.hidden_size,
            "layers": config.layers,
            "seed": seed,
            "epoch": epoch,
            "state_dict": {
                name: value.detach().cpu().clone() for name, value in network.state_dict().items()
            },
        },
        temporary,
    )
    os.replace(temporary, path)


def _load_checkpoint(
    path: Path, network: _Network, config: DecoderConfig, input_shape: tuple[int, int]
) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = (1, input_shape, config.hidden_size, config.layers)
    actual = (
        payload.get("format_version"),
        tuple(payload.get("input_shape", ())),
        payload.get("hidden_size"),
        payload.get("layers"),
    )
    if actual != expected:
        raise ValueError("Initial checkpoint architecture or detector feature shape does not match")
    network.load_state_dict(payload["state_dict"], strict=True)


def fit_decoder(
    train_x: InputArray,
    train_y: InputArray,
    valid_x: InputArray,
    valid_y: InputArray,
    config: DecoderConfig,
    seed: int,
    output_dir: Path | str,
) -> FitResult:
    """Select one checkpoint on validation loss; no test-set argument exists.

    Every training shot is visited once per completed epoch. Shuffling changes only
    visit order, never class balance. Checkpoint initialization resets Adam so a
    fine-tuning run has its own recorded optimizer and validation selection history.
    """
    started = time.monotonic()
    overall_deadline = min(
        started + config.max_seconds,
        math.inf if config.deadline_monotonic is None else config.deadline_monotonic,
    )
    _check_deadline(overall_deadline)
    inference_reserve = min(1.0, 0.05 * (overall_deadline - started))
    training_deadline = overall_deadline - inference_reserve
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    train = _features(train_x, "train_x")
    valid = _features(valid_x, "valid_x")
    if train.shape[1:] != valid.shape[1:]:
        raise ValueError("Training and validation round/feature shapes must match")
    train_target = _targets(train_y, len(train), "train_y")
    valid_target = _targets(valid_y, len(valid), "valid_y")
    input_shape = (train.shape[1], train.shape[2])
    device = _resolve_device(config)
    output = Path(output_dir)
    names = ("config.json", "learning_curve.json", "summary.json", "best.pt")
    if any((output / name).exists() for name in names):
        raise FileExistsError("Decoder output directory already contains a recorded run")
    output.mkdir(parents=True, exist_ok=True)
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
        "decoder_source_sha256": _SOURCE_SHA256,
    }
    resolved = {
        "config": asdict(config),
        "seed": seed,
        "resolved_device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
        "versions": versions,
        "input_shape": input_shape,
        "train_shots": len(train),
        "validation_shots": len(valid),
        "training_positive_fraction": float(train_target.mean()),
        "validation_positive_fraction": float(valid_target.mean()),
        "input_sha256": {},
        "initial_checkpoint_sha256": None,
        "preparation_complete": False,
        "deterministic_algorithms": True,
        "float32_matmul_precision": "highest",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "class_reweighting": False,
        "replacement_sampling": False,
        "selection_metric": "validation_binary_cross_entropy",
        "budget_kind": "cooperative_wall_clock_including_prediction",
        "absolute_deadline_monotonic": overall_deadline,
        "training_deadline_monotonic": training_deadline,
        "inference_reserve_seconds": inference_reserve,
    }
    _write_json(output / "config.json", resolved)
    try:
        resolved["input_sha256"] = {
            name: _array_hash(array, training_deadline)
            for name, array in (
                ("train_x", train),
                ("train_y", train_target),
                ("valid_x", valid),
                ("valid_y", valid_target),
            )
        }
        if config.init_checkpoint is not None:
            resolved["initial_checkpoint_sha256"] = hashlib.sha256(
                Path(config.init_checkpoint).read_bytes()
            ).hexdigest()
        _check_deadline(training_deadline)
    except DecoderBudgetExceededError:
        _write_json(
            output / "summary.json",
            {
                "stop_reason": "budget_during_preparation",
                "epochs_completed": 0,
                "elapsed_seconds": time.monotonic() - started,
                "checkpoint_available": False,
            },
        )
        raise
    resolved["preparation_complete"] = True
    _write_json(output / "config.json", resolved)
    checkpoint = output / "best.pt"
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    history: list[dict[str, Any]] = []
    stop_reason = "max_epochs"
    completed_epochs = 0
    batches_completed = 0
    best_epoch = 0
    with _deterministic_run(device, config.cpu_threads, seed):
        network = _Network(input_shape[1], config.hidden_size, config.layers).to(device)
        if config.init_checkpoint is not None:
            _load_checkpoint(Path(config.init_checkpoint), network, config, input_shape)
        optimizer = torch.optim.Adam(
            network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()
        try:
            initial_loss, best_errors = _evaluate(
                network, valid, valid_target, config.batch_size, device, training_deadline
            )
        except DecoderBudgetExceededError:
            _write_json(
                output / "summary.json",
                {
                    "stop_reason": "budget_before_initial_validation",
                    "epochs_completed": 0,
                    "elapsed_seconds": time.monotonic() - started,
                    "checkpoint_available": False,
                },
            )
            raise
        best_loss = initial_loss
        patience_reference = best_loss
        stale_epochs = 0
        history.append(
            {
                "epoch": 0,
                "training_loss": None,
                "validation_loss": best_loss,
                "validation_errors": best_errors,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        _save_checkpoint(checkpoint, network, config, input_shape, seed, 0)
        _write_json(output / "learning_curve.json", history)
        for epoch in range(1, config.max_epochs + 1):
            network.train()
            total_loss = 0.0
            shots_seen = 0
            order = rng.permutation(len(train))
            try:
                for start in range(0, len(train), config.batch_size):
                    _check_deadline(training_deadline)
                    batch_indices = order[start : start + config.batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    logits = network(_tensor(train[batch_indices], device))
                    loss = criterion(logits, _tensor(train_target[batch_indices], device))
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        network.parameters(), config.gradient_clip, error_if_nonfinite=True
                    )
                    optimizer.step()
                    total_loss += float(loss.item()) * len(batch_indices)
                    shots_seen += len(batch_indices)
                    batches_completed += 1
                    _check_deadline(training_deadline)
                validation_loss, validation_errors = _evaluate(
                    network, valid, valid_target, config.batch_size, device, training_deadline
                )
            except DecoderBudgetExceededError:
                stop_reason = "time_budget"
                break
            completed_epochs = epoch
            history.append(
                {
                    "epoch": epoch,
                    "training_loss": total_loss / shots_seen,
                    "validation_loss": validation_loss,
                    "validation_errors": validation_errors,
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            # Selection uses the actual minimum; min_delta only controls patience.
            if validation_loss < best_loss:
                best_loss, best_errors, best_epoch = validation_loss, validation_errors, epoch
                _save_checkpoint(checkpoint, network, config, input_shape, seed, epoch)
            if validation_loss < patience_reference - config.min_delta:
                patience_reference = validation_loss
                stale_epochs = 0
            else:
                stale_epochs += 1
            _write_json(output / "learning_curve.json", history)
            if stale_epochs >= config.patience:
                stop_reason = "validation_patience"
                break
        _load_checkpoint(checkpoint, network, config, input_shape)
        network.eval()
    majority = bool(train_target.mean() >= 0.5)
    warnings = ["Validation loss and early stopping do not by themselves establish QEC competence."]
    if stop_reason == "time_budget":
        warnings.append("The wall-clock budget stopped training; convergence is unestablished.")
    if best_epoch == completed_epochs and stop_reason == "max_epochs":
        warnings.append("The best checkpoint was the final epoch; more training may improve it.")
    if best_epoch == 0:
        warnings.append("Training did not improve validation loss over initialization.")
    summary = {
        "stop_reason": stop_reason,
        "epochs_completed": completed_epochs,
        "batches_completed": batches_completed,
        "best_epoch": best_epoch,
        "initial_validation_loss": initial_loss,
        "validation_loss": best_loss,
        "validation_errors": best_errors,
        "validation_shots": len(valid),
        "validation_logical_error_rate": best_errors / len(valid),
        "training_loss_at_best_epoch": history[best_epoch]["training_loss"],
        "last_training_loss": history[-1]["training_loss"],
        "majority_validation_errors_from_training_class": int(
            np.count_nonzero(valid_target != majority)
        ),
        "validation_loss_improved": best_loss < initial_loss,
        "validation_plateau_observed": stop_reason == "validation_patience",
        "elapsed_seconds": time.monotonic() - started,
        "inference_reserve_seconds": inference_reserve,
        "prediction_budget_remaining_seconds": max(0.0, overall_deadline - time.monotonic()),
        "device": str(device),
        "seed": seed,
        "checkpoint_available": True,
        "checkpoint": str(checkpoint),
        "warnings": warnings,
    }
    _write_json(output / "summary.json", summary)
    return FitResult(
        FittedDecoder(network, config, input_shape, device, overall_deadline), summary, checkpoint
    )


def load_decoder(
    checkpoint: Path | str,
    *,
    device: Literal["auto", "cpu", "cuda"] = "auto",
    max_seconds: float = 60.0,
) -> FittedDecoder:
    """Reload a locked checkpoint with a separately recorded inference-only budget.

    Training deadlines cannot be reused for a later held-out evaluation. This does
    not extend training or select a new checkpoint; all weights are loaded strictly.
    """
    started = time.monotonic()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    shape = tuple(payload.get("input_shape", ()))
    if len(shape) != 2 or any(not isinstance(n, int) or n < 1 for n in shape):
        raise ValueError("Checkpoint input shape must have two positive dimensions")
    config = DecoderConfig(
        hidden_size=payload["hidden_size"],
        layers=payload["layers"],
        device=device,
        max_seconds=max_seconds,
    )
    resolved_device = _resolve_device(config)
    input_shape = (shape[0], shape[1])
    with _deterministic_run(resolved_device, config.cpu_threads, 0):
        network = _Network(input_shape[1], config.hidden_size, config.layers).to(resolved_device)
        _load_checkpoint(Path(checkpoint), network, config, input_shape)
        network.eval()
    return FittedDecoder(network, config, input_shape, resolved_device, started + max_seconds)


def predict(
    model: FittedDecoder,
    x: InputArray,
    *,
    batch_size: int | None = None,
    deadline_monotonic: float | None = None,
) -> BitArray:
    """Return one observable prediction per shot, refusing a partial timed-out result."""
    size = model.config.batch_size if batch_size is None else batch_size
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("batch_size must be a positive integer")
    if deadline_monotonic is not None and not math.isfinite(deadline_monotonic):
        raise ValueError("deadline_monotonic must be finite")
    deadline = min(
        model.deadline_monotonic, math.inf if deadline_monotonic is None else deadline_monotonic
    )
    _check_deadline(deadline)
    features = _features(x, "x", allow_empty=True)
    if features.shape[1:] != model.input_shape:
        raise ValueError("Prediction round/feature shape must match training")
    result = np.empty(len(features), dtype=np.bool_)
    with (
        _deterministic_run(model.device, model.config.cpu_threads, None),
        torch.inference_mode(),
    ):
        model.network.eval()
        for start in range(0, len(features), size):
            _check_deadline(deadline)
            predictions = model.network(_tensor(features[start : start + size], model.device)) >= 0
            result[start : start + size] = predictions.cpu().numpy()
            _check_deadline(deadline)
    return result
