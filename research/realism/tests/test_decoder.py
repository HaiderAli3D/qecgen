"""Optional CPU evidence that the pilot learns parity and honours caller boundaries."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

pytest.importorskip("torch")

import torch

from research.realism import decoder
from research.realism.decoder import (
    DecoderBudgetExceededError,
    DecoderConcurrencyError,
    DecoderConfig,
    fit_decoder,
    predict,
)


def _parity() -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    # Second feature is an explicit validity mask, as at a syndrome boundary.
    x = np.array(
        [[[0, 1], [0, 1]], [[0, 1], [1, 1]], [[1, 1], [0, 1]], [[1, 1], [1, 1]]], dtype=np.bool_
    )
    return x, x[:, 0, 0] ^ x[:, 1, 0]


def test_cpu_learns_temporal_parity_reproducibly_and_records_selection(tmp_path: Path) -> None:
    x, y = _parity()
    config = DecoderConfig(
        hidden_size=12,
        batch_size=64,
        max_epochs=140,
        patience=50,
        learning_rate=0.05,
        min_delta=0.0,
        device="cpu",
        max_seconds=30,
    )
    train_x = np.tile(x, (16, 1, 1))
    train_y = np.tile(y, 16)
    first = fit_decoder(train_x, train_y, x, y, config, 123, tmp_path / "first")
    second = fit_decoder(train_x, train_y, x, y, config, 123, tmp_path / "second")
    assert first.summary["validation_errors"] == 0
    assert first.summary["validation_loss"] < 0.02
    assert first.summary["validation_loss"] == second.summary["validation_loss"]
    np.testing.assert_array_equal(predict(first.model, x, batch_size=2), y)
    np.testing.assert_array_equal(predict(first.model, x), predict(second.model, x))
    curves = json.loads((tmp_path / "first" / "learning_curve.json").read_text())
    best = min(curves, key=lambda row: row["validation_loss"])
    assert best["epoch"] == first.summary["best_epoch"]
    resolved = json.loads((tmp_path / "first" / "config.json").read_text())
    assert resolved["training_positive_fraction"] == 0.5
    assert resolved["input_sha256"]["train_x"]
    assert resolved["versions"]["decoder_source_sha256"]
    assert first.checkpoint_path.exists()


def test_fine_tuning_initializes_from_selected_checkpoint(tmp_path: Path) -> None:
    x, y = _parity()
    config = DecoderConfig(hidden_size=8, batch_size=4, max_epochs=1, device="cpu")
    original = fit_decoder(x, y, x, y, config, 23, tmp_path / "original")
    fine_config = DecoderConfig(
        hidden_size=8,
        batch_size=4,
        max_epochs=1,
        device="cpu",
        init_checkpoint=str(original.checkpoint_path),
    )
    fine = fit_decoder(x, y, x, y, fine_config, 99, tmp_path / "fine")
    assert fine.summary["initial_validation_loss"] == original.summary["validation_loss"]
    assert json.loads((tmp_path / "fine" / "config.json").read_text())["initial_checkpoint_sha256"]


def test_expired_deadline_never_starts_a_run(tmp_path: Path) -> None:
    x, y = _parity()
    config = DecoderConfig(device="cpu", deadline_monotonic=time.monotonic() - 1)
    with pytest.raises(DecoderBudgetExceededError):
        fit_decoder(x, y, x, y, config, 1, tmp_path / "never")
    assert not (tmp_path / "never").exists()


def test_prediction_rejects_expired_budget_and_different_shape(tmp_path: Path) -> None:
    x, y = _parity()
    fitted = fit_decoder(x, y, x, y, DecoderConfig(max_epochs=1, device="cpu"), 1, tmp_path)
    with pytest.raises(DecoderBudgetExceededError):
        predict(fitted.model, x, deadline_monotonic=time.monotonic() - 1)
    with pytest.raises(ValueError, match="shape"):
        predict(fitted.model, x[:, :1])
    assert predict(fitted.model, x[:0]).shape == (0,)


def test_rejects_nonbinary_features_and_misaligned_targets(tmp_path: Path) -> None:
    x, y = _parity()
    invalid = x.astype(np.float32)
    invalid[0, 0, 0] = 0.5
    config = DecoderConfig(device="cpu")
    with pytest.raises(ValueError, match="binary"):
        fit_decoder(invalid, y, x, y, config, 1, tmp_path)
    with pytest.raises(ValueError, match="shape"):
        fit_decoder(x, y[:-1], x, y, config, 1, tmp_path)
    with pytest.raises(ValueError, match="shapes must match"):
        fit_decoder(x, y, x[:, :1], y, config, 1, tmp_path)
    assert not (tmp_path / "config.json").exists()


def test_invalid_config_fails_before_training() -> None:
    with pytest.raises(ValueError, match="max_seconds"):
        DecoderConfig(max_seconds=float("nan"))
    with pytest.raises(ValueError, match="patience"):
        DecoderConfig(patience=0)


def test_training_restores_callers_torch_random_state(tmp_path: Path) -> None:
    x, y = _parity()
    before = torch.get_rng_state().clone()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    fitted = fit_decoder(x, y, x, y, DecoderConfig(max_epochs=1, device="cpu"), 42, tmp_path)
    predict(fitted.model, x)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.are_deterministic_algorithms_enabled() == old_determinism


def test_mid_training_deadline_keeps_validated_checkpoint_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    x, y = _parity()
    clock = [1000.0]
    original_tensor = decoder._tensor

    def timed_tensor(array: decoder.BitArray, device: torch.device) -> torch.Tensor:
        if torch.is_grad_enabled():
            clock[0] += 2.0
        return original_tensor(array, device)

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(decoder, "_tensor", timed_tensor)
    result = fit_decoder(
        x, y, x, y, DecoderConfig(max_seconds=3, batch_size=4, device="cpu"), 7, tmp_path
    )
    assert result.summary["stop_reason"] == "time_budget"
    assert result.summary["epochs_completed"] == 0
    assert result.summary["batches_completed"] == 1
    assert result.summary["best_epoch"] == 0
    assert result.summary["validation_loss"] == result.summary["initial_validation_loss"]
    assert len(json.loads((tmp_path / "learning_curve.json").read_text())) == 1
    with pytest.raises(DecoderBudgetExceededError):
        predict(result.model, x)


@pytest.mark.parametrize("max_seconds,shared_deadline", [(3.0, None), (30.0, 1003.0)])
def test_capped_training_reserves_prediction_time_without_extending_overall_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_seconds: float,
    shared_deadline: float | None,
) -> None:
    x, y = _parity()
    clock = [1000.0]
    training_started = [False]
    original_tensor = decoder._tensor

    def timed_tensor(array: decoder.BitArray, device: torch.device) -> torch.Tensor:
        if torch.is_grad_enabled():
            training_started[0] = True
            clock[0] += 1.43
        elif training_started[0]:
            clock[0] += 0.05
        return original_tensor(array, device)

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(decoder, "_tensor", timed_tensor)
    config = DecoderConfig(
        max_seconds=max_seconds,
        deadline_monotonic=shared_deadline,
        batch_size=4,
        device="cpu",
    )
    result = fit_decoder(x, y, x, y, config, 7, tmp_path)
    assert result.summary["stop_reason"] == "time_budget"
    assert result.summary["best_epoch"] == 0
    assert result.model.deadline_monotonic == 1003.0
    resolved = json.loads((tmp_path / "config.json").read_text())
    assert resolved["training_deadline_monotonic"] == pytest.approx(1002.85)
    assert resolved["inference_reserve_seconds"] == pytest.approx(0.15)
    predictions = predict(result.model, x)
    assert int(np.count_nonzero(predictions != y)) == result.summary["validation_errors"]
    assert clock[0] < result.model.deadline_monotonic
    clock[0] = 1003.0
    with pytest.raises(DecoderBudgetExceededError):
        predict(result.model, x, deadline_monotonic=1004.0)


def test_preparation_timeout_records_config_and_failed_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    x, y = _parity()
    clock = [1000.0]
    original_hash = decoder._array_hash

    def timed_hash(array: decoder.BitArray, deadline: float) -> str:
        clock[0] += 4.0
        return original_hash(array, deadline)

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(decoder, "_array_hash", timed_hash)
    with pytest.raises(DecoderBudgetExceededError):
        fit_decoder(x, y, x, y, DecoderConfig(max_seconds=3, device="cpu"), 7, tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert config["seed"] == 7
    assert config["preparation_complete"] is False
    assert summary["stop_reason"] == "budget_during_preparation"
    assert summary["checkpoint_available"] is False
    assert not (tmp_path / "best.pt").exists()


def test_overlapping_calls_are_rejected_without_corrupting_global_state() -> None:
    entered = threading.Event()
    release = threading.Event()
    previous_rng = torch.get_rng_state().clone()
    previous_cudnn = torch.backends.cudnn.enabled
    previous_determinism = torch.are_deterministic_algorithms_enabled()

    def hold_runtime() -> None:
        with decoder._deterministic_run(torch.device("cpu"), 1, 123):
            assert torch.backends.cudnn.enabled
            entered.set()
            assert release.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(hold_runtime)
        assert entered.wait(timeout=5)
        try:
            with (
                pytest.raises(DecoderConcurrencyError, match="separate processes"),
                decoder._deterministic_run(torch.device("cpu"), 1, 9),
            ):
                pytest.fail("Overlapping runtime mutation was allowed")
        finally:
            release.set()
        running.result(timeout=5)
    assert torch.equal(previous_rng, torch.get_rng_state())
    assert torch.backends.cudnn.enabled == previous_cudnn
    assert torch.are_deterministic_algorithms_enabled() == previous_determinism
