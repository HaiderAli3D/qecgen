"""The residual pipeline: inventory, pilot, build, build-all, manifest.

One dataset goes through four stages, and the order of the first two is the point:

1. **Resolve before staging.** :func:`build` resolves the source and its frozen decoder
   *before* ``run.staged()`` creates anything, so a source that cannot be verified (a
   missing Willow table, a wrong content hash, a network failure) raises
   :class:`SourceBlockedError` and leaves no ``<name>/`` directory behind. A directory
   that exists with nothing in it would read, in ``MANIFEST.md`` and to a user listing
   ``data/residual``, as a dataset that was started.
2. **The pilot gates the build.** A 300,000-row run is only launched once a pilot record
   for *this* configuration says the projected disk footprint fits three times over and
   the free space is re-checked at launch. ``skip_pilot_gate`` exists for the tiny CLI
   tests and nothing else.
3. **Rows land in checkpoints, never in final files.** Stage A writes raw chunks, stage B
   decodes and features them chunk by chunk, both through
   :class:`~qecgen.residual.checkpoint.ChunkStore`, which refuses chunks from a different
   configuration, source, decoder, schema, seed or software version by name. A rerun
   replays the seeded stream from row 0 — the sampler cannot start mid-stream — and
   skips the *writes* for the verified contiguous prefix it already holds, checking each
   replayed chunk's row range against the checkpoint's; only stage B and C work is saved.
4. **Publication is all-or-nothing.** Stage C assembles every artifact inside
   ``run.staged(<output_root>/<name>)``, validates the staged set *there*, and lets the
   two-phase commit move it into place; ``MANIFEST.md`` is rendered the same way inside
   ``run.staged(output_root)``. A crash anywhere in stage C leaves no file under a final
   name.

The pilot also runs the alignment investigation the brief asks for before scaling
(``_investigate_alignment``): the negative controls are the only part that can *detect*
misalignment — the self-consistency item shares the decoder's construction path and is
labelled as such — and nothing in it adjusts data or labels. It reports.
"""

from __future__ import annotations

import ctypes
import datetime as dt
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qecgen import qa, run
from qecgen.dataset import StreamingContentHasher, library_versions
from qecgen.environments import derive_seeds
from qecgen.qa import clopper_pearson
from qecgen.residual import SCHEMA_VERSION
from qecgen.residual.checkpoint import (
    CHUNK_SUFFIX,
    INDEX_FILENAME,
    CheckpointError,
    CheckpointIdentity,
    ChunkStore,
)
from qecgen.residual.config import (
    DecoderKind,
    GenerationMode,
    ResidualConfig,
    SourceKind,
    config_hash,
    load_config,
    resolved_dict,
)
from qecgen.residual.decoder import DecoderModel, decode_chunk, pm_wrong, truth_from_packed
from qecgen.residual.features import (
    FeatureContext,
    audit_feature_inputs,
    extract_features,
)
from qecgen.residual.graph import MatchingGraphSummary, TimeSlices, summarise_graph, time_slices
from qecgen.residual.report import render_manifest, render_note
from qecgen.residual.sanity import SanityModelUnavailableError, run_sanity_models
from qecgen.residual.sources import FetchCohort, ResolvedSource, RowChunk, resolve_source
from qecgen.residual.splits import SPLIT_CODES, SPLIT_NAMES, assign_splits
from qecgen.residual.validation import ValidationReport, validate_dataset_dir
from qecgen.residual.writers import (
    file_sha256,
    write_decoder_files,
    write_features_csv,
    write_json,
    write_raw_hdf5,
)
from qecgen.residual.zenodo import WillowSourceBlockedError
from qecgen.sampling import packed_width, unpack_bits

__all__ = [
    "BLOCKED_FILENAME",
    "BUILD_ORDER",
    "CHECKPOINT_DIRNAME",
    "EXPECTED_LEGACY_BAND",
    "INVENTORY_FILENAME",
    "MANIFEST_FILENAME",
    "PILOT_FILENAME",
    "PRIOR_ACCURACY_RANGE",
    "Log",
    "PilotGateError",
    "SourceBlockedError",
    "ValidationFailedError",
    "build",
    "build_all",
    "checkpoint_root",
    "dataset_dir",
    "inventory",
    "peak_working_set_bytes",
    "pilot",
    "write_manifest",
]

Log = Callable[[str], None]
"""Where progress lines go; the CLI hands in its console, the tests take the default."""

CHECKPOINT_DIRNAME = ".checkpoints"
"""Under ``output_root``; a dot-name so a directory listing reads datasets, not chunks."""

INVENTORY_FILENAME = "inventory.json"
PILOT_FILENAME = "pilot.json"
BLOCKED_FILENAME = "blocked.json"
MANIFEST_FILENAME = "MANIFEST.md"

EXPECTED_LEGACY_BAND: tuple[float, float] = (0.06, 0.16)
"""The brief's expectation for the two independent Stim configurations' PyMatching
error rate. Compared and discussed, never enforced, and not applied to device or
hardware sources."""

PRIOR_ACCURACY_RANGE: tuple[float, float] = (0.84, 0.94)
"""The previously observed PyMatching accuracy range the brief asks to compare against."""

BUILD_ORDER: tuple[str, ...] = (
    "device_static_d3_r3",
    "willow_d3_z_r10_si1000",
    "indep_d9_r200_p0005",
    "indep_d25_r25_p0005",
)
"""Required datasets first, cheapest first; anything marked ``additional`` goes last."""

_PILOT_CONTROL_SEED = 20260916
_RESAMPLE_ROWS = 16_000
_DETECTION_RATE_SHOTS = 4096
_CONTROL_HALF_TOLERANCE = 0.15
_DETECTION_RATE_TOLERANCE = 0.10
_DISK_SAFETY_FACTOR = 3


class SourceBlockedError(RuntimeError):
    """The source could not be resolved; the dataset is reported blocked, never built.

    Carries the dataset name and the verbatim reason so ``build-all`` can record the row
    the manifest shows and move on to the next configuration.
    """

    def __init__(self, dataset_name: str, reason: str) -> None:
        self.dataset_name = dataset_name
        self.reason = reason
        super().__init__(f"{dataset_name} is blocked: {reason}")


class PilotGateError(RuntimeError):
    """The pilot record forbids the full run (or a later disk check does)."""


class ValidationFailedError(RuntimeError):
    """The staged artifact set failed validation; nothing was published."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        failed = [f"{c.name}: {c.detail}" for c in report.checks if not c.passed]
        super().__init__(
            "validation failed inside staging; nothing published. " + "; ".join(failed)
        )


# ---------------------------------------------------------------------------
# Paths and small helpers


def checkpoint_root(config: ResidualConfig) -> Path:
    return config.output_root / CHECKPOINT_DIRNAME / config.dataset_name


def dataset_dir(config: ResidualConfig) -> Path:
    return config.output_root / config.dataset_name


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _silent(_message: str) -> None:
    return None


def _write_record(path: Path, payload: dict[str, Any]) -> None:
    """A checkpoint-side JSON record written through a sibling temp file.

    ``pilot.json`` gates a build, so a record half-written by a crash must not read as a
    verdict; ``os.replace`` makes it appear whole or not at all.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        write_json(tmp, payload)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _free_disk_bytes(path: Path) -> int:
    """Free bytes on the volume that will hold ``path`` (nearest existing ancestor)."""
    probe = path
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    return int(shutil.disk_usage(probe).free)


def peak_working_set_bytes() -> int | None:
    """Lifetime high-water mark of this process's resident memory, in bytes.

    Windows reports it through ``GetProcessMemoryInfo`` (a private copy of the helper in
    ``research/realism/production_benchmark.py``; qecgen must not import ``research``).
    Elsewhere ``resource.getrusage`` gives ``ru_maxrss`` in kilobytes on Linux and bytes on
    macOS. ``None`` means the platform offered no counter, which the pilot records as such
    rather than as zero.
    """
    if sys.platform == "win32":

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(Counters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize)
    else:
        try:
            import resource
        except ImportError:  # pragma: no cover - a platform with neither counter
            return None
        maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return maxrss if sys.platform == "darwin" else maxrss * 1024


def _interval_dict(successes: int, trials: int) -> dict[str, Any]:
    interval = clopper_pearson(successes, trials)
    return {
        "successes": successes,
        "trials": trials,
        "rate": interval.point,
        "ci_low": interval.low,
        "ci_high": interval.high,
    }


def _resolve_or_block(config: ResidualConfig, fetch_cohort: FetchCohort | None) -> ResolvedSource:
    """Resolve the source, or raise :class:`SourceBlockedError` with the verbatim reason.

    Only the refusals resolution raises on purpose become a blocked dataset: a missing or
    unreadable file (``OSError``, which ``FileNotFoundError`` is), a rejected input
    (``ValueError``, which ``ConfigError``, ``MultiObservableError`` and the remote-zip
    errors are) and the Willow fetch's own :class:`WillowSourceBlockedError`. Anything
    else — a ``TypeError``, an ``AttributeError`` — is a bug, and a manifest row reading
    "blocked: TypeError: ..." would file that bug under "fix the input" where nobody
    looks for it; it propagates with its traceback instead. The exception type stays in
    the reason so a missing file and a bad hash remain distinguishable.
    """
    try:
        return resolve_source(config, fetch_cohort=fetch_cohort)
    except (OSError, ValueError, WillowSourceBlockedError) as error:
        raise SourceBlockedError(config.dataset_name, f"{type(error).__name__}: {error}") from error


@dataclass(frozen=True)
class _Frozen:
    """The source, decoder and feature context of one run, resolved once."""

    source: ResolvedSource
    model: DecoderModel
    graph: MatchingGraphSummary
    slices: TimeSlices
    context: FeatureContext

    @property
    def n_detectors(self) -> int:
        return self.model.n_detectors


def _freeze(config: ResidualConfig, fetch_cohort: FetchCohort | None) -> _Frozen:
    source = _resolve_or_block(config, fetch_cohort)
    model = source.decoder
    graph = summarise_graph(model.dem, model.matching, model.n_detectors)
    slices = time_slices(source.circuit_for_coords)
    context = FeatureContext(n_detectors=model.n_detectors, slices=slices, graph=graph)
    audit_feature_inputs(context)
    return _Frozen(source=source, model=model, graph=graph, slices=slices, context=context)


def _source_hash(source: ResolvedSource) -> str:
    """The one digest that identifies the source rows for the checkpoint identity."""
    hashes = source.identity.hashes
    for key in ("content_hash", "table_sha256", "config_sha256"):
        if key in hashes:
            return str(hashes[key])
    raise ValueError(f"source identity carries no row digest: {sorted(hashes)}")


def _total_rows(config: ResidualConfig, source: ResolvedSource) -> int:
    if config.generation.mode is GenerationMode.SOURCE_ROWS:
        return source.identity.shots
    if config.generation.shots is None:  # pragma: no cover - config requires it
        raise ValueError("generation.shots is required outside source_rows mode")
    return config.generation.shots


def _row_stream(config: ResidualConfig, source: ResolvedSource) -> Iterator[RowChunk]:
    """The rows of this dataset in order: generated for extend/fresh, read for source_rows."""
    generation = config.generation
    if generation.mode is GenerationMode.SOURCE_ROWS:
        return source.iter_source_rows(config.pipeline.checkpoint_rows)
    if generation.shots is None or generation.seed is None:  # pragma: no cover
        raise ValueError("extend/fresh modes need generation.shots and generation.seed")
    return source.iter_generated_rows(generation.shots, generation.seed, generation.chunk_size)


def _regroup(stream: Iterable[RowChunk], rows: int) -> Iterator[RowChunk]:
    """Re-cut a chunk stream into chunks of exactly ``rows`` (the last may be shorter).

    The sampler's ``chunk_size`` is a reproducibility contract and cannot be changed to
    suit the checkpoint size; the checkpoint size is a working-set choice. Regrouping here
    keeps the two independent, and never holds more than one checkpoint's rows.
    """
    det_parts: list[np.ndarray] = []
    obs_parts: list[np.ndarray] = []
    held = 0
    for dets, obs in stream:
        if dets.shape[0] != obs.shape[0]:
            raise ValueError("detector and observable chunks disagree in row count")
        start = 0
        while start < dets.shape[0]:
            take = min(rows - held, dets.shape[0] - start)
            det_parts.append(dets[start : start + take])
            obs_parts.append(obs[start : start + take])
            held += take
            start += take
            if held == rows:
                yield np.concatenate(det_parts), np.concatenate(obs_parts)
                det_parts, obs_parts, held = [], [], 0
    if held:
        yield np.concatenate(det_parts), np.concatenate(obs_parts)


def _decode_rows(
    frozen: _Frozen, dets: np.ndarray, obs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(pm_guess, pm_weight, truth, pm_wrong)`` for one chunk; features are not involved."""
    guess, weight = decode_chunk(frozen.model, dets)
    truth = truth_from_packed(obs, frozen.model.n_observables)
    return guess, weight, truth, pm_wrong(guess, truth)


# ---------------------------------------------------------------------------
# inventory


def _decoder_record(frozen: _Frozen) -> dict[str, Any]:
    model = frozen.model
    graph = frozen.graph
    return {
        "kind": model.kind.value,
        "dem_sha256": model.dem_sha256,
        "dem_blake2b128": model.dem_blake2b128,
        "num_detectors": model.n_detectors,
        "num_observables": model.n_observables,
        "num_errors": model.dem.num_errors,
        "dem_stats": model.provenance.get("dem_stats"),
        "matching_mode": model.provenance.get("matching_mode"),
        "fitted_in_this_pipeline": False,
        "third_party_fitting": model.provenance.get("third_party_fitting"),
        "graph": {
            "digest": graph.digest(),
            "n_components": graph.n_components,
            "n_boundary_components": graph.n_boundary_components,
            "n_logical_components": graph.n_logical_components,
            "n_collapsed_pairs": graph.n_collapsed_pairs,
            "n_conflicting_pairs": graph.n_conflicting_pairs,
            "n_ignored_components": graph.n_ignored_components,
            "boundary_set_size": int(graph.boundary_adjacent.sum()),
            "logical_set_size": int(graph.logical_adjacent.sum()),
            "every_node_boundary_adjacent": bool(graph.boundary_adjacent.all()),
        },
        "time_slices": {
            "n_slices": frozen.slices.n_slices,
            "sizes": [int(s) for s in frozen.slices.sizes.tolist()],
        },
    }


def inventory(
    config: ResidualConfig, *, fetch_cohort: FetchCohort | None = None, log: Log = _silent
) -> dict[str, Any]:
    """Resolve source and decoder, decode nothing, and record what was found.

    Written to ``.checkpoints/<name>/inventory.json`` and never into the published
    directory, which only ``run.staged()`` may populate.
    """
    frozen = _freeze(config, fetch_cohort)
    identity = frozen.source.identity
    record = {
        "dataset_name": config.dataset_name,
        "config_hash": config_hash(config),
        "resolved_config": resolved_dict(config),
        "identity": identity.to_dict(),
        "decoder": _decoder_record(frozen),
        "verification": frozen.source.verification,
        "versions": library_versions(),
        "recorded_at": _now(),
    }
    _write_record(checkpoint_root(config) / INVENTORY_FILENAME, record)
    log(
        f"inventory: {config.dataset_name} <- {identity.kind.value} "
        f"({identity.shots} source rows, {identity.n_detectors} detectors, "
        f"{identity.n_observables} observable); decoder {frozen.model.kind.value} "
        f"sha256 {frozen.model.dem_sha256[:12]}..."
    )
    return record


# ---------------------------------------------------------------------------
# pilot: existing rows, alignment investigation, throughput, projection


def _dem_marginals(model: DecoderModel) -> np.ndarray:
    """Per-detector firing probability the DEM predicts if mechanisms were independent.

    ``(1 - prod_j (1 - 2 p_j)) / 2`` over the mechanisms (``error`` instructions) whose
    XOR-reduced detector support contains the detector — the rule of
    ``research/realism/fit_noise.detector_marginals``, restated here because qecgen never
    imports ``research``. First/last-slice and boundary detectors have distinct rates,
    which is what makes this sensitive to a permuted or reversed column order.
    """
    products = np.ones(model.n_detectors, dtype=np.float64)
    for instruction in model.dem.flattened():
        if instruction.type != "error":
            continue
        support: set[int] = set()
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                support.symmetric_difference_update([int(target.val)])
        probability = float(instruction.args_copy()[0])
        for detector in support:
            products[detector] *= 1.0 - 2.0 * probability
    return (1.0 - products) / 2.0


def _blocks(chunks: Sequence[RowChunk], rows: int) -> Iterator[RowChunk]:
    """Packed chunks re-cut to at most ``rows`` each, for anything that must unpack.

    A checkpoint chunk of the d=9 source is its whole 16,000 x 16,000 table; unpacking
    it in one piece is 256 MB per copy, and the controls make three. Blocks of
    ``feature_rows`` keep the pilot's working set at the size the build itself uses.
    """
    for dets, obs in chunks:
        for start in range(0, int(dets.shape[0]), rows):
            yield dets[start : start + rows], obs[start : start + rows]


def _control_error_rate(
    frozen: _Frozen,
    chunks: Sequence[RowChunk],
    block: int,
    transform: Callable[[np.ndarray], np.ndarray],
) -> dict[str, Any]:
    """Decode the stored rows after ``transform`` rewrote each block's *unpacked* bits."""
    n_detectors = frozen.n_detectors
    wrong = 0
    total = 0
    for dets, obs in _blocks(chunks, block):
        bits = transform(unpack_bits(dets, n_detectors))
        repacked = np.packbits(bits, axis=1, bitorder="little")
        _, _, _, mismatch = _decode_rows(frozen, repacked, obs)
        wrong += int(mismatch.sum())
        total += int(mismatch.shape[0])
    return _interval_dict(wrong, total)


def _investigate_alignment(
    config: ResidualConfig,
    frozen: _Frozen,
    chunks: Sequence[RowChunk],
    aligned_wrong: int,
    detection_events: int,
    resample_rows: int,
    log: Log,
) -> dict[str, Any]:
    """The five-item alignment protocol on the stored source rows (Task 12).

    Only the negative controls can *detect* misalignment; (1) is self-consistency and
    labelled so; (3), (4) and (5) are structural and statistical positives. Nothing here
    changes a row or a label — the outcome is recorded and printed for a reader to judge
    before scaling.
    """
    identity = frozen.source.identity
    model = frozen.model
    block = config.pipeline.feature_rows
    n_rows = sum(int(d.shape[0]) for d, _ in chunks)
    n_detectors = frozen.n_detectors
    aligned = _interval_dict(aligned_wrong, n_rows)
    concerns: list[str] = []
    report: dict[str, Any] = {
        "aligned": aligned,
        "note": (
            "self_consistency shares the decoder's construction path and cannot detect "
            "misalignment; the negative controls, marginals, re-sample and detection-event "
            "rate are the evidence"
        ),
    }

    # (1) self-consistency through qa.decode_stored_shots (same construction path).
    if model.kind is DecoderKind.OFFICIAL_DEM:
        report["self_consistency"] = {
            "applicable": False,
            "reason": (
                "the official DEM is not derived from decoder.circuit, so "
                "qa.decode_stored_shots would decode with a different model"
            ),
        }
    else:
        dets_all = np.concatenate([d for d, _ in chunks])
        obs_all = np.concatenate([o for _, o in chunks])
        p_value = identity.noise_parameters.get("p")
        estimate = qa.decode_stored_shots(
            model.circuit,
            dets_all,
            obs_all,
            distance=identity.distance,
            p=float(p_value) if isinstance(p_value, int | float) else 0.0,
            rounds=identity.rounds,
            chunk_size=block,
        )
        equal = estimate.interval.successes == aligned_wrong
        report["self_consistency"] = {
            "applicable": True,
            "label": "self-consistency only (same construction path)",
            "qa_failures": estimate.interval.successes,
            "pipeline_failures": aligned_wrong,
            "equal": equal,
        }
        if not equal:
            concerns.append("qa.decode_stored_shots failure count differs from the pipeline's")

    # (2) negative controls: permuted columns and big-endian unpacking.
    rng = np.random.default_rng(np.random.SeedSequence(_PILOT_CONTROL_SEED))
    permutation = rng.permutation(n_detectors)
    permuted = _control_error_rate(frozen, chunks, block, lambda bits: bits[:, permutation])

    def big_endian(bits: np.ndarray) -> np.ndarray:
        packed = np.packbits(bits, axis=1, bitorder="little")
        return np.unpackbits(packed, axis=1, count=n_detectors, bitorder="big").astype(bool)

    reversed_bits = _control_error_rate(frozen, chunks, block, big_endian)
    for label, control in (
        ("permutation_control", permuted),
        ("big_endian_control", reversed_bits),
    ):
        near_half = abs(control["rate"] - 0.5) <= _CONTROL_HALF_TOLERANCE
        exceeds = control["rate"] > aligned["ci_high"]
        report[label] = {
            **control,
            "seed": _PILOT_CONTROL_SEED if label == "permutation_control" else None,
            "near_half": near_half,
            "exceeds_aligned_ci": exceeds,
            "expectation": "near 50% and well above the aligned rate",
        }
        if not exceeds:
            concerns.append(
                f"{label} error rate {control['rate']:.4f} does not exceed the aligned CI"
            )

    # (3) structural positive check: per-detector empirical rate vs DEM marginal.
    fired = np.zeros(n_detectors, dtype=np.int64)
    for dets, _ in _blocks(chunks, block):
        fired += unpack_bits(dets, n_detectors).sum(axis=0, dtype=np.int64)
    empirical = fired / n_rows
    predicted = _dem_marginals(model)
    pearson: float | None = None
    if n_detectors > 1 and empirical.std() > 0 and predicted.std() > 0:
        pearson = float(np.corrcoef(empirical, predicted)[0, 1])
    report["marginals"] = {
        "n_detectors": n_detectors,
        "pearson_r": pearson,
        "max_abs_difference": float(np.abs(empirical - predicted).max()),
        "mean_empirical": float(empirical.mean()),
        "mean_predicted": float(predicted.mean()),
        "rule": "(1 - prod_j (1 - 2 p_j)) / 2 over DEM mechanisms touching each detector",
    }
    if pearson is not None and pearson < 0.9:
        concerns.append(f"per-detector marginal Pearson r {pearson:.3f} < 0.9")

    # (4) fresh re-sample from derive_seeds(seed, 2)[1]: CI overlap with the stored rows.
    if identity.seed is None or frozen.source.identity.kind is SourceKind.HARDWARE_WILLOW:
        report["resample"] = {"applicable": False, "reason": "no seeded sampling stream"}
    else:
        child_seed = derive_seeds(identity.seed, 2)[1]
        wrong = 0
        seen = 0
        for dets, obs in frozen.source.iter_generated_rows(
            resample_rows, child_seed, config.generation.chunk_size
        ):
            _, _, _, mismatch = _decode_rows(frozen, dets, obs)
            wrong += int(mismatch.sum())
            seen += int(mismatch.shape[0])
        resample = _interval_dict(wrong, seen)
        overlap = (
            resample["ci_low"] <= aligned["ci_high"] and aligned["ci_low"] <= resample["ci_high"]
        )
        report["resample"] = {
            **resample,
            "applicable": True,
            "n_rows": seen,
            "seed": child_seed,
            "seed_rule": "derive_seeds(source_seed, 2)[1]",
            "ci_overlaps_stored": overlap,
        }
        if not overlap:
            concerns.append("re-sampled LER CI does not overlap the stored rows' CI")

    # (5) stored detection-event rate vs a fresh circuit sample.
    stored_rate = detection_events / (n_rows * n_detectors)
    if model.kind in (DecoderKind.CIRCUIT_DEM, DecoderKind.STATIC_PROFILE_DEM):
        fresh_rate = qa.detection_event_rate(
            model.circuit, _DETECTION_RATE_SHOTS, identity.seed if identity.seed is not None else 0
        )
        relative = abs(stored_rate - fresh_rate) / fresh_rate if fresh_rate > 0 else math.inf
        within = relative <= _DETECTION_RATE_TOLERANCE
        report["detection_event_rate"] = {
            "applicable": True,
            "stored": stored_rate,
            "fresh_circuit_sample": fresh_rate,
            "fresh_shots": _DETECTION_RATE_SHOTS,
            "relative_difference": relative if math.isfinite(relative) else None,
            "within_10_percent": within,
        }
        if not within:
            concerns.append("stored detection-event rate differs from the circuit's by > 10%")
    else:
        report["detection_event_rate"] = {
            "applicable": False,
            "stored": stored_rate,
            "reason": "no exact circuit model to sample (reference or official DEM)",
        }

    report["concerns"] = concerns
    log(
        "alignment: "
        f"aligned {aligned['rate']:.4f}, permuted {permuted['rate']:.4f}, "
        f"big-endian {reversed_bits['rate']:.4f}, marginal r "
        f"{'n/a' if pearson is None else f'{pearson:.3f}'}; "
        + ("no concerns" if not concerns else "concerns: " + "; ".join(concerns))
    )
    return report


def _expectation_report(config: ResidualConfig, existing: dict[str, Any]) -> dict[str, Any]:
    rate = float(existing["pm_error_rate"])
    if config.source.kind is not SourceKind.LEGACY_ML_CSV:
        return {
            "applied": False,
            "reason": (
                "the 6-16% band is an expectation for the independent Stim configurations only"
            ),
        }
    low, high = EXPECTED_LEGACY_BAND
    acc_low, acc_high = PRIOR_ACCURACY_RANGE
    return {
        "applied": True,
        "band": [low, high],
        "within_band": low <= rate <= high,
        "ci_overlaps_band": float(existing["pm_ci_low"]) <= high
        and float(existing["pm_ci_high"]) >= low,
        "prior_accuracy_range": [acc_low, acc_high],
        "accuracy": 1.0 - rate,
        "within_prior_accuracy_range": acc_low <= 1.0 - rate <= acc_high,
    }


def _throughput(
    config: ResidualConfig, frozen: _Frozen, generated_rows: int, log: Log
) -> dict[str, Any]:
    """Sample, decode, feature and write ``generated_rows`` fresh rows, timing each stage.

    Writes go to a temporary directory *inside* the checkpoint root, so they measure the
    volume the real build will use and vanish afterwards; nothing lands under the
    published directory.
    """
    generation = config.generation
    n_detectors = frozen.n_detectors
    timings = {"sampling": 0.0, "decoding": 0.0, "features": 0.0, "writing": 0.0}
    raw_chunks: list[RowChunk] = []
    feature_chunks: list[dict[str, np.ndarray]] = []
    rows = 0
    if generation.mode is GenerationMode.SOURCE_ROWS:
        stream: Iterator[RowChunk] = frozen.source.iter_source_rows(generation.chunk_size)
        origin = "source rows (no sampling stream)"
    else:
        if generation.seed is None:  # pragma: no cover - config requires it
            raise ValueError("generation.seed is required")
        stream = frozen.source.iter_generated_rows(
            generated_rows, generation.seed, generation.chunk_size
        )
        origin = "configured sampling stream"
    sampled_rows = 0
    started = time.perf_counter()
    for dets, obs in stream:
        sampled = time.perf_counter()
        timings["sampling"] += sampled - started
        # The generated stream samples exactly generated_rows rows, in the call sizes
        # chunk_sizes(generated_rows, chunk_size) (d9: one sample(10000) call for a
        # 10,000-row pilot at chunk 16000), so it never overshoots. The source-rows stream
        # reads whole chunk_size chunks and can hand back more than the pilot keeps
        # (Willow: a 10,000-row chunk for a smaller pilot), so sampling time is charged
        # per row *read* and the surplus is dropped rather than decoded.
        sampled_rows += int(dets.shape[0])
        take = min(int(dets.shape[0]), generated_rows - rows)
        dets = dets[:take]
        obs = obs[:take]
        guess, weight, truth, wrong = _decode_rows(frozen, dets, obs)
        decoded = time.perf_counter()
        timings["decoding"] += decoded - sampled
        features = np.concatenate(
            [
                extract_features(
                    dets[s : s + config.pipeline.feature_rows],
                    guess[s : s + config.pipeline.feature_rows],
                    weight[s : s + config.pipeline.feature_rows],
                    frozen.context,
                )
                for s in range(0, take, config.pipeline.feature_rows)
            ]
        )
        featured = time.perf_counter()
        timings["features"] += featured - decoded
        raw_chunks.append((dets, obs))
        feature_chunks.append(
            {
                "features": features,
                "pm_guess": guess,
                "pm_weight": weight,
                "truth": truth,
                "pm_wrong": wrong,
            }
        )
        rows += take
        if rows >= generated_rows:
            break
        started = time.perf_counter()
    if rows == 0:
        raise ValueError("the pilot stream produced no rows")

    root = checkpoint_root(config)
    root.mkdir(parents=True, exist_ok=True)
    codes = np.zeros(rows, dtype=np.int8)
    codes[:] = SPLIT_CODES["train"]
    with tempfile.TemporaryDirectory(dir=root, prefix="pilot-scratch-") as scratch_name:
        scratch = Path(scratch_name)
        started = time.perf_counter()
        write_features_csv(scratch / "pilot_features.csv", feature_chunks, codes)
        write_raw_hdf5(
            scratch / "pilot_raw.h5",
            raw_chunks,
            (np.arange(s, s + d.shape[0], dtype=np.int64) for s, (d, _) in _offsets(raw_chunks)),
            codes,
            {
                "n_detectors": n_detectors,
                "n_observables": 1,
                "source_content_hash": _source_hash(frozen.source),
                "config_hash": config_hash(config),
                "decoder_dem_sha256": frozen.model.dem_sha256,
            },
        )
        timings["writing"] = time.perf_counter() - started
        csv_bytes = (scratch / "pilot_features.csv").stat().st_size
        h5_bytes = (scratch / "pilot_raw.h5").stat().st_size
    packed_bytes = rows * (packed_width(n_detectors) + 1)
    pm_failures = int(sum(int(c["pm_wrong"].sum()) for c in feature_chunks))
    per_row = {stage: seconds / rows for stage, seconds in timings.items()}
    per_row["sampling"] = timings["sampling"] / sampled_rows
    log(
        f"throughput: {rows} rows from the {origin}; per row "
        + ", ".join(f"{k} {v * 1e3:.3f} ms" for k, v in per_row.items())
    )
    return {
        "generated_rows": rows,
        "sampled_rows": sampled_rows,
        "origin": origin,
        "pm_wrong": pm_failures,
        "seconds": timings,
        "seconds_per_row": per_row,
        "seconds_per_row_total": sum(per_row.values()),
        "bytes_per_row": {
            "features_csv": csv_bytes / rows,
            "raw_h5": h5_bytes / rows,
            "packed": packed_bytes / rows,
            "checkpoint_estimate": packed_bytes / rows + 8.0 * 24 + 32.0,
        },
        "raw_h5_gzip_ratio": h5_bytes / packed_bytes if packed_bytes else None,
        "peak_working_set_bytes": peak_working_set_bytes(),
    }


def _offsets(chunks: Sequence[RowChunk]) -> Iterator[tuple[int, RowChunk]]:
    offset = 0
    for chunk in chunks:
        yield offset, chunk
        offset += int(chunk[0].shape[0])


def _edge_prototype(
    frozen: _Frozen, chunks: Sequence[RowChunk], rows: int, log: Log
) -> dict[str, Any]:
    """Time ``decode_to_edges_array`` one shot at a time and state the fault-id ambiguity.

    Optional schema v2 material: recorded so PROGRESS.md can make the include/exclude
    decision from measurements rather than from a guess about throughput.
    """
    n_detectors = frozen.n_detectors
    matching = frozen.model.matching
    done = 0
    edges_total = 0
    started = time.perf_counter()
    for dets, _ in chunks:
        bits = unpack_bits(dets, n_detectors)
        for row in bits:
            if done >= rows:
                break
            edges_total += int(np.asarray(matching.decode_to_edges_array(row)).shape[0])
            done += 1
        if done >= rows:
            break
    elapsed = time.perf_counter() - started
    graph = frozen.graph
    pair_counts: dict[tuple[int, int], set[frozenset[int]]] = {}
    for record in graph.edge_records:
        key = (record.u, -1 if record.v is None else record.v)
        pair_counts.setdefault(key, set()).add(record.observables)
    merged = sum(1 for observables in pair_counts.values() if len(observables) > 1)
    log(f"edge prototype: {done} rows, {elapsed / max(done, 1) * 1e3:.3f} ms/row")
    return {
        "rows": done,
        "seconds_per_row": elapsed / done if done else None,
        "mean_edges_per_row": edges_total / done if done else None,
        "merged_pairs_with_differing_observables": merged,
        "fault_ids_unambiguous": merged == 0,
        "note": (
            "PyMatching merges parallel edges and keeps the first-inserted fault_ids; "
            "edge-level observable features are only unambiguous when no merged pair "
            "carries differing observable sets"
        ),
    }


def pilot(
    config: ResidualConfig,
    *,
    generated_rows: int = 10_000,
    resample_rows: int = _RESAMPLE_ROWS,
    edge_prototype_rows: int = 0,
    fetch_cohort: FetchCohort | None = None,
    log: Log = _silent,
) -> dict[str, Any]:
    """Decode every existing source row, investigate alignment, measure throughput, project.

    Writes ``.checkpoints/<name>/pilot.json`` whose ``resources_sufficient`` verdict gates
    :func:`build`. Printing ``pm_wrong: k / n`` here is the brief's "print and record".
    """
    if generated_rows < 1:
        raise ValueError(f"generated_rows must be >= 1, got {generated_rows}")
    frozen = _freeze(config, fetch_cohort)
    identity = frozen.source.identity
    record: dict[str, Any] = {
        "dataset_name": config.dataset_name,
        "config_hash": config_hash(config),
        "source_hash": _source_hash(frozen.source),
        "decoder_dem_sha256": frozen.model.dem_sha256,
        "started_at": _now(),
    }

    if identity.shots > 0:
        chunks: list[RowChunk] = []
        wrong_total = 0
        detection_events = 0
        block = config.pipeline.feature_rows
        for dets, obs in frozen.source.iter_source_rows(config.pipeline.checkpoint_rows):
            _, _, _, wrong = _decode_rows(frozen, dets, obs)
            wrong_total += int(wrong.sum())
            # Counted in feature_rows blocks for the reason _blocks states: unpacking a
            # whole checkpoint chunk of the d=9 source is a 256 MB bool array per copy.
            for part, _ in _blocks(((dets, obs),), block):
                detection_events += int(unpack_bits(part, frozen.n_detectors).sum())
            chunks.append((dets, obs))
        n_rows = sum(int(d.shape[0]) for d, _ in chunks)
        if n_rows != identity.shots:
            raise ValueError(f"source streamed {n_rows} rows but declares {identity.shots}")
        interval = clopper_pearson(wrong_total, n_rows)
        existing = {
            "n_rows": n_rows,
            "pm_wrong": wrong_total,
            "pm_wrong_fraction": interval.point,
            "pm_error_rate": interval.point,
            "pm_ci_low": interval.low,
            "pm_ci_high": interval.high,
            "accuracy": 1.0 - interval.point,
            "detection_event_rate": detection_events / (n_rows * frozen.n_detectors),
        }
        log(
            f"pm_wrong: {wrong_total} / {n_rows} ({interval.point:.6f}) on the existing source rows"
        )
        log(
            f"PyMatching LER {interval.point:.6f} [{interval.low:.6f}, {interval.high:.6f}], "
            f"detection-event rate {existing['detection_event_rate']:.6f}"
        )
        record["existing_rows"] = existing
        record["expectation"] = _expectation_report(config, existing)
        record["alignment"] = _investigate_alignment(
            config, frozen, chunks, wrong_total, detection_events, resample_rows, log
        )
        if edge_prototype_rows > 0:
            record["edge_prototype"] = _edge_prototype(frozen, chunks, edge_prototype_rows, log)
        del chunks
    else:
        record["existing_rows"] = None
        record["expectation"] = {"applied": False, "reason": "the source has no existing rows"}
        record["alignment"] = {"applicable": False, "reason": "no existing rows to investigate"}

    throughput = _throughput(config, frozen, generated_rows, log)
    record["throughput"] = throughput
    total_rows = _total_rows(config, frozen.source)
    per_row_bytes = throughput["bytes_per_row"]
    projected_bytes = int(
        total_rows
        * (
            per_row_bytes["features_csv"]
            + per_row_bytes["raw_h5"]
            + per_row_bytes["checkpoint_estimate"]
        )
    )
    free = _free_disk_bytes(config.output_root)
    sufficient = free >= _DISK_SAFETY_FACTOR * projected_bytes
    record["projection"] = {
        "rows": total_rows,
        "seconds": total_rows * throughput["seconds_per_row_total"],
        "bytes": projected_bytes,
        "free_disk_bytes": free,
        "safety_factor": _DISK_SAFETY_FACTOR,
    }
    record["resources_sufficient"] = sufficient
    record["finished_at"] = _now()
    _write_record(checkpoint_root(config) / PILOT_FILENAME, record)
    log(
        f"projection: {total_rows} rows, {record['projection']['seconds'] / 60:.1f} min, "
        f"{projected_bytes / 2**20:.1f} MiB; free {free / 2**30:.1f} GiB; "
        f"resources_sufficient={sufficient}"
    )
    return record


# ---------------------------------------------------------------------------
# build


def _load_pilot(config: ResidualConfig, frozen: _Frozen) -> dict[str, Any] | None:
    """The pilot record for *this* run — same configuration hash, source rows and
    decoder — or ``None``.

    The identity comparison lives here rather than in the gate because the gate is not
    the only reader: ``build`` also loads the record in ``source_rows`` mode and under
    ``skip_pilot_gate``, where no gate runs, and carries its source-row failure counts
    into the published summary. A record left behind by another configuration under the
    same dataset name (a changed seed, a re-fetched source, a different decoder member)
    would then be reported as this dataset's pilot numbers with nothing to say otherwise.
    """
    path = checkpoint_root(config) / PILOT_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    current = (config_hash(config), _source_hash(frozen.source), frozen.model.dem_sha256)
    recorded = (
        payload.get("config_hash"),
        payload.get("source_hash"),
        payload.get("decoder_dem_sha256"),
    )
    return payload if recorded == current else None


def _pilot_gate(
    config: ResidualConfig,
    frozen: _Frozen,
    *,
    pilot_rows: int,
    fetch_cohort: FetchCohort | None,
    log: Log,
) -> dict[str, Any]:
    """Require a pilot record for this configuration; run one inline when absent.

    A record keyed on a different configuration hash, source or decoder is not a verdict
    about this run (:func:`_load_pilot` returns ``None`` for it) and is replaced.
    """
    record = _load_pilot(config, frozen)
    if record is None:
        log("pilot gate: no pilot record for this configuration; running the pilot inline")
        record = pilot(config, generated_rows=pilot_rows, fetch_cohort=fetch_cohort, log=log)
    if record.get("resources_sufficient") is not True:
        raise PilotGateError(
            f"{config.dataset_name}: pilot.json records resources_sufficient="
            f"{record.get('resources_sufficient')!r}; the full run is not launched"
        )
    projected = int(record["projection"]["bytes"])
    free = _free_disk_bytes(config.output_root)
    if free < projected:
        raise PilotGateError(
            f"{config.dataset_name}: free disk {free} bytes is below the projected "
            f"{projected} bytes at launch"
        )
    return record


def _discard_checkpoints(root: Path) -> None:
    """Remove chunk files and the index only; inventory/pilot records stay."""
    if not root.is_dir():
        return
    for path in root.iterdir():
        if path.is_file() and (
            path.name == INDEX_FILENAME or path.suffix == CHUNK_SUFFIX or path.name.endswith(".tmp")
        ):
            path.unlink()


def _stage_a(
    config: ResidualConfig, frozen: _Frozen, store: ChunkStore, total_rows: int, log: Log
) -> None:
    """Rows -> raw checkpoint chunks. Completed chunks are skipped, never rewritten."""
    completed = store.contiguous_completed(stage="raw")
    if len(completed) and store.completed_raw()[completed[-1]].row_end >= total_rows:
        log(f"stage A: resumed; all {total_rows} raw rows already checkpointed")
        return
    if completed:
        log(f"stage A: resuming after {len(completed)} completed raw chunk(s)")
    rows_seen = 0
    for index, (dets, obs) in enumerate(
        _regroup(_row_stream(config, frozen.source), config.pipeline.checkpoint_rows)
    ):
        n_rows = int(dets.shape[0])
        if rows_seen + n_rows > total_rows:
            raise ValueError(
                f"the row stream carries more than the {total_rows} rows the dataset declares"
            )
        if index in completed:
            existing = store.completed_raw()[index]
            if (existing.row_start, existing.row_count) != (rows_seen, n_rows):
                raise ValueError(
                    f"chunk {index} covers rows [{existing.row_start}, {existing.row_end}) in "
                    f"the checkpoint but [{rows_seen}, {rows_seen + n_rows}) in the stream"
                )
        else:
            store.write_raw(index, rows_seen, dets, obs)
        rows_seen += n_rows
        log(f"stage A: raw chunk {index} rows {rows_seen}/{total_rows}")
    if rows_seen != total_rows:
        raise ValueError(f"the row stream ended after {rows_seen} rows; {total_rows} were declared")


def _check_source_prefix(
    config: ResidualConfig, frozen: _Frozen, store: ChunkStore, log: Log
) -> None:
    """Rows ``[0, source_shots)`` of the checkpointed stream must hash to the source.

    The call-size rule was applied at resolution; the content hash is the ground truth,
    and it is computed from the checkpoint files the build will publish, so a resumed
    run is held to the same proof as a fresh one.
    """
    if config.generation.mode is not GenerationMode.EXTEND:
        return
    if not config.generation.require_source_prefix:
        return
    identity = frozen.source.identity
    n_rows = identity.shots
    expected = identity.hashes["content_hash"]
    hasher = StreamingContentHasher()
    seen = 0
    for index in store.contiguous_completed(stage="raw"):
        if seen >= n_rows:
            break
        dets, obs = store.read_raw(index)
        take = min(int(dets.shape[0]), n_rows - seen)
        hasher.update(dets[:take], obs[:take])
        seen += take
    if seen != n_rows:
        raise ValueError(f"only {seen} checkpointed rows; {n_rows} needed for the prefix check")
    digest = hasher.hexdigest(n_rows, identity.n_detectors, identity.n_observables)
    if digest != expected:
        raise ValueError(
            f"generated rows 0..{n_rows - 1} hash to {digest}, not the source content_hash "
            f"{expected}; the extension does not reproduce its source and is not published"
        )
    log(f"prefix: rows 0..{n_rows - 1} reproduce the source content_hash {expected[:12]}...")


def _stage_b(config: ResidualConfig, frozen: _Frozen, store: ChunkStore, log: Log) -> None:
    """Per raw chunk: decode -> features (in ``feature_rows`` blocks) -> truth/pm_wrong."""
    block = config.pipeline.feature_rows
    raw = store.contiguous_completed(stage="raw")
    done = store.completed_features()
    for index in raw:
        if index in done:
            continue
        dets, obs = store.read_raw(index)
        n_rows = int(dets.shape[0])
        guesses: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        features: list[np.ndarray] = []
        for start in range(0, n_rows, block):
            part = dets[start : start + block]
            guess, weight = decode_chunk(frozen.model, part)
            features.append(extract_features(part, guess, weight, frozen.context))
            guesses.append(guess)
            weights.append(weight)
        guess_all = np.concatenate(guesses)
        weight_all = np.concatenate(weights)
        truth = truth_from_packed(obs, frozen.model.n_observables)
        wrong = pm_wrong(guess_all, truth)
        record = store.write_features(
            index, np.concatenate(features), guess_all, weight_all, truth, wrong
        )
        log(f"stage B: chunk {index} features written; pm_wrong {record.pm_wrong_count}/{n_rows}")


def _summary(
    config: ResidualConfig,
    frozen: _Frozen,
    *,
    n_rows: int,
    codes: np.ndarray,
    failures: int,
    sanity: dict[str, Any],
    pilot_record: dict[str, Any] | None,
) -> dict[str, Any]:
    identity = frozen.source.identity
    model = frozen.model
    name = config.dataset_name
    interval = clopper_pearson(failures, n_rows)
    generated = config.generation.mode is not GenerationMode.SOURCE_ROWS
    source_label = identity.paths.get("source") or identity.paths.get("table") or "unknown"
    limitations = [
        "pm_weight is PyMatching's summed matching-edge weight, not a physical fault count",
        "not a qecgen dataset: no exporter registration, no qecgen manifest, no Nexus "
        "compatibility claim",
    ]
    if bool(frozen.graph.boundary_adjacent.all()):
        limitations.append(
            "every detector node is boundary-adjacent in this matching graph, so "
            "frac_fired_boundary_adjacent equals frac_fired"
        )
    deviations = [
        "raw HDF5 arrays live under group /residual so qecgen.exporters.hdf5 reads the file "
        "as foreign rather than as an interrupted qecgen write",
    ]
    if (
        generated
        and identity.chunk_size is not None
        and config.generation.chunk_size != identity.chunk_size
    ):
        deviations.append(
            f"generation chunk_size {config.generation.chunk_size} (source recorded "
            f"{identity.chunk_size}) chosen by the call-size-sequence rule so the source rows "
            "are reproduced as the prefix"
        )
    pilot_summary = None
    if pilot_record is not None and pilot_record.get("existing_rows"):
        existing = pilot_record["existing_rows"]
        alignment = pilot_record.get("alignment", {})
        pilot_summary = {
            # Named so a reader can see the pilot numbers belong to this configuration;
            # _load_pilot guarantees it, the summary states it.
            "pilot_config_hash": pilot_record.get("config_hash"),
            "n_source_rows": existing["n_rows"],
            "pm_wrong_source_rows": existing["pm_wrong"],
            "pm_error_rate_source_rows": existing["pm_error_rate"],
            "alignment_concerns": alignment.get("concerns"),
            "expectation": pilot_record.get("expectation"),
        }
    return {
        "dataset_name": name,
        "additional": config.additional,
        "source": f"{identity.kind.value}: {Path(source_label).name}",
        "source_paths": identity.paths,
        "source_hashes": identity.hashes,
        "provenance_status": identity.dem_available,
        "provenance_limitations": identity.provenance_limitations,
        "distance": identity.distance,
        "rounds": identity.rounds,
        "basis": identity.basis,
        "orientation": identity.noise_parameters.get("orientation"),
        "n_detectors": identity.n_detectors,
        "n_observables": identity.n_observables,
        "noise_model": identity.noise_model,
        "noise_parameters": identity.noise_parameters,
        "n_runs": n_rows,
        "seed": config.generation.seed if generated else None,
        "chunk_size": config.generation.chunk_size if generated else None,
        "generation_mode": config.generation.mode.value,
        "versions": library_versions(),
        "decoder_method": str(model.provenance.get("method")),
        "decoder_source": identity.dem_available,
        "matching_mode": str(model.provenance.get("matching_mode")),
        "decoder_path": f"{name}_decoder.dem",
        "decoder_sha256": model.dem_sha256,
        "config_hash": config_hash(config),
        "schema_version": SCHEMA_VERSION,
        "split_method": config.splits.method.value,
        "split_seed": config.splits.seed,
        "split_fractions": dict(config.splits.fractions),
        "split_counts": {
            split: int((codes == SPLIT_CODES[split]).sum())
            for split in SPLIT_NAMES
            if split in config.splits.fractions
        },
        "pm_failures": failures,
        "pm_error_rate": interval.point,
        "pm_ci_low": interval.low,
        "pm_ci_high": interval.high,
        "always_zero_accuracy": 1.0 - interval.point,
        "sanity": sanity,
        "pilot": pilot_summary,
        "limitations": limitations,
        "deviations": deviations,
        "built_at": _now(),
    }


def _stage_c(
    config: ResidualConfig,
    frozen: _Frozen,
    store: ChunkStore,
    total_rows: int,
    pilot_record: dict[str, Any] | None,
    log: Log,
) -> Path:
    """Assemble, validate and publish the artifact set atomically."""
    name = config.dataset_name
    indices = store.contiguous_completed(stage="features")
    covered = sum(store.completed_features()[i].row_count for i in indices)
    if covered != total_rows:
        raise ValueError(f"feature chunks cover {covered} rows; the dataset declares {total_rows}")
    codes = assign_splits(
        total_rows, config.splits.method.value, dict(config.splits.fractions), config.splits.seed
    )
    failures = int(store.aggregate()["pm_wrong_total"])
    destination = dataset_dir(config)

    def feature_chunks() -> Iterator[dict[str, np.ndarray]]:
        for index in indices:
            record = store.completed_features()[index]
            chunk = store.read_features(index)
            chunk["run_id"] = np.arange(record.row_start, record.row_end, dtype=np.int64)
            yield chunk

    def raw_chunks() -> Iterator[RowChunk]:
        for index in indices:
            yield store.read_raw(index)

    def run_ids() -> Iterator[np.ndarray]:
        for index in indices:
            record = store.completed_features()[index]
            yield np.arange(record.row_start, record.row_end, dtype=np.int64)

    with run.staged(destination) as staging:
        scratch = staging.scratch
        features_csv = scratch / f"{name}_features.csv"
        written = write_features_csv(features_csv, feature_chunks(), codes)
        log(f"stage C: features CSV {written} rows")
        write_raw_hdf5(
            scratch / f"{name}_raw.h5",
            raw_chunks(),
            run_ids(),
            codes,
            {
                "n_detectors": frozen.n_detectors,
                "n_observables": frozen.model.n_observables,
                "source_content_hash": _source_hash(frozen.source),
                "config_hash": config_hash(config),
                "decoder_dem_sha256": frozen.model.dem_sha256,
            },
        )
        write_decoder_files(scratch, name, frozen.model, frozen.graph, slices=frozen.slices)
        write_json(scratch / f"{name}_resolved_config.json", resolved_dict(config))

        sanity: dict[str, Any]
        if not config.sanity_model.enabled:
            sanity = {
                "skipped_reason": "disabled in the configuration (sanity_model.enabled=false)"
            }
        else:
            try:
                sanity = run_sanity_models(features_csv, seed=config.sanity_model.seed)
            except SanityModelUnavailableError as error:
                sanity = {"skipped_reason": str(error)}
        write_json(scratch / f"{name}_sanity.json", sanity)

        summary = _summary(
            config,
            frozen,
            n_rows=total_rows,
            codes=codes,
            failures=failures,
            sanity=sanity,
            pilot_record=pilot_record,
        )
        summary_path = scratch / f"{name}_summary.json"
        write_json(summary_path, summary)
        note = render_note(summary, summary_sha256=file_sha256(summary_path))
        (scratch / f"{name}_note.md").write_text(note, encoding="utf-8")

        log("stage C: validating the staged artifact set")
        report = validate_dataset_dir(scratch, spot_rows=config.pipeline.spot_check_rows)
        write_json(scratch / f"{name}_validation.json", report.to_dict())
        if not report.ok:
            raise ValidationFailedError(report)
        log("stage C: validation ok; publishing")
    return destination


def build(
    config: ResidualConfig,
    *,
    resume: bool = True,
    fresh: bool = False,
    skip_pilot_gate: bool = False,
    pilot_rows: int = 10_000,
    fetch_cohort: FetchCohort | None = None,
    log: Log = _silent,
) -> Path:
    """Build and publish one residual dataset; returns the published directory.

    Order matters and is the module's contract: resolve (a blocked source raises before
    any directory exists), gate on the pilot, checkpoint stages A and B, publish stage C
    inside ``run.staged``, then re-render ``MANIFEST.md``. ``fresh`` (or ``resume=False``)
    discards this dataset's chunk files first; a checkpoint written under a different
    identity is refused by :class:`~qecgen.residual.checkpoint.CheckpointIdentityError`
    naming the differing fields, never merged.
    """
    frozen = _freeze(config, fetch_cohort)
    total_rows = _total_rows(config, frozen.source)
    if total_rows < 1:
        raise ValueError(f"{config.dataset_name} would have {total_rows} rows")
    log(
        f"build: {config.dataset_name} ({total_rows} rows, {frozen.n_detectors} detectors, "
        f"decoder {frozen.model.kind.value})"
    )

    pilot_record: dict[str, Any] | None
    if config.generation.mode is GenerationMode.SOURCE_ROWS:
        pilot_record = _load_pilot(config, frozen)
    elif skip_pilot_gate:
        log("pilot gate: skipped (--skip-pilot-gate)")
        pilot_record = _load_pilot(config, frozen)
    else:
        pilot_record = _pilot_gate(
            config, frozen, pilot_rows=pilot_rows, fetch_cohort=fetch_cohort, log=log
        )
    if pilot_record is not None and pilot_record.get("existing_rows"):
        existing = pilot_record["existing_rows"]
        log(
            f"pm_wrong: {existing['pm_wrong']} / {existing['n_rows']} "
            f"({existing['pm_wrong_fraction']:.6f}) on the existing source rows (pilot)"
        )

    root = checkpoint_root(config)
    if fresh or not resume:
        _discard_checkpoints(root)
        log("checkpoints: discarded (fresh build)")
    identity = CheckpointIdentity(
        config_hash=config_hash(config),
        source_hash=_source_hash(frozen.source),
        decoder_dem_sha256=frozen.model.dem_sha256,
        graph_digest=frozen.graph.digest(),
        schema_version=SCHEMA_VERSION,
        versions=library_versions(),
        seed=config.generation.seed,
        chunk_size=config.generation.chunk_size,
        shots=total_rows,
    )
    store = ChunkStore(root, identity)
    if store.completed_raw():
        aggregate = store.aggregate()
        log(
            f"checkpoints: resuming from {aggregate['raw_rows']} raw and "
            f"{aggregate['feature_rows']} featured rows"
        )

    _stage_a(config, frozen, store, total_rows, log)
    _check_source_prefix(config, frozen, store, log)
    _stage_b(config, frozen, store, log)
    failures = int(store.aggregate()["pm_wrong_total"])
    log(f"pm_wrong: {failures} / {total_rows} ({failures / total_rows:.6f}) on the full dataset")

    published = _stage_c(config, frozen, store, total_rows, pilot_record, log)
    blocked = root / BLOCKED_FILENAME
    if blocked.exists():
        blocked.unlink()
    write_manifest(config.output_root)
    log(f"published {published}")
    return published


# ---------------------------------------------------------------------------
# manifest and build-all


def _completed_summaries(output_root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    if not output_root.is_dir():
        return summaries
    for child in sorted(output_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        candidates = sorted(child.glob("*_summary.json"))
        if len(candidates) != 1:
            continue
        payload = json.loads(candidates[0].read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("dataset_name") == child.name:
            summaries.append(payload)
    summaries.sort(key=lambda s: (bool(s.get("additional", False)), str(s["dataset_name"])))
    return summaries


def _blocked_records(output_root: Path, completed: set[str]) -> list[dict[str, Any]]:
    blocked: list[dict[str, Any]] = []
    checkpoints = output_root / CHECKPOINT_DIRNAME
    if not checkpoints.is_dir():
        return blocked
    for child in sorted(checkpoints.iterdir()):
        path = child / BLOCKED_FILENAME
        if child.name in completed or not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "reason" in payload:
            payload.setdefault("dataset_name", child.name)
            blocked.append(payload)
    blocked.sort(key=lambda b: (bool(b.get("additional", False)), str(b["dataset_name"])))
    return blocked


def write_manifest(output_root: Path) -> Path:
    """Render ``MANIFEST.md`` from every published summary and every blocked record.

    Inside ``run.staged(output_root)`` so a crash mid-write never leaves a truncated
    manifest under the final name beside intact datasets.
    """
    summaries = _completed_summaries(output_root)
    blocked = _blocked_records(output_root, {str(s["dataset_name"]) for s in summaries})
    text = render_manifest(summaries, blocked)
    with run.staged(output_root) as staging:
        (staging.scratch / MANIFEST_FILENAME).write_text(text, encoding="utf-8")
    return output_root / MANIFEST_FILENAME


def _build_key(config: ResidualConfig) -> tuple[int, int, str]:
    rank = (
        BUILD_ORDER.index(config.dataset_name)
        if config.dataset_name in BUILD_ORDER
        else len(BUILD_ORDER)
    )
    return (int(config.additional), rank, config.dataset_name)


def build_all(
    config_dir: Path,
    *,
    repo_root: Path,
    fresh: bool = False,
    skip_pilot_gate: bool = False,
    pilot_rows: int = 10_000,
    fetch_cohort: FetchCohort | None = None,
    log: Log = _silent,
) -> list[dict[str, Any]]:
    """Build every ``*.json`` in ``config_dir``: required first, additional last.

    A blocked or failed dataset is recorded with its reason (and a ``blocked.json`` under
    its checkpoint directory so the manifest shows the row) and the loop continues; the
    manifest of every output root touched is re-rendered at the end. "Failed" is reserved
    for the pipeline's own refusals raised after resolution; anything else is a bug and
    propagates, because a manifest row cannot stand in for a traceback and the next
    dataset would be built on the same broken code.
    """
    paths = sorted(Path(config_dir).glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no *.json configurations in {config_dir}")
    configs = sorted((load_config(path, repo_root) for path in paths), key=_build_key)
    results: list[dict[str, Any]] = []
    roots: list[Path] = []
    for config in configs:
        if config.output_root not in roots:
            roots.append(config.output_root)
        entry: dict[str, Any] = {
            "dataset_name": config.dataset_name,
            "additional": config.additional,
            "config_path": None if config.config_path is None else str(config.config_path),
        }
        try:
            published = build(
                config,
                fresh=fresh,
                skip_pilot_gate=skip_pilot_gate,
                pilot_rows=pilot_rows,
                fetch_cohort=fetch_cohort,
                log=log,
            )
        except SourceBlockedError as error:
            entry.update(status="blocked", reason=error.reason)
            log(f"blocked: {config.dataset_name}: {error.reason}")
        except (
            PilotGateError,
            ValidationFailedError,
            CheckpointError,
            ValueError,
            OSError,
        ) as error:
            entry.update(status="failed", reason=f"build failed: {type(error).__name__}: {error}")
            log(f"failed: {config.dataset_name}: {type(error).__name__}: {error}")
        else:
            entry.update(status="completed", path=str(published))
        if entry["status"] != "completed":
            _write_record(
                checkpoint_root(config) / BLOCKED_FILENAME,
                {
                    "dataset_name": config.dataset_name,
                    "additional": config.additional,
                    "source": config.source.kind.value,
                    "status": entry["status"],
                    "reason": entry["reason"],
                    "recorded_at": _now(),
                },
            )
        results.append(entry)
    for root in roots:
        write_manifest(root)
    return results


def as_json(results: Sequence[dict[str, Any]]) -> str:
    """Compact rendering of ``build_all`` results for the terminal log."""
    return json.dumps(list(results), indent=2, sort_keys=True, allow_nan=False)
