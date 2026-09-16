"""Validation of one published residual dataset: fifteen named checks covering the brief's
fourteen assertions, plus spot checks.

Everything the pipeline writes is re-derived here from the published files alone and
compared against what those files claim. Three traps decide the shape of the module:

* **A check is a verdict, not an exception.** Every one of the brief's fourteen assertions
  is reported by name with a ``passed`` flag and a detail string, and a failure inside one
  check never hides the others: a reordered CSV column must show up as ``schema_exact``
  failing while ``pm_wrong_consistent`` still passes, because columns are located by
  *name*. A validator that raised on the first problem would report one defect per run
  and a reader could not tell "one column moved" from "the file is garbage". Only a
  missing artifact raises, because nothing can be validated without it.
* **The little-endian claim is tested by a check that big-endian data cannot pass.** The
  ``bit_order`` attribute is a statement, not evidence. The spot rows are unpacked with
  :func:`qecgen.sampling.unpack_bits`, decoded again and re-featured through the pure-
  Python reference, and must reproduce the CSV; then at least one row whose packed bytes
  are *asymmetric* under bit reversal is unpacked with NumPy's default order and must
  **not** reproduce it. A byte's popcount survives reversal, so ``n_fired_total`` alone
  cannot tell the orders apart; the comparison covers PyMatching's guess and weight and
  every feature, and if no randomly chosen row distinguishes them the raw file is scanned
  for one that does, so the check fails only when the dataset genuinely cannot witness
  its own bit order.
* **Recomputed, never trusted.** The source content hash is recomputed by streaming the
  source file again; the configuration hash is recomputed from the resolved config; the
  decoder is rebuilt from the published ``.dem`` text and, by default, *also* from the
  resolved configuration so the two constructions are shown to hash alike; the note's
  numbers are parsed back out of the rendered text and compared with counts taken from
  the CSV. A recorded value is only ever the *expected* side of a comparison.
"""

from __future__ import annotations

import array
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pymatching
import stim

from qecgen.dataset import StreamingContentHasher, dem_digest
from qecgen.exporters.ml_csv import read_manifest_only
from qecgen.hardware import load_willow_derived
from qecgen.qa import clopper_pearson
from qecgen.residual import SCHEMA_VERSION
from qecgen.residual.config import (
    DecoderKind,
    GenerationMode,
    ResidualConfig,
    SourceKind,
    config_hash,
    from_resolved_dict,
)
from qecgen.residual.features import (
    ALL_COLUMNS,
    FEATURE_COLUMNS,
    INTEGER_COLUMNS,
    FeatureContext,
    audit_feature_columns,
    audit_feature_inputs,
    extract_features_reference,
)
from qecgen.residual.graph import summarise_graph, time_slices
from qecgen.residual.report import NOTE_FIELDS, SUMMARY_SHA_LABEL, parse_note_fields
from qecgen.residual.sources import iter_ml_csv_rows, resolve_source
from qecgen.residual.splits import SPLIT_CODES, SPLIT_NAMES
from qecgen.residual.writers import FEATURE_CONTEXT_FIELDS, RAW_GROUP, file_sha256
from qecgen.sampling import packed_width, unpack_bits

__all__ = [
    "CHECK_NAMES",
    "Check",
    "ValidationReport",
    "validate_dataset_dir",
]

CHECK_NAMES: tuple[str, ...] = (
    "row_count_csv_equals_raw",
    "run_id_unique_contiguous_identical",
    "raw_widths_match_true_widths",
    "bit_packing_little_endian",
    "binary_columns",
    "pm_wrong_consistent",
    "batch_vs_single_decode",
    "no_nan_inf",
    "schema_exact",
    "fractions_in_range",
    "note_matches_data",
    "checksums_match",
    "source_prefix_matches",
    "feature_inputs_audit",
    "no_fit_on_held_out",
)
"""Fifteen named checks covering the brief's fourteen assertions, in the brief's order
(item 13 and 14 are ``feature_inputs_audit`` and ``no_fit_on_held_out``; the spot checks
feed items 4 and 7)."""

_SPOT_FLOAT_TOLERANCE = 1e-9
"""Spot-check floats: the reference extractor shares no arithmetic with the fast path."""

_EXACT_FLOAT_TOLERANCE = 1e-12
"""Note/summary floats: the same expressions on the same integers; anything larger than
round-off is a stale number."""

_RAW_BLOCK_ROWS = 8192
_BIT_ORDER_SCAN_ROWS = 4096
_SOURCE_STREAM_ROWS = 2000
_BINARY_COLUMNS: tuple[str, ...] = ("pm_guess", "truth", "pm_wrong")
_NUMERIC_COLUMNS: tuple[str, ...] = (*FEATURE_COLUMNS, "truth", "pm_wrong", "run_id")
_UNIT_INTERVAL_COLUMNS: tuple[str, ...] = (
    *(name for name in FEATURE_COLUMNS if name.startswith("frac_")),
    "round_of_max_fired",
    "fired_round_center",
)
_NON_NEGATIVE_COLUMNS: tuple[str, ...] = (
    *(name for name in FEATURE_COLUMNS if name.startswith("n_")),
    "fired_per_round_mean",
    "fired_per_round_max",
    "fired_per_round_std",
    "fired_per_round_range",
    "pm_weight",
    "pm_weight_per_fired",
)
_THIRD_PARTY_FITTING_KEYS: frozenset[str] = frozenset(
    {"method", "objective", "data", "overlap_with_this_cohort"}
)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ValidationReport:
    checks: tuple[Check, ...]
    ok: bool
    spot_checks: tuple[dict[str, Any], ...]
    dataset_name: str = ""

    def check(self, name: str) -> Check:
        for entry in self.checks:
            if entry.name == name:
                return entry
        raise KeyError(f"no check named {name!r}; known: {list(CHECK_NAMES)}")

    def to_dict(self) -> dict[str, Any]:
        """JSON form for ``<name>_validation.json`` (``write_json`` refuses NaN, so every
        number in the spot records is finite by construction)."""
        return {
            "dataset_name": self.dataset_name,
            "ok": self.ok,
            "checks": [dataclasses.asdict(c) for c in self.checks],
            "spot_checks": [dict(s) for s in self.spot_checks],
            "validated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        }


class _CheckFailedError(ValueError):
    """A check that did not pass; the message is the recorded detail."""


# ---------------------------------------------------------------------------
# Artifact loading


@dataclass(frozen=True)
class _Paths:
    name: str
    dataset_dir: Path
    features_csv: Path
    raw_h5: Path
    note: Path
    resolved_config: Path
    decoder_dem: Path
    decoder_metadata: Path
    summary: Path
    sanity: Path | None


def _locate(dataset_dir: Path) -> _Paths:
    """Name the artifact set from the one ``*_resolved_config.json`` in the directory.

    The directory name is not used: the pipeline validates inside a ``.qecgen-partial-*``
    scratch directory before publishing, where the directory name is a staging token.
    """
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"{dataset_dir} is not a directory")
    configs = sorted(dataset_dir.glob("*_resolved_config.json"))
    if len(configs) != 1:
        raise FileNotFoundError(
            f"{dataset_dir} holds {len(configs)} *_resolved_config.json files; exactly one "
            "names the dataset"
        )
    name = configs[0].name[: -len("_resolved_config.json")]
    sanity = dataset_dir / f"{name}_sanity.json"
    paths = _Paths(
        name=name,
        dataset_dir=dataset_dir,
        features_csv=dataset_dir / f"{name}_features.csv",
        raw_h5=dataset_dir / f"{name}_raw.h5",
        note=dataset_dir / f"{name}_note.md",
        resolved_config=configs[0],
        decoder_dem=dataset_dir / f"{name}_decoder.dem",
        decoder_metadata=dataset_dir / f"{name}_decoder_metadata.json",
        summary=dataset_dir / f"{name}_summary.json",
        sanity=sanity if sanity.is_file() else None,
    )
    for field in ("features_csv", "raw_h5", "note", "decoder_dem", "decoder_metadata", "summary"):
        path = getattr(paths, field)
        if not Path(path).is_file():
            raise FileNotFoundError(f"{field} artifact {path} is missing; nothing to validate")
    return paths


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return payload


class _FeatureTable:
    """The feature CSV, parsed once by column *name* into flat float64 storage.

    Columns are found by name rather than position so that a reordered header fails the
    schema check and nothing else. Integer columns are additionally screened as text: a
    ``1.0`` in ``pm_wrong`` parses to the right float and would otherwise pass every
    numeric comparison while breaking the writer's formatting contract. Values go into an
    ``array.array`` as they are read so a 304,000-row table never exists as Python
    strings.
    """

    def __init__(self, path: Path) -> None:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header is None:
                raise ValueError(f"{path} is empty")
            self.header: list[str] = list(header)
            index: dict[str, int] = {}
            for position, name in enumerate(self.header):
                index.setdefault(name, position)
            numeric = [name for name in _NUMERIC_COLUMNS if name in index]
            numeric_at = [index[name] for name in numeric]
            integer_at = [
                (slot, name) for slot, name in enumerate(numeric) if name in INTEGER_COLUMNS
            ]
            split_at = index.get("split")
            buffer = array.array("d")
            splits: list[str] = []
            bad_integer: set[str] = set()
            n_rows = 0
            for row in reader:
                if not row:
                    continue
                if len(row) != len(self.header):
                    raise ValueError(
                        f"{path}:{reader.line_num}: {len(row)} cells but the header has "
                        f"{len(self.header)}"
                    )
                for slot, name in integer_at:
                    if not row[numeric_at[slot]].lstrip("-").isdigit():
                        bad_integer.add(name)
                buffer.extend(float(row[at]) for at in numeric_at)
                if split_at is not None:
                    splits.append(row[split_at])
                n_rows += 1
        matrix = np.frombuffer(buffer, dtype=np.float64).reshape(n_rows, len(numeric))
        self._columns: dict[str, np.ndarray] = {
            name: matrix[:, slot] for slot, name in enumerate(numeric)
        }
        self.split_names: list[str] | None = splits if split_at is not None else None
        self.n_rows = n_rows
        self.bad_integer_columns: frozenset[str] = frozenset(bad_integer)

    def column(self, name: str) -> np.ndarray:
        try:
            return self._columns[name]
        except KeyError:
            raise _CheckFailedError(f"the feature CSV has no column {name!r}") from None


def _attr_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class _RawFile:
    """The raw HDF5, with the detector matrix left on disk and streamed in blocks."""

    def __init__(self, path: Path) -> None:
        self.handle = h5py.File(path, "r")
        try:
            self.attrs: dict[str, Any] = {
                str(key): _attr_value(value) for key, value in self.handle.attrs.items()
            }
            if RAW_GROUP not in self.handle:
                raise ValueError(f"{path} has no /{RAW_GROUP} group; not a residual raw file")
            group = self.handle[RAW_GROUP]
            self.detectors = group["detectors"]
            self.observables = np.asarray(group["observables"], dtype=np.uint8)
            self.run_id = np.asarray(group["run_id"], dtype=np.int64)
            self.split = np.asarray(group["split"], dtype=np.int8)
            shape = tuple(int(s) for s in self.detectors.shape)
            if len(shape) != 2:
                raise ValueError(f"raw detectors must be 2-D, got shape {shape}")
            self.n_rows, self.det_width = shape
            if self.observables.ndim != 2:
                raise ValueError(f"raw observables must be 2-D, got shape {self.observables.shape}")
        except Exception:
            self.handle.close()
            raise

    def close(self) -> None:
        self.handle.close()

    def blocks(self, stop: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
        end = self.n_rows if stop is None else min(stop, self.n_rows)
        for start in range(0, end, _RAW_BLOCK_ROWS):
            block_end = min(start + _RAW_BLOCK_ROWS, end)
            yield start, np.asarray(self.detectors[start:block_end], dtype=np.uint8)

    def row(self, index: int) -> np.ndarray:
        return np.asarray(self.detectors[index], dtype=np.uint8).reshape(1, self.det_width)


@dataclass(frozen=True)
class _DecoderState:
    dem: stim.DetectorErrorModel
    dem_text: str
    dem_sha256: str
    matching: pymatching.Matching
    circuit: stim.Circuit
    context: FeatureContext
    detail: str


def _circuit_from_dem_coordinates(dem: stim.DetectorErrorModel) -> stim.Circuit:
    """A coordinate-only circuit: one measurement and one ``DETECTOR`` per DEM detector.

    Used when the configuration is not to be re-resolved: the DEM is the model of record
    and Stim copies the circuit's detector coordinates into it, so its coordinates define
    the same time slices. A detector without coordinates ends up without any, and
    :func:`time_slices` refuses it rather than filing it under slice 0.
    """
    coordinates = dem.get_detector_coordinates()
    circuit = stim.Circuit()
    for detector in range(dem.num_detectors):
        circuit.append("M", [0])
        circuit.append("DETECTOR", [stim.target_rec(-1)], list(coordinates.get(detector, [])))
    return circuit


def _require_coordinates_equal(dem: stim.DetectorErrorModel, circuit: stim.Circuit) -> None:
    dem_coordinates = dem.get_detector_coordinates()
    circuit_coordinates = circuit.get_detector_coordinates()
    if dem.num_detectors != circuit.num_detectors:
        raise ValueError(
            f"published DEM has {dem.num_detectors} detectors, the circuit of record "
            f"{circuit.num_detectors}"
        )
    mismatched = [
        d
        for d in range(dem.num_detectors)
        if list(dem_coordinates.get(d, [])) != list(circuit_coordinates.get(d, []))
    ]
    if mismatched:
        raise ValueError(
            f"detector coordinates differ between the published DEM and the circuit of "
            f"record for {len(mismatched)} detector(s) (first: D{mismatched[0]})"
        )


def _parse_split_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for part in text.split(","):
        name, _, value = part.strip().partition("=")
        if not name or not value.strip().lstrip("-").isdigit():
            raise _CheckFailedError(f"note split sizes {text!r} do not parse as name=count pairs")
        counts[name] = int(value)
    return counts


def _nonzero(counts: Mapping[str, Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in counts.items() if int(v) != 0}


def _close(a: float, b: float, tolerance: float) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance


# ---------------------------------------------------------------------------
# The validator


class _Validator:
    def __init__(
        self, paths: _Paths, *, rebuild_decoder: bool, spot_rows: int, spot_seed: int
    ) -> None:
        self.paths = paths
        self.rebuild_decoder = rebuild_decoder
        self.spot_rows = spot_rows
        self.spot_seed = spot_seed
        self.table = _FeatureTable(paths.features_csv)
        self.raw = _RawFile(paths.raw_h5)
        try:
            self.summary = _load_json(paths.summary)
            self.metadata = _load_json(paths.decoder_metadata)
            self.sanity: dict[str, Any] | None = (
                None if paths.sanity is None else _load_json(paths.sanity)
            )
            self.note_fields = parse_note_fields(paths.note.read_text(encoding="utf-8"))
        except Exception:
            self.raw.close()
            raise
        self.config: ResidualConfig | None = None
        self.config_error: str | None = None
        try:
            self.config = from_resolved_dict(_load_json(paths.resolved_config))
        except Exception as error:
            self.config_error = f"{type(error).__name__}: {error}"
        self._decoder: _DecoderState | None = None
        self._decoder_error: str | None = None
        self._spot: dict[str, Any] | None = None
        self._spot_error: str | None = None

    def close(self) -> None:
        self.raw.close()

    # -- shared state -------------------------------------------------------

    def _require_config(self) -> ResidualConfig:
        if self.config is None:
            raise _CheckFailedError(
                f"resolved configuration could not be loaded: {self.config_error}"
            )
        return self.config

    def _n_detectors(self) -> int:
        return int(self.raw.attrs["n_detectors"])

    def _circuit_of_record(
        self, dem: stim.DetectorErrorModel, published_sha: str
    ) -> tuple[stim.Circuit, str]:
        """Rebuild the decoder from the resolved configuration and hand back its circuit.

        Simulated kinds go through :func:`resolve_source`, which rebuilds the noisy circuit
        and its DEM; that DEM must hash to the published ``.dem`` or the published model is
        not the one the configuration describes. The Willow kind never resolves (it would
        reach for the network): its circuit of record is the cached ``circuit_ideal.stim``
        or the mirror file, either verified against the configured sha256.
        """
        config = self._require_config()
        if config.source.kind is SourceKind.HARDWARE_WILLOW:
            source = config.source
            if source.circuit is None or source.expected is None or source.zenodo is None:
                raise _CheckFailedError(
                    "hardware_willow configuration lacks circuit/expected/zenodo"
                )
            cached = source.zenodo.cache_dir / "circuit_ideal.stim"
            path = cached if cached.is_file() else source.circuit
            text = path.read_bytes()
            digest = hashlib.sha256(text).hexdigest()
            if digest != source.expected.circuit_sha256:
                raise _CheckFailedError(
                    f"circuit of record {path} hashes to {digest}, expected "
                    f"{source.expected.circuit_sha256}"
                )
            if config.decoder.expected_sha256 != published_sha:
                raise _CheckFailedError(
                    f"published .dem sha256 {published_sha} differs from the configured "
                    f"official member sha256 {config.decoder.expected_sha256}"
                )
            return stim.Circuit(text.decode("utf-8")), f"circuit of record {path} (sha256 verified)"
        resolved = resolve_source(config)
        if resolved.decoder.dem_sha256 != published_sha:
            raise _CheckFailedError(
                "decoder rebuilt from the resolved configuration hashes to "
                f"{resolved.decoder.dem_sha256}; the published .dem hashes to {published_sha}"
            )
        return (
            resolved.decoder.circuit,
            f"decoder rebuilt from the resolved configuration ({config.decoder.kind.value}) "
            "hashes to the published .dem",
        )

    def decoder(self) -> _DecoderState:
        if self._decoder is not None:
            return self._decoder
        if self._decoder_error is not None:
            raise _CheckFailedError(self._decoder_error)
        try:
            dem_bytes = self.paths.decoder_dem.read_bytes()
            dem_text = dem_bytes.decode("utf-8")
            dem_sha = hashlib.sha256(dem_bytes).hexdigest()
            dem = stim.DetectorErrorModel(dem_text)
            matching = pymatching.Matching.from_detector_error_model(dem)
            if self.rebuild_decoder:
                circuit, detail = self._circuit_of_record(dem, dem_sha)
            else:
                circuit = _circuit_from_dem_coordinates(dem)
                detail = "coordinates taken from the published DEM (rebuild_decoder=False)"
            _require_coordinates_equal(dem, circuit)
            n_detectors = self._n_detectors()
            if dem.num_detectors != n_detectors:
                raise _CheckFailedError(
                    f"published DEM has {dem.num_detectors} detectors; the raw file declares "
                    f"{n_detectors}"
                )
            context = FeatureContext(
                n_detectors=n_detectors,
                slices=time_slices(circuit),
                graph=summarise_graph(dem, matching, n_detectors),
            )
            audit_feature_inputs(context)
        except Exception as error:
            self._decoder_error = f"decoder could not be rebuilt: {type(error).__name__}: {error}"
            raise _CheckFailedError(self._decoder_error) from error
        self._decoder = _DecoderState(
            dem=dem,
            dem_text=dem_text,
            dem_sha256=dem_sha,
            matching=matching,
            circuit=circuit,
            context=context,
            detail=detail,
        )
        return self._decoder

    def _csv_row(self, index: int) -> tuple[list[float], int, int, int]:
        features = [float(self.table.column(name)[index]) for name in FEATURE_COLUMNS]
        truth = int(self.table.column("truth")[index])
        wrong = int(self.table.column("pm_wrong")[index])
        run_id = int(self.table.column("run_id")[index])
        return features, truth, wrong, run_id

    def _decode_one(self, state: _DecoderState, bits: np.ndarray) -> tuple[int, float, list[float]]:
        prediction, weight = state.matching.decode(bits.astype(np.uint8), return_weight=True)
        guess = int(np.asarray(prediction).ravel()[0])
        reference = extract_features_reference(
            [int(b) for b in bits.tolist()], guess, float(weight), state.context
        )
        return guess, float(weight), reference

    @staticmethod
    def _compare_features(reference: list[float], observed: list[float]) -> tuple[bool, float]:
        worst = 0.0
        agree = True
        for name, expected, actual in zip(FEATURE_COLUMNS, reference, observed, strict=True):
            if name in INTEGER_COLUMNS:
                if round(expected) != round(actual) or actual != round(actual):
                    agree = False
                    worst = max(worst, abs(expected - actual))
            elif not _close(expected, actual, _SPOT_FLOAT_TOLERANCE):
                agree = False
                worst = max(worst, abs(expected - actual) if math.isfinite(actual) else math.inf)
            else:
                worst = max(worst, abs(expected - actual))
        return agree, worst

    def _spot_record(self, state: _DecoderState, index: int, selected_for: str) -> dict[str, Any]:
        n_detectors = state.context.n_detectors
        packed = self.raw.row(index)
        little = unpack_bits(packed, n_detectors)[0]
        big = np.unpackbits(packed, axis=1, count=n_detectors, bitorder="big")[0].astype(bool)
        guess, weight, reference = self._decode_one(state, little)
        csv_features, csv_truth, csv_wrong, csv_run_id = self._csv_row(index)
        csv_guess = round(csv_features[FEATURE_COLUMNS.index("pm_guess")])
        csv_weight = csv_features[FEATURE_COLUMNS.index("pm_weight")]
        features_match, worst = self._compare_features(reference, csv_features)
        raw_truth = int(unpack_bits(self.raw.observables[index : index + 1], 1)[0, 0])
        has_asymmetric = not np.array_equal(little, big)
        big_differs: bool | None = None
        if has_asymmetric:
            big_guess, big_weight, big_reference = self._decode_one(state, big)
            big_match, _ = self._compare_features(big_reference, csv_features)
            big_differs = (
                big_guess != csv_guess
                or not _close(big_weight, csv_weight, _SPOT_FLOAT_TOLERANCE)
                or not big_match
            )
        ok = (
            features_match
            and guess == csv_guess
            and _close(weight, csv_weight, _SPOT_FLOAT_TOLERANCE)
            and raw_truth == csv_truth
            and csv_wrong == int(guess != csv_truth)
            and csv_run_id == index
            and int(self.raw.run_id[index]) == index
        )
        return {
            "row": index,
            "selected_for": selected_for,
            "run_id_csv": csv_run_id,
            "run_id_raw": int(self.raw.run_id[index]),
            "n_fired": int(little.sum()),
            "pm_guess_single": guess,
            "pm_guess_csv": csv_guess,
            "pm_weight_single": weight,
            "pm_weight_csv": csv_weight,
            "truth_raw": raw_truth,
            "truth_csv": csv_truth,
            "pm_wrong_csv": csv_wrong,
            "features_match_reference": features_match,
            "max_abs_feature_diff": worst if math.isfinite(worst) else None,
            "has_asymmetric_byte": has_asymmetric,
            "big_endian_differs": big_differs,
            "ok": ok,
        }

    def spot(self) -> dict[str, Any]:
        """Decode and re-feature the spot rows once; both bit-order checks read this."""
        if self._spot is not None:
            return self._spot
        if self._spot_error is not None:
            raise _CheckFailedError(self._spot_error)
        try:
            state = self.decoder()
            n_rows = min(self.raw.n_rows, self.table.n_rows)
            if n_rows == 0:
                raise _CheckFailedError("no rows to spot-check")
            rng = np.random.default_rng(np.random.SeedSequence(self.spot_seed))
            chosen = sorted(
                int(i) for i in rng.choice(n_rows, size=min(self.spot_rows, n_rows), replace=False)
            )
            records = [self._spot_record(state, index, "random") for index in chosen]
            if not any(r["big_endian_differs"] for r in records):
                extra = self._find_bit_order_witness(state, set(chosen), n_rows)
                if extra is not None:
                    records.append(extra)
            packed = np.concatenate([self.raw.row(r["row"]) for r in records])
            predictions, weights = state.matching.decode_batch(
                packed,
                return_weights=True,
                bit_packed_shots=True,
                bit_packed_predictions=True,
                enable_correlations=False,
            )
            batch_guess = unpack_bits(np.asarray(predictions, dtype=np.uint8), 1)[:, 0]
            batch_weight = np.asarray(weights, dtype=np.float64).reshape(-1)
            batch_agrees = True
            for record, guess, weight in zip(records, batch_guess, batch_weight, strict=True):
                record["pm_guess_batch"] = int(guess)
                record["pm_weight_batch"] = float(weight)
                same = int(guess) == record["pm_guess_single"] and _close(
                    float(weight), float(record["pm_weight_single"]), _SPOT_FLOAT_TOLERANCE
                )
                record["batch_matches_single"] = same
                batch_agrees = batch_agrees and same
        except _CheckFailedError as error:
            self._spot_error = str(error)
            raise
        except Exception as error:
            self._spot_error = f"spot checks could not run: {type(error).__name__}: {error}"
            raise _CheckFailedError(self._spot_error) from error
        self._spot = {"records": records, "batch_agrees": batch_agrees}
        return self._spot

    def _find_bit_order_witness(
        self, state: _DecoderState, taken: set[int], n_rows: int
    ) -> dict[str, Any] | None:
        """Scan the first rows for one that tells the two bit orders apart.

        A row qualifies when its big-endian reading fails to reproduce the CSV (the
        witness a correct dataset provides) *or* when its little-endian reading fails to
        (direct evidence against the dataset); either way the row is worth recording.
        """
        n_detectors = state.context.n_detectors
        for start, block in self.raw.blocks(stop=min(n_rows, _BIT_ORDER_SCAN_ROWS)):
            little = unpack_bits(block, n_detectors)
            big = np.unpackbits(block, axis=1, count=n_detectors, bitorder="big").astype(bool)
            candidates = np.flatnonzero((little != big).any(axis=1))
            for offset in candidates.tolist():
                index = start + int(offset)
                if index in taken:
                    continue
                record = self._spot_record(state, index, "bit_order_witness")
                if record["big_endian_differs"] or not record["ok"]:
                    return record
        return None

    def _recomputed_summary(self) -> dict[str, Any]:
        wrong = self.table.column("pm_wrong")
        n_rows = int(wrong.shape[0])
        if n_rows == 0:
            raise _CheckFailedError("the feature CSV has no rows")
        failures = int(np.rint(wrong).sum())
        interval = clopper_pearson(failures, n_rows)
        if self.table.split_names is None:
            raise _CheckFailedError("the feature CSV has no split column")
        counts = {name: self.table.split_names.count(name) for name in SPLIT_NAMES}
        return {
            "n_runs": n_rows,
            "pm_failures": failures,
            "pm_error_rate": interval.point,
            "pm_ci_low": interval.low,
            "pm_ci_high": interval.high,
            "always_zero_accuracy": 1.0 - interval.point,
            "split_counts": _nonzero(counts),
        }

    def _source_digest(self) -> tuple[str, str, str]:
        """``(label, digest, how)``: the source identity recomputed from the source file."""
        config = self._require_config()
        source = config.source
        kind = source.kind
        if kind in (SourceKind.LEGACY_ML_CSV, SourceKind.DEVICE_ML_CSV):
            if source.path is None:
                raise _CheckFailedError("source.path is missing from the resolved configuration")
            manifest = read_manifest_only(source.path)
            recorded = str(manifest["content_hash"])
            if recorded != source.expected_content_hash:
                raise _CheckFailedError(
                    f"{source.path}: manifest content_hash {recorded} differs from the "
                    f"configured expected_content_hash {source.expected_content_hash}"
                )
            rows = 0
            for dets, _ in iter_ml_csv_rows(source.path, manifest, _SOURCE_STREAM_ROWS):
                rows += int(dets.shape[0])
            return "content_hash", recorded, f"streamed {rows} source rows and re-hashed them"
        if kind is SourceKind.DEVICE_CONFIG:
            if source.path is None:
                raise _CheckFailedError("source.path is missing from the resolved configuration")
            return "config_sha256", file_sha256(source.path), "sha256 of the device config file"
        if source.table is None or source.circuit is None or source.expected is None:
            raise _CheckFailedError("hardware_willow configuration lacks table/circuit/expected")
        table_sha = file_sha256(source.table)
        circuit_sha = file_sha256(source.circuit)
        if table_sha != source.expected.table_sha256:
            raise _CheckFailedError(
                f"{source.table}: sha256 {table_sha} differs from expected.table_sha256"
            )
        if circuit_sha != source.expected.circuit_sha256:
            raise _CheckFailedError(
                f"{source.circuit}: sha256 {circuit_sha} differs from expected.circuit_sha256"
            )
        return "table_sha256", table_sha, "sha256 of the mirror table and circuit files"

    def _raw_prefix_digest(self, n_rows: int) -> str:
        if n_rows > self.raw.n_rows:
            raise _CheckFailedError(
                f"the raw file holds {self.raw.n_rows} rows but the source has {n_rows}; "
                "the source cannot be its prefix"
            )
        hasher = StreamingContentHasher()
        for start, block in self.raw.blocks(stop=n_rows):
            hasher.update(block, self.raw.observables[start : start + block.shape[0]])
        return hasher.hexdigest(n_rows, self._n_detectors(), int(self.raw.attrs["n_observables"]))

    # -- the checks -----------------------------------------------------------

    def check_row_count_csv_equals_raw(self) -> str:
        counts = {
            "csv": self.table.n_rows,
            "raw_detectors": self.raw.n_rows,
            "raw_observables": int(self.raw.observables.shape[0]),
            "raw_run_id": int(self.raw.run_id.shape[0]),
            "raw_split": int(self.raw.split.shape[0]),
        }
        if len(set(counts.values())) != 1:
            raise _CheckFailedError(f"row counts disagree: {counts}")
        if self.table.n_rows == 0:
            raise _CheckFailedError("the dataset has no rows")
        return f"{self.table.n_rows} rows in every artifact"

    def check_run_id_unique_contiguous_identical(self) -> str:
        n_rows = self.table.n_rows
        expected = np.arange(n_rows, dtype=np.int64)
        csv_ids = self.table.column("run_id")
        if not np.array_equal(csv_ids, expected.astype(np.float64)):
            raise _CheckFailedError("CSV run_id is not the contiguous row index 0..n-1")
        if self.raw.run_id.shape != (n_rows,) or not np.array_equal(self.raw.run_id, expected):
            raise _CheckFailedError("raw run_id is not the contiguous row index 0..n-1")
        valid_codes = np.asarray(sorted(SPLIT_CODES.values()), dtype=np.int8)
        if self.raw.split.shape != (n_rows,) or not np.isin(self.raw.split, valid_codes).all():
            raise _CheckFailedError(
                "raw split codes are missing or outside the recorded code table"
            )
        recorded_codes = json.loads(str(self.raw.attrs.get("split_codes", "null")))
        if recorded_codes != SPLIT_CODES:
            raise _CheckFailedError(f"raw split_codes attribute {recorded_codes} != {SPLIT_CODES}")
        if self.table.split_names is None:
            raise _CheckFailedError("the feature CSV has no split column")
        raw_names = [SPLIT_NAMES[int(code)] for code in self.raw.split.tolist()]
        if raw_names != self.table.split_names:
            differing = sum(a != b for a, b in zip(raw_names, self.table.split_names, strict=True))
            raise _CheckFailedError(f"split differs between CSV and raw on {differing} row(s)")
        return f"run_id 0..{n_rows - 1} identical in CSV and raw; splits identical"

    def check_raw_widths_match_true_widths(self) -> str:
        attrs = self.raw.attrs
        n_detectors = attrs.get("n_detectors")
        n_observables = attrs.get("n_observables")
        if not isinstance(n_detectors, int) or n_detectors < 1:
            raise _CheckFailedError(f"raw attribute n_detectors is {n_detectors!r}")
        if not isinstance(n_observables, int) or n_observables != 1:
            raise _CheckFailedError(
                f"raw attribute n_observables is {n_observables!r}; schema v1 needs 1"
            )
        if self.raw.det_width != packed_width(n_detectors):
            raise _CheckFailedError(
                f"raw detectors are {self.raw.det_width} bytes wide; {n_detectors} detectors "
                f"pack to {packed_width(n_detectors)}"
            )
        if int(self.raw.observables.shape[1]) != packed_width(n_observables):
            raise _CheckFailedError(
                f"raw observables are {self.raw.observables.shape[1]} bytes wide"
            )
        declared = {
            "decoder_metadata.num_detectors": self.metadata.get("num_detectors"),
            "summary.n_detectors": self.summary.get("n_detectors"),
        }
        wrong = {k: v for k, v in declared.items() if v != n_detectors}
        if wrong:
            raise _CheckFailedError(f"true width {n_detectors} disagrees with {wrong}")
        if self.metadata.get("num_observables") != 1 or self.summary.get("n_observables") != 1:
            raise _CheckFailedError(
                "decoder metadata / summary do not declare exactly one observable"
            )
        if n_detectors % 8:
            mask = np.uint8((0xFF << (n_detectors % 8)) & 0xFF)
            for _, block in self.raw.blocks():
                if np.any(block[:, -1] & mask):
                    raise _CheckFailedError(
                        "detector bits beyond the true width are set; the packed width is "
                        "not the true width"
                    )
        if np.any(self.raw.observables & np.uint8(0xFE)):
            raise _CheckFailedError("observable bits beyond the single observable are set")
        return f"{n_detectors} detectors in {self.raw.det_width} bytes, 1 observable in 1 byte"

    def check_bit_packing_little_endian(self) -> str:
        bit_order = self.raw.attrs.get("bit_order")
        if bit_order != "little":
            raise _CheckFailedError(f"raw attribute bit_order is {bit_order!r}, not 'little'")
        spot = self.spot()
        records = spot["records"]
        bad = [r["row"] for r in records if not r["ok"]]
        if bad:
            raise _CheckFailedError(
                f"little-endian re-decode/re-feature does not reproduce the CSV on rows {bad}"
            )
        witnesses = [r["row"] for r in records if r["big_endian_differs"]]
        if not witnesses:
            raise _CheckFailedError(
                "no spot row with an asymmetric byte distinguishes big-endian from "
                "little-endian unpacking, so the little-endian claim has no witness"
            )
        return (
            f"bit_order=little; {len(records)} rows reproduce the CSV little-endian and "
            f"row(s) {witnesses} do not reproduce it big-endian"
        )

    def check_binary_columns(self) -> str:
        for name in _BINARY_COLUMNS:
            values = self.table.column(name)
            if name in self.table.bad_integer_columns:
                raise _CheckFailedError(f"column {name} holds a non-integer literal")
            if not np.isin(values, (0.0, 1.0)).all():
                raise _CheckFailedError(f"column {name} holds values outside {{0, 1}}")
        return "pm_guess, truth and pm_wrong are 0/1 on every row"

    def check_pm_wrong_consistent(self) -> str:
        guess = self.table.column("pm_guess")
        truth = self.table.column("truth")
        wrong = self.table.column("pm_wrong")
        expected = (guess != truth).astype(np.float64)
        mismatched = int(np.count_nonzero(wrong != expected))
        if mismatched:
            raise _CheckFailedError(f"pm_wrong != (pm_guess != truth) on {mismatched} row(s)")
        n_rows = min(self.table.n_rows, int(self.raw.observables.shape[0]))
        raw_truth = unpack_bits(self.raw.observables[:n_rows], 1)[:, 0].astype(np.float64)
        differing = int(np.count_nonzero(raw_truth != truth[:n_rows]))
        if differing or n_rows != self.table.n_rows:
            raise _CheckFailedError(
                f"CSV truth differs from the raw observable bit on {differing} row(s)"
            )
        return f"pm_wrong == (pm_guess != truth) and truth == raw observable on {n_rows} rows"

    def check_batch_vs_single_decode(self) -> str:
        spot = self.spot()
        records = spot["records"]
        if not spot["batch_agrees"]:
            bad = [r["row"] for r in records if not r.get("batch_matches_single", False)]
            raise _CheckFailedError(f"batched and single-shot decoding disagree on rows {bad}")
        wrong = [r["row"] for r in records if not r["ok"]]
        if wrong:
            raise _CheckFailedError(
                f"single-shot decoding does not reproduce the CSV on rows {wrong}"
            )
        return f"batched and single-shot PyMatching agree with the CSV on {len(records)} rows"

    def check_no_nan_inf(self) -> str:
        bad = [name for name in _NUMERIC_COLUMNS if not np.isfinite(self.table.column(name)).all()]
        if bad:
            raise _CheckFailedError(f"NaN or infinity in column(s) {bad}")
        return f"all {len(_NUMERIC_COLUMNS)} numeric columns finite"

    def check_schema_exact(self) -> str:
        audit_feature_columns(self.table.header)
        if self.table.bad_integer_columns:
            raise _CheckFailedError(
                f"integer column(s) {sorted(self.table.bad_integer_columns)} hold non-integer "
                "literals"
            )
        versions = {
            "raw.feature_schema_version": self.raw.attrs.get("feature_schema_version"),
            "decoder_metadata.feature_schema_version": self.metadata.get("feature_schema_version"),
            "summary.schema_version": self.summary.get("schema_version"),
        }
        if self.sanity is not None and "models" in self.sanity:
            versions["sanity.schema_version"] = self.sanity.get("schema_version")
        wrong = {k: v for k, v in versions.items() if v != SCHEMA_VERSION}
        if wrong:
            raise _CheckFailedError(f"schema version is {SCHEMA_VERSION}; recorded {wrong}")
        return f"header == schema v{SCHEMA_VERSION} ({len(ALL_COLUMNS)} columns); integer dtypes"

    def check_fractions_in_range(self) -> str:
        problems: list[str] = []
        for name in _UNIT_INTERVAL_COLUMNS:
            values = self.table.column(name)
            if np.any(values < 0.0) or np.any(values > 1.0):
                problems.append(f"{name} outside [0, 1]")
        spread = self.table.column("fired_round_spread")
        if np.any(spread < 0.0) or np.any(spread > 0.5):
            problems.append("fired_round_spread outside [0, 0.5]")
        for name in _NON_NEGATIVE_COLUMNS:
            if np.any(self.table.column(name) < 0.0):
                problems.append(f"{name} negative")
        if problems:
            raise _CheckFailedError("; ".join(problems))
        return "every normalised fraction in [0, 1], spread <= 0.5, counts and weights >= 0"

    def check_note_matches_data(self) -> str:
        recomputed = self._recomputed_summary()
        problems: list[str] = []
        summary_sha = file_sha256(self.paths.summary)
        note_sha = self.note_fields.get(SUMMARY_SHA_LABEL)
        if note_sha != summary_sha:
            problems.append(
                f"note {SUMMARY_SHA_LABEL} {note_sha} != sha256 of the summary file {summary_sha}"
            )
        for key in ("n_runs", "pm_failures"):
            if int(self.summary.get(key, -1)) != recomputed[key]:
                problems.append(f"summary {key} {self.summary.get(key)!r} != {recomputed[key]}")
        for key in ("pm_error_rate", "pm_ci_low", "pm_ci_high", "always_zero_accuracy"):
            value = self.summary.get(key)
            if not isinstance(value, int | float) or not _close(
                float(value), float(recomputed[key]), _EXACT_FLOAT_TOLERANCE
            ):
                problems.append(f"summary {key} {value!r} != {recomputed[key]!r}")
        if _nonzero(self.summary.get("split_counts", {})) != recomputed["split_counts"]:
            problems.append(
                f"summary split_counts {self.summary.get('split_counts')} != "
                f"{recomputed['split_counts']}"
            )
        fields = self.note_fields
        try:
            if int(fields[NOTE_FIELDS["n_runs"]]) != recomputed["n_runs"]:
                problems.append("note unique runs != CSV rows")
            if int(fields[NOTE_FIELDS["pm_failures"]]) != recomputed["pm_failures"]:
                problems.append(
                    f"note PyMatching failures {fields[NOTE_FIELDS['pm_failures']]} != "
                    f"{recomputed['pm_failures']} counted in the CSV"
                )
            for key in ("pm_error_rate", "always_zero_accuracy"):
                if not _close(
                    float(fields[NOTE_FIELDS[key]]), recomputed[key], _EXACT_FLOAT_TOLERANCE
                ):
                    problems.append(f"note {NOTE_FIELDS[key]} != {recomputed[key]!r}")
            low, high = json.loads(fields[NOTE_FIELDS["pm_ci"]])
            if not (
                _close(float(low), recomputed["pm_ci_low"], _EXACT_FLOAT_TOLERANCE)
                and _close(float(high), recomputed["pm_ci_high"], _EXACT_FLOAT_TOLERANCE)
            ):
                problems.append("note 95% confidence interval != recomputed Clopper-Pearson")
            if (
                _nonzero(_parse_split_counts(fields[NOTE_FIELDS["split_counts"]]))
                != (recomputed["split_counts"])
            ):
                problems.append("note split sizes != CSV split counts")
        except (KeyError, ValueError) as error:
            problems.append(f"note field missing or unparsable: {error}")
        if problems:
            raise _CheckFailedError("; ".join(problems))
        return (
            f"note and summary state {recomputed['pm_failures']} failures in "
            f"{recomputed['n_runs']} runs, rate {recomputed['pm_error_rate']!r}, CI "
            f"[{recomputed['pm_ci_low']!r}, {recomputed['pm_ci_high']!r}], as recomputed"
        )

    def check_checksums_match(self) -> str:
        problems: list[str] = []
        dem_sha = file_sha256(self.paths.decoder_dem)
        dem_text = self.paths.decoder_dem.read_bytes().decode("utf-8")
        recorded = {
            "decoder_metadata.dem_sha256": self.metadata.get("dem_sha256"),
            "raw.decoder_dem_sha256": self.raw.attrs.get("decoder_dem_sha256"),
            "summary.decoder_sha256": self.summary.get("decoder_sha256"),
        }
        wrong = {k: v for k, v in recorded.items() if v != dem_sha}
        if wrong:
            problems.append(f"published .dem hashes to {dem_sha}; recorded {wrong}")
        if self.metadata.get("dem_blake2b128") != dem_digest(dem_text):
            problems.append("decoder_metadata.dem_blake2b128 != blake2b-128 of the .dem text")
        if self.metadata.get("dem_file") != self.paths.decoder_dem.name:
            problems.append(
                f"decoder_metadata.dem_file {self.metadata.get('dem_file')!r} != "
                f"{self.paths.decoder_dem.name!r}"
            )
        if self.config is None:
            problems.append(f"resolved configuration: {self.config_error}")
        else:
            config = self.config
            recomputed_hash = config_hash(config)
            recorded_hashes = {
                "raw.config_hash": self.raw.attrs.get("config_hash"),
                "summary.config_hash": self.summary.get("config_hash"),
            }
            wrong_hashes = {k: v for k, v in recorded_hashes.items() if v != recomputed_hash}
            if wrong_hashes:
                problems.append(
                    f"config hash recomputed as {recomputed_hash}; recorded {wrong_hashes}"
                )
            if config.decoder.kind is DecoderKind.OFFICIAL_DEM and (
                config.decoder.expected_sha256 != dem_sha
            ):
                problems.append(
                    f"official .dem sha256 {dem_sha} != configured expected_sha256 "
                    f"{config.decoder.expected_sha256}"
                )
            if self.metadata.get("kind") != config.decoder.kind.value:
                problems.append(
                    f"decoder_metadata.kind {self.metadata.get('kind')!r} != configured "
                    f"{config.decoder.kind.value!r}"
                )
            # Broader than _CheckFailedError on purpose: a source file that has gone
            # missing (OSError) or whose rows no longer hash to its manifest (ValueError
            # from the streaming reader) is a checksum problem like any other and must be
            # listed *beside* the decoder and config-hash findings. Left to propagate, it
            # would reach the driver's catch-all and replace the whole detail string with
            # one exception name, hiding every other mismatch this check had found.
            try:
                label, digest, how = self._source_digest()
            except (_CheckFailedError, ValueError, OSError) as error:
                problems.append(f"source: {type(error).__name__}: {error}")
            else:
                recorded_source = self.raw.attrs.get("source_content_hash")
                if recorded_source != digest:
                    problems.append(
                        f"raw.source_content_hash {recorded_source!r} != recomputed source "
                        f"{label} {digest} ({how})"
                    )
                summary_hashes = self.summary.get("source_hashes")
                if not isinstance(summary_hashes, Mapping) or summary_hashes.get(label) != digest:
                    problems.append(f"summary.source_hashes[{label!r}] != recomputed {digest}")
        if problems:
            raise _CheckFailedError("; ".join(problems))
        return "source, decoder and configuration checksums recomputed and equal to the recorded"

    def check_source_prefix_matches(self) -> str:
        config = self._require_config()
        mode = config.generation.mode
        kind = config.source.kind
        if mode is GenerationMode.FRESH:
            return "skipped: fresh mode extends no source, so no prefix is claimed"
        if kind is SourceKind.HARDWARE_WILLOW:
            return self._check_willow_rows(config)
        if config.source.path is None:
            raise _CheckFailedError("source.path is missing from the resolved configuration")
        manifest = read_manifest_only(config.source.path)
        source_shots = int(manifest["shots"])
        expected = str(manifest["content_hash"])
        if mode is GenerationMode.SOURCE_ROWS and source_shots != self.raw.n_rows:
            raise _CheckFailedError(
                f"source_rows mode: the raw file holds {self.raw.n_rows} rows, the source "
                f"{source_shots}"
            )
        digest = self._raw_prefix_digest(source_shots)
        if digest != expected:
            raise _CheckFailedError(
                f"raw rows 0..{source_shots - 1} hash to {digest}; the source content_hash is "
                f"{expected}"
            )
        return f"raw rows 0..{source_shots - 1} re-hash to the source content_hash {expected}"

    def _check_willow_rows(self, config: ResidualConfig) -> str:
        source = config.source
        if source.table is None or source.circuit is None or source.expected is None:
            raise _CheckFailedError("hardware_willow configuration lacks table/circuit/expected")
        expected = source.expected
        cohort = load_willow_derived(
            source.table,
            source.circuit,
            {
                "table_sha256": expected.table_sha256,
                "circuit_sha256": expected.circuit_sha256,
                "distance": expected.distance,
                "basis": expected.basis,
                "rounds": expected.rounds,
                "orientation": expected.orientation,
            },
        )
        cohort_dets = np.asarray(cohort.detectors, dtype=np.uint8)
        cohort_obs = np.asarray(cohort.observables, dtype=np.uint8)
        if cohort_dets.shape[0] != self.raw.n_rows:
            raise _CheckFailedError(
                f"the raw file holds {self.raw.n_rows} rows, the mirror cohort "
                f"{cohort_dets.shape[0]}"
            )
        differing = 0
        for start, block in self.raw.blocks():
            stop = start + block.shape[0]
            differing += int(
                (
                    (block != cohort_dets[start:stop]).any(axis=1)
                    | (self.raw.observables[start:stop] != cohort_obs[start:stop]).any(axis=1)
                ).sum()
            )
        if differing:
            raise _CheckFailedError(f"raw rows differ from the mirror cohort on {differing} row(s)")
        return f"all {self.raw.n_rows} raw rows equal the verified mirror cohort bit for bit"

    def check_feature_inputs_audit(self) -> str:
        problems: list[str] = []
        recorded_fields = self.metadata.get("feature_context_fields")
        if recorded_fields != list(FEATURE_CONTEXT_FIELDS):
            problems.append(
                f"decoder_metadata.feature_context_fields {recorded_fields!r} != "
                f"{list(FEATURE_CONTEXT_FIELDS)}"
            )
        sanity = self.sanity
        sanity_ran = sanity is not None and "models" in sanity
        if sanity is not None and sanity_ran:
            if sanity.get("feature_columns") != list(FEATURE_COLUMNS):
                problems.append("sanity feature_columns != FEATURE_COLUMNS")
            if sanity.get("target") != "pm_wrong":
                problems.append(f"sanity target is {sanity.get('target')!r}")
            for name, model in dict(sanity["models"]).items():
                if model.get("feature_columns") != list(FEATURE_COLUMNS):
                    problems.append(f"sanity model {name} feature_columns != FEATURE_COLUMNS")
        try:
            state = self.decoder()
        except _CheckFailedError as error:
            problems.append(str(error))
        else:
            graph = state.context.graph
            recorded_graph = self.metadata.get("graph", {})
            if recorded_graph.get("digest") != graph.digest():
                problems.append("decoder_metadata.graph.digest != digest of the rebuilt graph")
            sizes = [int(s) for s in state.context.slices.sizes.tolist()]
            recorded_sizes = self.metadata.get("time_slices", {}).get("sizes")
            if recorded_sizes != sizes:
                problems.append(
                    f"decoder_metadata.time_slices.sizes {recorded_sizes} != rebuilt {sizes}"
                )
        if problems:
            raise _CheckFailedError("; ".join(problems))
        sanity_note = (
            "sanity inputs == FEATURE_COLUMNS" if sanity_ran else "sanity model absent or skipped"
        )
        return (
            f"feature context fields {list(FEATURE_CONTEXT_FIELDS)}; graph digest and slice "
            f"sizes reproduced; {sanity_note}"
        )

    def check_no_fit_on_held_out(self) -> str:
        problems: list[str] = []
        fitted = self.metadata.get("fitted_in_this_pipeline", None)
        if fitted is not False:
            problems.append(f"decoder_metadata.fitted_in_this_pipeline is {fitted!r}, not false")
        fitting = self.metadata.get("third_party_fitting")
        if fitting is not None and (
            not isinstance(fitting, Mapping) or not set(fitting) >= _THIRD_PARTY_FITTING_KEYS
        ):
            problems.append(
                "third_party_fitting is present but does not disclose "
                f"{sorted(_THIRD_PARTY_FITTING_KEYS)}"
            )
        sanity = self.sanity
        sanity_ran = sanity is not None and "models" in sanity
        if sanity is not None and sanity_ran:
            expected = {
                "fit_split": "train",
                "threshold_split": "validation",
                "evaluation_split": "test",
            }
            scopes: list[tuple[str, Mapping[str, Any]]] = [("sanity", sanity)]
            scopes.extend(
                (f"sanity model {name}", model) for name, model in dict(sanity["models"]).items()
            )
            for label, scope in scopes:
                for key, value in expected.items():
                    if scope.get(key) != value:
                        problems.append(f"{label}.{key} is {scope.get(key)!r}, not {value!r}")
        if problems:
            raise _CheckFailedError("; ".join(problems))
        disclosure = (
            "third-party fitting disclosed" if fitting is not None else "no third-party fitting"
        )
        sanity_note = (
            "sanity fit on train, threshold on validation, evaluated on test"
            if sanity_ran
            else "sanity model absent or skipped"
        )
        return (
            "no matching weights fitted in this pipeline; no validation/test rows of this "
            f"dataset entered any fitting performed by this pipeline; {disclosure}; {sanity_note}"
        )

    # -- driver -------------------------------------------------------------

    def run(self, name: str, check: Callable[[], str]) -> Check:
        try:
            return Check(name=name, passed=True, detail=check())
        except _CheckFailedError as error:
            return Check(name=name, passed=False, detail=str(error))
        except Exception as error:
            return Check(name=name, passed=False, detail=f"{type(error).__name__}: {error}")


def validate_dataset_dir(
    dataset_dir: Path,
    *,
    rebuild_decoder: bool = True,
    spot_rows: int = 5,
    spot_seed: int = 0,
) -> ValidationReport:
    """Run every check in :data:`CHECK_NAMES` on a published (or staged) dataset directory.

    ``rebuild_decoder=True`` also re-resolves the configuration and requires the decoder
    it rebuilds to hash to the published ``.dem`` (Willow: the circuit of record is the
    cached ideal circuit, verified by sha256, and the ``.dem`` must equal the configured
    official member digest). With ``False`` the published DEM's own coordinates define
    the time slices and the configuration is used only for its hashes and source paths.
    ``spot_rows`` random rows (seeded by ``spot_seed``) are decoded single-shot, batched,
    and re-featured through the pure-Python reference; one more row may be added as a
    bit-order witness. A missing artifact raises; every other defect is a failed check.
    """
    if spot_rows < 1:
        raise ValueError(f"spot_rows must be >= 1, got {spot_rows}")
    paths = _locate(Path(dataset_dir))
    validator = _Validator(
        paths, rebuild_decoder=rebuild_decoder, spot_rows=spot_rows, spot_seed=spot_seed
    )
    try:
        checks = tuple(
            validator.run(name, getattr(validator, f"check_{name}")) for name in CHECK_NAMES
        )
        spot_records: tuple[dict[str, Any], ...] = ()
        if validator._spot is not None:
            spot_records = tuple(dict(r) for r in validator._spot["records"])
    finally:
        validator.close()
    return ValidationReport(
        checks=checks,
        ok=all(c.passed for c in checks),
        spot_checks=spot_records,
        dataset_name=paths.name,
    )
