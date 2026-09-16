"""Final artifact writers: feature CSV, compact raw HDF5, decoder files, JSON.

Every function here writes to the path it is given and nothing else; the *caller*
(``pipeline.build``) hands it a path inside ``run.staged(...).scratch`` so that the
two-phase commit publishes the whole artifact set or none of it. A writer that opened
the destination directly would recreate the trap ``staged`` exists for: an interrupted
build leaving a truncated ``<name>_features.csv`` under its final name, which still
parses row by row and only a row-count check would notice.

Three conventions are pinned here because a reader cannot recover them from the file:

* **Integer columns are written as integers.** ``extract_features`` returns float64 for
  every column, so ``pm_guess`` would otherwise land as ``1.0`` and a consumer coercing
  the column to bool or int by string would read ``"1.0"`` differently from ``"1"``. The
  writer refuses a non-integral value in an integer column rather than rounding it,
  because that value can only come from a broken extractor.
* **Floats are written as ``repr(float(x))``**, the shortest string that round-trips
  exactly, so the CSV reproduces the float64 the pipeline computed bit for bit and the
  validation spot checks can compare with a tolerance that means something.
* **The raw HDF5 keeps its arrays under the ``/residual`` group.** ``qecgen.exporters.hdf5``
  treats an HDF5 file with a root ``detectors`` dataset but no manifest as an
  *interrupted qecgen write*, and ``ui/datasets.list_datasets`` would flag every residual
  raw file as corruption. Under a group the same reader says "not ours", which is the
  truth. ``bit_order`` is recorded as an attribute and is always little-endian.
"""

from __future__ import annotations

import hashlib
import json
import numbers
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from qecgen.dataset import library_versions
from qecgen.residual import SCHEMA_VERSION
from qecgen.residual.config import DecoderKind
from qecgen.residual.decoder import DecoderModel
from qecgen.residual.features import ALL_COLUMNS, FEATURE_COLUMNS, INTEGER_COLUMNS
from qecgen.residual.graph import MatchingGraphSummary, TimeSlices, time_slices
from qecgen.residual.splits import SPLIT_CODES, SPLIT_NAMES
from qecgen.sampling import packed_width

__all__ = [
    "FEATURE_CONTEXT_FIELDS",
    "RAW_FORMAT",
    "RAW_FORMAT_VERSION",
    "RAW_GROUP",
    "decoder_metadata",
    "file_sha256",
    "write_decoder_files",
    "write_features_csv",
    "write_json",
    "write_raw_hdf5",
]

RAW_GROUP = "residual"
"""HDF5 group holding the raw arrays; never the root (see the module docstring)."""

RAW_FORMAT = "qecgen-residual-raw"
RAW_FORMAT_VERSION = 1

FEATURE_CONTEXT_FIELDS: tuple[str, ...] = ("n_detectors", "slices", "graph")
"""What feature extraction may see; recorded in the decoder metadata so validation can
assert it (the leakage audit in :mod:`qecgen.residual.features` enforces it at runtime)."""

_HDF5_CHUNK_ROWS = 8192
_REQUIRED_RAW_ATTRS: tuple[str, ...] = (
    "n_detectors",
    "n_observables",
    "source_content_hash",
    "config_hash",
    "decoder_dem_sha256",
)
_WRITER_OWNED_RAW_ATTRS: tuple[str, ...] = ("format", "format_version", "bit_order", "split_codes")

_MODE_TEXT: dict[DecoderKind, str] = {
    DecoderKind.CIRCUIT_DEM: "standard matching on the exact DEM of the rebuilt noisy circuit",
    DecoderKind.STATIC_PROFILE_DEM: (
        "standard matching on the exact DEM of the one fixed noisy circuit the static "
        "profile produces"
    ),
    DecoderKind.FROZEN_REFERENCE_DEM: (
        "standard matching on a frozen reference DEM; not an exact DEM of the dynamic process"
    ),
    DecoderKind.OFFICIAL_DEM: "standard matching on the publisher's shipped prior (used verbatim)",
}

_PM_GUESS_COLUMN = FEATURE_COLUMNS.index("pm_guess")
_PM_WEIGHT_COLUMN = FEATURE_COLUMNS.index("pm_weight")


def file_sha256(path: Path) -> str:
    """sha256 of a file's bytes, streamed; the digest every checksum claim is made with."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Canonical JSON: sorted keys, two-space indent, NaN refused, LF line ends.

    Serialised to a string *before* the file is opened so a NaN raises without leaving a
    half-written file behind; ``json.dump`` streams and would truncate mid-object.
    ``newline="\\n"`` is load-bearing: without it ``write_text`` translates every ``\\n``
    to ``\\r\\n`` on Windows, so ``summary_sha256`` (the note's digest of this file) and
    the decoder-metadata bytes would differ between platforms for identical content.
    """
    text = json.dumps(dict(payload), sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _validate_split_codes(split_codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(split_codes)
    if codes.ndim != 1:
        raise ValueError(f"split codes must be 1-D, got shape {codes.shape}")
    if codes.size == 0:
        raise ValueError("split codes are empty; a dataset with no rows is not publishable")
    if not np.issubdtype(codes.dtype, np.integer):
        raise ValueError(f"split codes must be integers, got dtype {codes.dtype}")
    valid = np.asarray(sorted(SPLIT_CODES.values()))
    if not np.isin(codes, valid).all():
        raise ValueError(f"split codes contain values outside {SPLIT_CODES}")
    return codes.astype(np.int8)


def _check_run_ids(run_id: np.ndarray, offset: int, n_rows: int, where: str) -> None:
    """``run_id`` is the row index, and it is checked rather than assigned from an
    explicit array: two artifacts written from independent iterators must agree, and the
    only way to guarantee that is to refuse anything but ``offset .. offset + n``."""
    ids = np.asarray(run_id)
    if ids.shape != (n_rows,) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(
            f"{where}: run_id must be an integer array of shape ({n_rows},), got "
            f"{ids.dtype} {ids.shape}"
        )
    expected = np.arange(offset, offset + n_rows, dtype=np.int64)
    if not np.array_equal(ids.astype(np.int64), expected):
        raise ValueError(
            f"{where}: run_id must be contiguous row indices {offset}..{offset + n_rows - 1}"
        )


def _format_integer_column(values: np.ndarray, name: str) -> list[str]:
    rounded = np.rint(values)
    if not np.array_equal(rounded, values):
        raise ValueError(
            f"column {name} is declared integer but holds a non-integral value; "
            "this can only come from a broken extractor, so it is refused rather than rounded"
        )
    return [str(int(v)) for v in rounded.astype(np.int64)]


def _format_float_column(values: np.ndarray) -> list[str]:
    if not np.isfinite(values).all():
        raise ValueError("a float column holds NaN or infinity")
    return [repr(float(v)) for v in values]


def _feature_chunk_arrays(chunk: Mapping[str, np.ndarray], offset: int) -> tuple[np.ndarray, ...]:
    """Validate one feature chunk and return its column arrays in ``ALL_COLUMNS`` order."""
    missing = [name for name in ("features", "truth", "pm_wrong") if name not in chunk]
    if missing:
        raise ValueError(f"feature chunk lacks {missing}")
    features = np.asarray(chunk["features"], dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"features chunk has shape {features.shape}; expected (rows, {len(FEATURE_COLUMNS)})"
        )
    n_rows = features.shape[0]
    columns: dict[str, np.ndarray] = {}
    for name in ("truth", "pm_wrong"):
        values = np.asarray(chunk[name])
        if values.shape != (n_rows,):
            raise ValueError(f"{name} has shape {values.shape}; expected ({n_rows},)")
        if not np.isin(values, (0, 1)).all():
            raise ValueError(f"{name} must be binary")
        columns[name] = values.astype(np.int64)
    # pm_guess and pm_weight live inside the feature block; a separately supplied copy
    # must agree with it, or the CSV would carry a guess the features were not computed on.
    if "pm_guess" in chunk:
        guess = np.asarray(chunk["pm_guess"])
        if guess.shape != (n_rows,) or not np.array_equal(
            guess.astype(np.float64), features[:, _PM_GUESS_COLUMN]
        ):
            raise ValueError("pm_guess disagrees with the pm_guess feature column")
    if "pm_weight" in chunk:
        weight = np.asarray(chunk["pm_weight"], dtype=np.float64)
        if weight.shape != (n_rows,) or not np.array_equal(weight, features[:, _PM_WEIGHT_COLUMN]):
            raise ValueError("pm_weight disagrees with the pm_weight feature column")
    expected_wrong = (columns["truth"] != features[:, _PM_GUESS_COLUMN].astype(np.int64)).astype(
        np.int64
    )
    if not np.array_equal(columns["pm_wrong"], expected_wrong):
        raise ValueError("pm_wrong != (pm_guess != truth) inside a feature chunk")
    run_id = chunk.get("run_id")
    if run_id is None:
        run_id = np.arange(offset, offset + n_rows, dtype=np.int64)
    _check_run_ids(run_id, offset, n_rows, "features CSV")
    columns["run_id"] = np.asarray(run_id, dtype=np.int64)
    return (*(features[:, j] for j in range(len(FEATURE_COLUMNS))), *columns.values())


def write_features_csv(
    path: Path, chunks: Iterable[Mapping[str, np.ndarray]], split_codes: np.ndarray
) -> int:
    """Write the schema v1 feature CSV; returns the row count.

    Each chunk maps ``features`` (float64 ``(n, 24)`` in ``FEATURE_COLUMNS`` order),
    ``truth`` and ``pm_wrong`` (binary ``(n,)``), optionally ``pm_guess``/``pm_weight``
    (checked against the feature block) and ``run_id`` (checked against the row index).
    ``split_codes`` covers every row of the dataset and is written as split *names*, the
    form a human reads; the raw HDF5 keeps the codes and the code table.
    """
    codes = _validate_split_codes(split_codes)
    total = int(codes.shape[0])
    label_names = ("truth", "pm_wrong", "run_id")
    offset = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(ALL_COLUMNS) + "\n")
        for chunk in chunks:
            arrays = _feature_chunk_arrays(chunk, offset)
            n_rows = arrays[0].shape[0]
            if offset + n_rows > total:
                raise ValueError(
                    f"feature chunks carry more than the {total} rows the split codes cover"
                )
            cells: list[list[str]] = []
            for name, values in zip((*FEATURE_COLUMNS, *label_names), arrays, strict=True):
                if name in INTEGER_COLUMNS:
                    cells.append(_format_integer_column(values, name))
                else:
                    cells.append(_format_float_column(values))
            cells.append([SPLIT_NAMES[int(c)] for c in codes[offset : offset + n_rows]])
            fh.writelines(",".join(row) + "\n" for row in zip(*cells, strict=True))
            offset += n_rows
    if offset != total:
        raise ValueError(
            f"feature chunks carry {offset} rows but the split codes cover {total}; "
            "the CSV would disagree with the raw file"
        )
    return offset


def _validate_raw_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in _REQUIRED_RAW_ATTRS if name not in attrs]
    if missing:
        raise ValueError(f"raw HDF5 attributes missing: {missing}")
    owned = sorted(set(attrs) & set(_WRITER_OWNED_RAW_ATTRS))
    if owned:
        raise ValueError(
            f"raw HDF5 attributes {owned} are written by the writer and cannot be supplied "
            "(bit_order is always little)"
        )
    payload = dict(attrs)
    for name in ("n_detectors", "n_observables"):
        value = payload[name]
        # numbers.Integral rather than int: the widths arrive as NumPy scalars (a shape
        # element, an attribute read back from another HDF5 file), which are Integral
        # but not int. bool is excluded by name because it is Integral too and True
        # would otherwise pass as a width of 1.
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
            raise ValueError(f"raw HDF5 attribute {name} must be a positive int, got {value!r}")
        payload[name] = int(value)
    payload.setdefault("feature_schema_version", SCHEMA_VERSION)
    payload.setdefault("versions", library_versions())
    return payload


def _store_attr(target: Any, name: str, value: Any) -> None:
    """Scalars go in as they are; anything structured is stored as canonical JSON text so
    a reader never has to guess how a dict was flattened."""
    if isinstance(value, (dict, list, tuple)):
        target.attrs[name] = json.dumps(value, sort_keys=True, allow_nan=False)
    elif value is None:
        target.attrs[name] = "null"
    else:
        target.attrs[name] = value


def write_raw_hdf5(
    path: Path,
    raw_chunks: Iterable[tuple[np.ndarray, np.ndarray]],
    run_ids: Iterable[np.ndarray],
    split_codes: np.ndarray,
    attrs: Mapping[str, Any],
) -> int:
    """Write the compact raw file; returns the row count.

    ``split_codes`` fixes the row count up front, so every dataset is created at its
    final shape and filled chunk by chunk: a chunk stream that ends early or runs over
    is refused, never padded or truncated. The packed widths are checked against the
    *true* widths in ``attrs`` because a packed width never implies the true one.
    """
    codes = _validate_split_codes(split_codes)
    payload = _validate_raw_attrs(attrs)
    total = int(codes.shape[0])
    n_detectors = int(payload["n_detectors"])
    n_observables = int(payload["n_observables"])
    det_width = packed_width(n_detectors)
    obs_width = packed_width(n_observables)
    chunk_rows = min(_HDF5_CHUNK_ROWS, total)

    offset = 0
    with h5py.File(path, "w") as handle:
        _store_attr(handle, "format", RAW_FORMAT)
        _store_attr(handle, "format_version", RAW_FORMAT_VERSION)
        _store_attr(handle, "bit_order", "little")
        _store_attr(handle, "split_codes", SPLIT_CODES)
        for name, value in payload.items():
            _store_attr(handle, name, value)
        group = handle.create_group(RAW_GROUP)
        detectors = group.create_dataset(
            "detectors",
            shape=(total, det_width),
            dtype=np.uint8,
            chunks=(chunk_rows, det_width),
            compression="gzip",
            compression_opts=4,
        )
        observables = group.create_dataset(
            "observables",
            shape=(total, obs_width),
            dtype=np.uint8,
            chunks=(chunk_rows, obs_width),
            compression="gzip",
            compression_opts=4,
        )
        run_id_set = group.create_dataset(
            "run_id", shape=(total,), dtype=np.int64, chunks=(chunk_rows,), compression="gzip"
        )
        split_set = group.create_dataset(
            "split", shape=(total,), dtype=np.int8, chunks=(chunk_rows,), compression="gzip"
        )
        split_set[:] = codes

        for (det_chunk, obs_chunk), ids in zip(raw_chunks, run_ids, strict=True):
            det = np.asarray(det_chunk)
            obs = np.asarray(obs_chunk)
            if det.dtype != np.uint8 or det.ndim != 2 or det.shape[1] != det_width:
                raise ValueError(
                    f"raw detectors chunk has dtype {det.dtype} shape {det.shape}; expected "
                    f"uint8 (rows, {det_width}) for {n_detectors} detectors"
                )
            n_rows = det.shape[0]
            if obs.dtype != np.uint8 or obs.shape != (n_rows, obs_width):
                raise ValueError(
                    f"raw observables chunk has dtype {obs.dtype} shape {obs.shape}; expected "
                    f"uint8 ({n_rows}, {obs_width}) for {n_observables} observables"
                )
            _check_run_ids(ids, offset, n_rows, "raw HDF5")
            if offset + n_rows > total:
                raise ValueError(
                    f"raw chunks carry more than the {total} rows the split codes cover"
                )
            detectors[offset : offset + n_rows] = det
            observables[offset : offset + n_rows] = obs
            run_id_set[offset : offset + n_rows] = np.asarray(ids, dtype=np.int64)
            offset += n_rows
    if offset != total:
        raise ValueError(
            f"raw chunks carry {offset} rows but the split codes cover {total}; the file "
            "would be padded with zero rows that were never sampled"
        )
    return offset


def decoder_metadata(
    name: str,
    model: DecoderModel,
    graph: MatchingGraphSummary,
    slices: TimeSlices | None = None,
) -> dict[str, Any]:
    """The decoder metadata record; separated from the write so tests and the note can
    read it without touching disk."""
    if slices is None:
        slices = time_slices(model.circuit)
    if model.enable_correlations:
        raise ValueError("correlated matching is a separate configuration; refusing to record it")
    provenance = dict(model.provenance)
    fitted = provenance.get("fitted_in_this_pipeline", False)
    if fitted is not False:
        raise ValueError(
            "this pipeline never fits matching weights; a decoder claiming "
            f"fitted_in_this_pipeline={fitted!r} cannot be published"
        )
    if graph.n_detectors != model.n_detectors:
        raise ValueError(
            f"graph summary covers {graph.n_detectors} detectors; model has {model.n_detectors}"
        )
    return {
        "dataset_name": name,
        "dem_file": f"{name}_decoder.dem",
        "kind": model.kind.value,
        "dem_sha256": model.dem_sha256,
        "dem_blake2b128": model.dem_blake2b128,
        "num_detectors": model.n_detectors,
        "num_observables": model.n_observables,
        "num_errors": model.dem.num_errors,
        "dem_stats": provenance.get("dem_stats"),
        "enable_correlations": False,
        "matching_mode": "standard matching (enable_correlations=False)",
        "mode_text": _MODE_TEXT[model.kind],
        "matching_construction": provenance.get("matching_construction"),
        "fitted_in_this_pipeline": False,
        "third_party_fitting": provenance.get("third_party_fitting"),
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
            "n_matching_edges": len(graph.matching_edge_weights),
        },
        "time_slices": {
            "n_slices": slices.n_slices,
            "sizes": [int(s) for s in slices.sizes.tolist()],
            "times": [float(t) for t in slices.times],
            "rule": (
                "slice = rank of the detector's latest time coordinate "
                "(qecgen.hardware.detector_anchors)"
            ),
        },
        "feature_context_fields": list(FEATURE_CONTEXT_FIELDS),
        "feature_schema_version": SCHEMA_VERSION,
        "provenance": provenance,
        "versions": library_versions(),
    }


def write_decoder_files(
    dataset_dir: Path,
    name: str,
    model: DecoderModel,
    graph: MatchingGraphSummary,
    *,
    slices: TimeSlices | None = None,
) -> None:
    """Write ``<name>_decoder.dem`` and ``<name>_decoder_metadata.json`` into ``dataset_dir``.

    The ``.dem`` is ``dem_text`` encoded as UTF-8 and nothing else: for an official DEM
    that *is* the member's bytes, so the published file hashes to the recorded member
    sha256, and for a circuit-derived DEM it is the exact text the matcher was built from.
    """
    payload = decoder_metadata(name, model, graph, slices)
    dem_path = dataset_dir / payload["dem_file"]
    dem_path.write_bytes(model.dem_text.encode("utf-8"))
    if file_sha256(dem_path) != model.dem_sha256:
        raise ValueError("written .dem does not hash to the model's dem_sha256")
    write_json(dataset_dir / f"{name}_decoder_metadata.json", payload)
