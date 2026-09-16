"""Artifact writers, note and manifest rendering, atomic publication."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import stim

from qecgen.exporters.base import NotAQecgenDatasetError
from qecgen.exporters.hdf5 import read_manifest_only
from qecgen.residual import SCHEMA_VERSION, writers
from qecgen.residual.config import DecoderKind
from qecgen.residual.decoder import (
    DecoderModel,
    decode_chunk,
    from_circuit,
    pm_wrong,
    truth_from_packed,
)
from qecgen.residual.features import (
    ALL_COLUMNS,
    FEATURE_COLUMNS,
    INTEGER_COLUMNS,
    FeatureContext,
    extract_features,
)
from qecgen.residual.graph import MatchingGraphSummary, summarise_graph, time_slices
from qecgen.residual.report import (
    MANIFEST_COLUMNS,
    NOTE_FIELDS,
    NOTE_LABELS,
    parse_note_fields,
    render_manifest,
    render_note,
)
from qecgen.residual.splits import SPLIT_CODES, SPLIT_NAMES, assign_splits
from qecgen.residual.writers import (
    RAW_FORMAT,
    RAW_GROUP,
    decoder_metadata,
    file_sha256,
    write_decoder_files,
    write_features_csv,
    write_json,
    write_raw_hdf5,
)
from qecgen.run import PARTIAL_PREFIX, staged
from qecgen.sampling import iter_chunks, packed_width, unpack_bits

N_ROWS = 64


@pytest.fixture(scope="module")
def model(d3_circuit: stim.Circuit) -> DecoderModel:
    return from_circuit(d3_circuit, DecoderKind.CIRCUIT_DEM, {"test": "writers"})


@pytest.fixture(scope="module")
def graph(model: DecoderModel) -> MatchingGraphSummary:
    return summarise_graph(model.dem, model.matching, model.n_detectors)


@pytest.fixture(scope="module")
def context(model: DecoderModel, graph: MatchingGraphSummary) -> FeatureContext:
    return FeatureContext(
        n_detectors=model.n_detectors, slices=time_slices(model.circuit), graph=graph
    )


@pytest.fixture(scope="module")
def raw_chunks(d3_circuit: stim.Circuit) -> list[tuple[np.ndarray, np.ndarray]]:
    """Two packed chunks of 32 rows; row 0 carries a planted asymmetric first byte."""
    chunks = [
        (c.detectors.copy(), c.observables.copy()) for c in iter_chunks(d3_circuit, N_ROWS, 7, 32)
    ]
    chunks[0][0][0, 0] = 0b0000_0001
    return chunks


@pytest.fixture(scope="module")
def feature_chunks(
    raw_chunks: list[tuple[np.ndarray, np.ndarray]], model: DecoderModel, context: FeatureContext
) -> list[dict[str, np.ndarray]]:
    out: list[dict[str, np.ndarray]] = []
    for detectors, observables in raw_chunks:
        guess, weight = decode_chunk(model, detectors)
        features = extract_features(detectors, guess, weight, context)
        truth = truth_from_packed(observables, 1)
        out.append(
            {
                "features": features,
                "pm_guess": guess,
                "pm_weight": weight,
                "truth": truth,
                "pm_wrong": pm_wrong(guess, truth),
            }
        )
    return out


@pytest.fixture(scope="module")
def split_codes() -> np.ndarray:
    return assign_splits(
        N_ROWS, "seeded_permutation", {"train": 0.7, "validation": 0.15, "test": 0.15}, 3
    )


def _attrs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "n_detectors": 24,
        "n_observables": 1,
        "source_content_hash": "a" * 64,
        "config_hash": "b" * 64,
        "decoder_dem_sha256": "c" * 64,
    }
    base.update(overrides)
    return base


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)
    return header, rows


class TestFeaturesCsv:
    def test_header_is_exactly_all_columns(
        self, tmp_path: Path, feature_chunks: list[dict[str, np.ndarray]], split_codes: np.ndarray
    ) -> None:
        path = tmp_path / "x_features.csv"
        n = write_features_csv(path, feature_chunks, split_codes)
        header, rows = _read_csv(path)
        assert header == list(ALL_COLUMNS)
        assert n == N_ROWS == len(rows)

    def test_integer_columns_contain_no_decimal_point(
        self, tmp_path: Path, feature_chunks: list[dict[str, np.ndarray]], split_codes: np.ndarray
    ) -> None:
        path = tmp_path / "x_features.csv"
        write_features_csv(path, feature_chunks, split_codes)
        header, rows = _read_csv(path)
        for name in INTEGER_COLUMNS:
            column = header.index(name)
            for row in rows:
                assert "." not in row[column], (name, row[column])
                int(row[column])

    def test_floats_round_trip_exactly_and_split_is_a_name(
        self, tmp_path: Path, feature_chunks: list[dict[str, np.ndarray]], split_codes: np.ndarray
    ) -> None:
        path = tmp_path / "x_features.csv"
        write_features_csv(path, feature_chunks, split_codes)
        header, rows = _read_csv(path)
        stacked = np.concatenate([c["features"] for c in feature_chunks])
        for j, name in enumerate(FEATURE_COLUMNS):
            column = header.index(name)
            for i, row in enumerate(rows):
                assert float(row[column]) == stacked[i, j], (name, i)
        run_column = header.index("run_id")
        assert [int(r[run_column]) for r in rows] == list(range(N_ROWS))
        split_column = header.index("split")
        names = [SPLIT_NAMES[int(c)] for c in split_codes]
        assert [r[split_column] for r in rows] == names
        truth_column = header.index("truth")
        wrong_column = header.index("pm_wrong")
        guess_column = header.index("pm_guess")
        for row in rows:
            assert int(row[wrong_column]) == int(int(row[guess_column]) != int(row[truth_column]))

    def test_split_length_mismatch_is_refused(
        self, tmp_path: Path, feature_chunks: list[dict[str, np.ndarray]], split_codes: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="split"):
            write_features_csv(tmp_path / "x.csv", feature_chunks, split_codes[:-1])

    def test_non_integral_value_in_integer_column_is_refused(
        self, tmp_path: Path, feature_chunks: list[dict[str, np.ndarray]], split_codes: np.ndarray
    ) -> None:
        broken = [dict(c) for c in feature_chunks]
        broken[0]["features"] = broken[0]["features"].copy()
        broken[0]["features"][0, FEATURE_COLUMNS.index("n_fired_total")] = 1.5
        with pytest.raises(ValueError, match="n_fired_total"):
            write_features_csv(tmp_path / "x.csv", broken, split_codes)

    def test_explicit_run_id_must_be_contiguous(
        self, tmp_path: Path, feature_chunks: list[dict[str, np.ndarray]], split_codes: np.ndarray
    ) -> None:
        broken = [dict(c) for c in feature_chunks]
        broken[1]["run_id"] = np.arange(32, 64, dtype=np.int64) + 1
        with pytest.raises(ValueError, match="run_id"):
            write_features_csv(tmp_path / "x.csv", broken, split_codes)

    def test_missing_chunk_key_is_refused_by_name(
        self, tmp_path: Path, feature_chunks: list[dict[str, np.ndarray]], split_codes: np.ndarray
    ) -> None:
        """A bare KeyError names nothing about *which* chunk field is absent."""
        broken = [dict(c) for c in feature_chunks]
        del broken[1]["truth"]
        with pytest.raises(ValueError, match=r"feature chunk lacks \['truth'\]"):
            write_features_csv(tmp_path / "x.csv", broken, split_codes)


class TestRawHdf5:
    def test_layout_attrs_dtypes_and_compression(
        self,
        tmp_path: Path,
        raw_chunks: list[tuple[np.ndarray, np.ndarray]],
        split_codes: np.ndarray,
    ) -> None:
        path = tmp_path / "x_raw.h5"
        run_ids = [np.arange(0, 32, dtype=np.int64), np.arange(32, 64, dtype=np.int64)]
        n = write_raw_hdf5(path, raw_chunks, run_ids, split_codes, _attrs())
        assert n == N_ROWS
        with h5py.File(path, "r") as handle:
            assert "detectors" not in handle
            group = handle[RAW_GROUP]
            detectors = group["detectors"]
            observables = group["observables"]
            assert detectors.dtype == np.uint8 and detectors.shape == (N_ROWS, packed_width(24))
            assert observables.dtype == np.uint8 and observables.shape == (N_ROWS, 1)
            assert group["run_id"].dtype == np.int64 and group["run_id"].shape == (N_ROWS,)
            assert group["split"].dtype == np.int8 and group["split"].shape == (N_ROWS,)
            assert detectors.compression == "gzip" and detectors.compression_opts == 4
            assert detectors.chunks == (N_ROWS, packed_width(24))
            attrs = handle.attrs
            assert attrs["format"] == RAW_FORMAT
            assert int(attrs["format_version"]) == 1
            assert int(attrs["n_detectors"]) == 24
            assert int(attrs["n_observables"]) == 1
            assert attrs["bit_order"] == "little"
            assert attrs["source_content_hash"] == "a" * 64
            assert attrs["config_hash"] == "b" * 64
            assert attrs["decoder_dem_sha256"] == "c" * 64
            assert int(attrs["feature_schema_version"]) == SCHEMA_VERSION
            assert json.loads(str(attrs["split_codes"])) == SPLIT_CODES
            versions = json.loads(str(attrs["versions"]))
            for name in ("stim", "pymatching", "numpy", "scipy", "h5py", "sinter", "qecgen"):
                assert name in versions
            np.testing.assert_array_equal(np.asarray(group["run_id"]), np.arange(N_ROWS))
            np.testing.assert_array_equal(np.asarray(group["split"]), split_codes)
            stacked = np.concatenate([c[0] for c in raw_chunks])
            np.testing.assert_array_equal(np.asarray(detectors), stacked)

    def test_planted_byte_unpacks_little_endian(
        self,
        tmp_path: Path,
        raw_chunks: list[tuple[np.ndarray, np.ndarray]],
        split_codes: np.ndarray,
    ) -> None:
        path = tmp_path / "x_raw.h5"
        run_ids = [np.arange(0, 32, dtype=np.int64), np.arange(32, 64, dtype=np.int64)]
        write_raw_hdf5(path, raw_chunks, run_ids, split_codes, _attrs())
        with h5py.File(path, "r") as handle:
            stored = np.asarray(handle[RAW_GROUP]["detectors"])
        little = unpack_bits(stored[:1], 24)[0]
        assert little[0] and not little[7]
        big = np.unpackbits(stored[:1], axis=1, count=24, bitorder="big")[0]
        assert big[7] and not big[0]
        assert not np.array_equal(little, big.astype(bool))

    def test_not_a_qecgen_dataset(
        self,
        tmp_path: Path,
        raw_chunks: list[tuple[np.ndarray, np.ndarray]],
        split_codes: np.ndarray,
    ) -> None:
        """No root `detectors` dataset, so the qecgen reader says "not ours" rather than
        "interrupted write"."""
        path = tmp_path / "x_raw.h5"
        run_ids = [np.arange(0, 32, dtype=np.int64), np.arange(32, 64, dtype=np.int64)]
        write_raw_hdf5(path, raw_chunks, run_ids, split_codes, _attrs())
        with pytest.raises(NotAQecgenDatasetError):
            read_manifest_only(path)

    def test_run_id_and_width_mismatches_are_refused(
        self,
        tmp_path: Path,
        raw_chunks: list[tuple[np.ndarray, np.ndarray]],
        split_codes: np.ndarray,
    ) -> None:
        bad_ids = [np.arange(0, 32, dtype=np.int64), np.arange(33, 65, dtype=np.int64)]
        with pytest.raises(ValueError, match="run_id"):
            write_raw_hdf5(tmp_path / "a.h5", raw_chunks, bad_ids, split_codes, _attrs())
        run_ids = [np.arange(0, 32, dtype=np.int64), np.arange(32, 64, dtype=np.int64)]
        with pytest.raises(ValueError, match="detectors"):
            write_raw_hdf5(
                tmp_path / "b.h5", raw_chunks, run_ids, split_codes, _attrs(n_detectors=40)
            )
        with pytest.raises(ValueError, match="missing"):
            attrs = _attrs()
            del attrs["config_hash"]
            write_raw_hdf5(tmp_path / "c.h5", raw_chunks, run_ids, split_codes, attrs)
        with pytest.raises(ValueError, match="bit_order"):
            write_raw_hdf5(
                tmp_path / "d.h5", raw_chunks, run_ids, split_codes, _attrs(bit_order="big")
            )

    def test_numpy_integer_widths_are_accepted_and_stored_as_int(
        self,
        tmp_path: Path,
        raw_chunks: list[tuple[np.ndarray, np.ndarray]],
        split_codes: np.ndarray,
    ) -> None:
        """The widths arrive as NumPy scalars from shapes and attributes; ``bool`` and
        non-positive values stay refused, and a float or string never passes as a width."""
        run_ids = [np.arange(0, 32, dtype=np.int64), np.arange(32, 64, dtype=np.int64)]
        path = tmp_path / "np.h5"
        attrs = _attrs(n_detectors=np.int64(24), n_observables=np.int32(1))
        assert write_raw_hdf5(path, raw_chunks, run_ids, split_codes, attrs) == N_ROWS
        with h5py.File(path, "r") as handle:
            assert int(handle.attrs["n_detectors"]) == 24
            assert int(handle.attrs["n_observables"]) == 1
        for bad in (True, np.bool_(True), 0, np.int64(0), 24.0, "24"):
            with pytest.raises(ValueError, match="n_detectors"):
                write_raw_hdf5(
                    tmp_path / "bad.h5", raw_chunks, run_ids, split_codes, _attrs(n_detectors=bad)
                )

    def test_run_id_identical_in_csv_and_hdf5(
        self,
        tmp_path: Path,
        raw_chunks: list[tuple[np.ndarray, np.ndarray]],
        feature_chunks: list[dict[str, np.ndarray]],
        split_codes: np.ndarray,
    ) -> None:
        run_ids = [np.arange(0, 32, dtype=np.int64), np.arange(32, 64, dtype=np.int64)]
        write_raw_hdf5(tmp_path / "x_raw.h5", raw_chunks, run_ids, split_codes, _attrs())
        write_features_csv(tmp_path / "x_features.csv", feature_chunks, split_codes)
        header, rows = _read_csv(tmp_path / "x_features.csv")
        csv_ids = [int(r[header.index("run_id")]) for r in rows]
        with h5py.File(tmp_path / "x_raw.h5", "r") as handle:
            raw_ids = np.asarray(handle[RAW_GROUP]["run_id"]).tolist()
        assert csv_ids == raw_ids


class TestDecoderFiles:
    def test_dem_bytes_and_metadata(
        self, tmp_path: Path, model: DecoderModel, graph: MatchingGraphSummary
    ) -> None:
        write_decoder_files(tmp_path, "x", model, graph)
        dem_path = tmp_path / "x_decoder.dem"
        assert dem_path.read_bytes() == model.dem_text.encode("utf-8")
        assert file_sha256(dem_path) == model.dem_sha256
        meta = json.loads((tmp_path / "x_decoder_metadata.json").read_text(encoding="utf-8"))
        assert meta["kind"] == "circuit_dem"
        assert meta["dem_sha256"] == model.dem_sha256
        assert meta["dem_blake2b128"] == model.dem_blake2b128
        assert meta["num_detectors"] == 24 and meta["num_observables"] == 1
        assert meta["num_errors"] == model.dem.num_errors
        assert meta["dem_stats"] == model.provenance["dem_stats"]
        assert meta["enable_correlations"] is False
        assert meta["fitted_in_this_pipeline"] is False
        assert meta["third_party_fitting"] is None
        assert meta["feature_context_fields"] == ["n_detectors", "slices", "graph"]
        assert meta["graph"]["digest"] == graph.digest()
        assert meta["graph"]["boundary_set_size"] == int(graph.boundary_adjacent.sum())
        assert meta["graph"]["logical_set_size"] == int(graph.logical_adjacent.sum())
        assert meta["graph"]["n_conflicting_pairs"] == 0
        assert meta["time_slices"]["sizes"] == [4, 8, 8, 4]
        assert meta["matching_mode"].startswith("standard matching")
        assert "stim" in meta["versions"] and "qecgen" in meta["versions"]
        assert meta["provenance"]["test"] == "writers"
        assert meta["dem_file"] == "x_decoder.dem"

    def test_decoder_metadata_is_public_and_is_what_gets_written(
        self, tmp_path: Path, model: DecoderModel, graph: MatchingGraphSummary
    ) -> None:
        assert "decoder_metadata" in writers.__all__
        write_decoder_files(tmp_path, "x", model, graph)
        written = json.loads((tmp_path / "x_decoder_metadata.json").read_text(encoding="utf-8"))
        record = json.loads(json.dumps(decoder_metadata("x", model, graph), sort_keys=True))
        assert record == written


class TestWriteJson:
    def test_sorted_indented_and_nan_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "p.json"
        write_json(path, {"b": 1, "a": {"z": [1, 2], "y": None}})
        text = path.read_text(encoding="utf-8")
        assert text.index('"a"') < text.index('"b"')
        assert text.startswith("{\n  ")
        assert text.endswith("\n")
        with pytest.raises(ValueError):
            write_json(tmp_path / "q.json", {"x": float("nan")})
        assert not (tmp_path / "q.json").exists()

    def test_bytes_use_lf_on_every_platform(self, tmp_path: Path) -> None:
        """``summary_sha256`` is a digest of these bytes; CRLF on Windows would make it a
        platform-dependent number for identical content."""
        payload = {"a": [1, 2], "b": {"c": None, "d": "x"}}
        path = tmp_path / "lf.json"
        write_json(path, payload)
        raw = path.read_bytes()
        assert raw.endswith(b"\n")
        assert b"\r" not in raw
        expected = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
        assert raw == expected
        assert file_sha256(path) == hashlib.sha256(expected).hexdigest()


class TestAtomicPublication:
    def test_failure_inside_staged_leaves_nothing(
        self, tmp_path: Path, feature_chunks: list[dict[str, np.ndarray]], split_codes: np.ndarray
    ) -> None:
        destination = tmp_path / "out"
        with pytest.raises(RuntimeError, match="planted"), staged(destination) as staging:
            write_features_csv(staging.scratch / "x_features.csv", feature_chunks, split_codes)
            assert (staging.scratch / "x_features.csv").exists()
            raise RuntimeError("planted")
        assert not list(destination.glob("*_features.csv"))
        assert not list(destination.glob(f"{PARTIAL_PREFIX}*"))
        assert not list(tmp_path.rglob(f"{PARTIAL_PREFIX}*"))

    def test_success_inside_staged_publishes(
        self, tmp_path: Path, feature_chunks: list[dict[str, np.ndarray]], split_codes: np.ndarray
    ) -> None:
        destination = tmp_path / "out"
        with staged(destination) as staging:
            write_features_csv(staging.scratch / "x_features.csv", feature_chunks, split_codes)
        assert (destination / "x_features.csv").exists()
        assert not list(destination.glob(f"{PARTIAL_PREFIX}*"))


def _summary(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "dataset_name": "indep_d3_test",
        "additional": False,
        "source": "legacy_ml_csv: d=3 pilot",
        "source_paths": {"table": "C:/x/pilot.ml.csv", "manifest": "C:/x/pilot.ml.manifest.json"},
        "source_hashes": {"content_hash": "a" * 64},
        "provenance_status": "verified by content hash",
        "provenance_limitations": ["none"],
        "distance": 3,
        "rounds": 3,
        "basis": "Z",
        "orientation": None,
        "n_detectors": 24,
        "n_observables": 1,
        "noise_model": "stim_uniform_circuit_level",
        "noise_parameters": {"p": 0.01},
        "n_runs": 64,
        "seed": 7,
        "chunk_size": 32,
        "versions": {
            "stim": "1.16.0",
            "sinter": "1.16.0",
            "pymatching": "2.4.0",
            "numpy": "2.3.3",
            "scipy": "1.16.2",
            "qecgen": "0.1.0",
        },
        "decoder_method": "circuit.detector_error_model(decompose_errors=True)",
        "decoder_source": "rebuilt legacy circuit",
        "matching_mode": "standard matching (enable_correlations=False)",
        "decoder_path": "indep_d3_test_decoder.dem",
        "decoder_sha256": "c" * 64,
        "config_hash": "b" * 64,
        "schema_version": 1,
        "split_method": "seeded_permutation",
        "split_seed": 3,
        "split_fractions": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "split_counts": {"train": 45, "validation": 10, "test": 9},
        "pm_failures": 5,
        "pm_error_rate": 0.078125,
        "pm_ci_low": 0.025806,
        "pm_ci_high": 0.173338,
        "always_zero_accuracy": 0.921875,
        "sanity": {
            "models": {
                "logistic_regression": {
                    "balanced_accuracy": 0.51,
                    "roc_auc": 0.55,
                    "pr_auc": 0.1,
                    "pm_test_error_rate": 0.08,
                    "corrected_test_error_rate": 0.08,
                    "n_flips": 0,
                }
            }
        },
        "limitations": ["d=3: every node is boundary-adjacent"],
        "deviations": ["raw arrays under /residual"],
    }
    base.update(overrides)
    return base


class TestRenderNote:
    def test_every_required_field_and_summary_sha_present(self) -> None:
        note = render_note(_summary(), summary_sha256="e" * 64)
        for label in NOTE_FIELDS.values():
            assert f"{label}:" in note, label
        assert "summary_sha256: " + "e" * 64 in note
        fields = parse_note_fields(note)
        assert fields["summary_sha256"] == "e" * 64
        assert fields[NOTE_FIELDS["pm_failures"]] == "5"
        assert fields[NOTE_FIELDS["pm_error_rate"]] == repr(0.078125)
        assert fields[NOTE_FIELDS["always_zero_accuracy"]] == repr(0.921875)
        assert "0.025806" in fields[NOTE_FIELDS["pm_ci"]]
        assert fields[NOTE_FIELDS["n_runs"]] == "64"
        assert "train=45" in fields[NOTE_FIELDS["split_counts"]]
        assert "boundary-adjacent" in note
        assert "1.16.0" in note and "2.4.0" in note

    def test_missing_field_is_an_error(self) -> None:
        summary = _summary()
        del summary["pm_failures"]
        with pytest.raises(ValueError, match="pm_failures"):
            render_note(summary, summary_sha256="e" * 64)

    def test_bad_digest_is_refused(self) -> None:
        with pytest.raises(ValueError, match="summary_sha256"):
            render_note(_summary(), summary_sha256="nope")

    def test_additional_and_skipped_sanity_are_stated(self) -> None:
        note = render_note(
            _summary(additional=True, sanity={"skipped_reason": "sklearn missing"}),
            summary_sha256="e" * 64,
        )
        assert "additional" in note.lower()
        assert "sklearn missing" in note

    def test_summary_sha_matches_file(self, tmp_path: Path) -> None:
        path = tmp_path / "s.json"
        write_json(path, _summary())
        assert file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()

    def test_numpy_scalars_render_as_plain_numbers(self) -> None:
        """The summary is built from NumPy arithmetic. Under NumPy 2 a ``np.float64``
        rendered by ``repr`` reads ``np.float64(0.5)`` and a ``np.int64`` is not an
        ``int``, so it used to reach ``json.dumps`` and raise."""
        summary = _summary(
            additional=np.bool_(False),
            distance=np.int64(3),
            rounds=np.int32(3),
            n_detectors=np.int64(24),
            n_observables=np.int64(1),
            seed=np.int64(7),
            chunk_size=np.int64(32),
            schema_version=np.int64(1),
            split_seed=np.int64(3),
            pm_failures=np.int64(5),
            pm_error_rate=np.float64(0.078125),
            pm_ci_low=np.float64(0.025806),
            pm_ci_high=np.float64(0.173338),
            always_zero_accuracy=np.float64(0.921875),
        )
        metrics = summary["sanity"]["models"]["logistic_regression"]
        metrics["n_flips"] = np.int64(0)
        metrics["roc_auc"] = np.float64(0.55)
        note = render_note(summary, summary_sha256="e" * 64)
        manifest = render_manifest([summary], [])
        assert "np." not in note
        assert "np." not in manifest
        fields = parse_note_fields(note)
        assert fields[NOTE_FIELDS["n_detectors"]] == "24"
        assert fields[NOTE_FIELDS["distance"]] == "3"
        assert fields[NOTE_FIELDS["seed"]] == "7"
        assert fields[NOTE_FIELDS["pm_failures"]] == "5"
        assert fields[NOTE_FIELDS["pm_error_rate"]] == repr(0.078125)
        assert fields["Split seed"] == "3"
        assert "n_flips=0" in note
        assert "roc_auc=0.55" in note
        assert "| 3 | 3 | Z | 64 | 24 |" in manifest
        assert "n_flips=0" in manifest

    def test_prose_bullets_with_colons_are_not_fields(self) -> None:
        """Limitations are sentences; two sharing a ``label:`` prefix used to make the
        whole note unparsable, and one starting with a real label used to collide."""
        summary = _summary(
            provenance_limitations=[
                "si1000 prior: not reproducible from the circuit",
                "si1000 prior: shipped verbatim by the publisher",
            ],
            limitations=["Seed: the source's seed is not this dataset's"],
            deviations=["Split seed: chosen after the fact"],
        )
        note = render_note(summary, summary_sha256="e" * 64)
        fields = parse_note_fields(note)
        assert "si1000 prior" not in fields
        assert set(fields) <= NOTE_LABELS
        assert set(NOTE_FIELDS.values()) <= set(fields)
        assert fields[NOTE_FIELDS["seed"]] == "7"
        assert fields["Split seed"] == "3"
        assert fields["summary_sha256"] == "e" * 64
        assert "si1000 prior: shipped verbatim" in note
        # A real label repeated outside the prose sections is still an error.
        with pytest.raises(ValueError, match="more than once"):
            parse_note_fields(note + "\n- Seed: 8\n")


class TestRenderManifest:
    def test_one_row_per_dataset_with_flags(self) -> None:
        summaries = [_summary(), _summary(dataset_name="willow_rl", additional=True)]
        blocked = [{"dataset_name": "willow_si1000", "reason": "Zenodo unreachable | retry"}]
        text = render_manifest(summaries, blocked)
        lines = [line for line in text.splitlines() if line.startswith("|")]
        header = [cell.strip() for cell in lines[0].strip("|").split("|")]
        assert header == list(MANIFEST_COLUMNS)
        rows = lines[2:]
        assert len(rows) == 3
        assert "indep_d3_test" in rows[0] and "completed" in rows[0]
        assert "additional" in rows[1]
        assert "willow_si1000" in rows[2] and "blocked" in rows[2]
        assert "Zenodo unreachable" in rows[2]
        assert "\\|" in rows[2]
        assert "5" in rows[0] and repr(0.078125) in rows[0]
        assert "balanced_accuracy" in rows[0] or "bal" in rows[0]

    def test_empty_manifest_still_renders_header(self) -> None:
        text = render_manifest([], [])
        assert "|" in text
        assert "no datasets" in text.lower()
        assert "0 completed, 0 blocked, 0 failed" in text

    def test_failed_row_keeps_its_status_word(self) -> None:
        """A build that broke after resolution is not a missing input; the row and the
        count must say ``failed``, and an unknown status is refused rather than guessed."""
        entries = [
            {"dataset_name": "a_blocked", "reason": "table missing", "status": "blocked"},
            {
                "dataset_name": "b_failed",
                "reason": "build failed: ValueError: x",
                "status": "failed",
            },
            {"dataset_name": "c_unstated", "reason": "no status key"},
        ]
        text = render_manifest([], entries)
        assert "0 completed, 2 blocked, 1 failed" in text
        rows = {line.split(" | ")[0].strip("| "): line for line in text.splitlines() if "_" in line}
        assert rows["a_blocked"].endswith("| blocked: table missing |")
        assert rows["b_failed"].endswith("| failed: build failed: ValueError: x |")
        assert rows["c_unstated"].endswith("| blocked: no status key |")
        with pytest.raises(ValueError, match="status"):
            render_manifest([], [{"dataset_name": "x", "reason": "r", "status": "exploded"}])
