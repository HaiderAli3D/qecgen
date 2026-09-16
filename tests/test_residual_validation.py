"""Validation of a published residual dataset: the fifteen named checks (the brief's
fourteen assertions) and the spot checks.

The fixture publishes a tiny d=3 dataset with the same writers the pipeline uses (the
pipeline module itself is a later task), then every test corrupts one artifact of a private
copy and asserts that exactly the named check catches it. Nothing here touches ``data/``.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from qecgen.dataset import library_versions
from qecgen.environments import build_single_environment
from qecgen.exporters.ml_csv import MLCSVExporter, read_manifest_only
from qecgen.qa import clopper_pearson
from qecgen.residual import SCHEMA_VERSION
from qecgen.residual.config import config_hash, parse_config, resolved_dict
from qecgen.residual.decoder import decode_chunk, pm_wrong, truth_from_packed
from qecgen.residual.features import ALL_COLUMNS, FeatureContext, extract_features
from qecgen.residual.graph import summarise_graph, time_slices
from qecgen.residual.report import render_note
from qecgen.residual.sanity import SanityModelUnavailableError, run_sanity_models
from qecgen.residual.sources import resolve_source
from qecgen.residual.splits import SPLIT_CODES, SPLIT_NAMES, assign_splits
from qecgen.residual.validation import (
    CHECK_NAMES,
    ValidationReport,
    validate_dataset_dir,
)
from qecgen.residual.writers import (
    RAW_GROUP,
    file_sha256,
    write_decoder_files,
    write_features_csv,
    write_json,
    write_raw_hdf5,
)
from qecgen.sampling import iter_chunks, unpack_bits

NAME = "legacy_d3_validation"
SOURCE_SHOTS = 64
SHOTS = 200
CHUNK = 32
SEED = 1
N_DETECTORS = 24


def _build_dataset(root: Path) -> Path:
    """Publish a complete artifact set for a 200-row extension of a 64-row d=3 source."""
    source_dir = root / "source"
    source_dir.mkdir()
    source_path = source_dir / "legacy.ml.csv"
    MLCSVExporter().write(
        build_single_environment(
            distance=3, p=0.01, shots=SOURCE_SHOTS, seed=SEED, chunk_size=CHUNK
        ),
        source_path,
    )
    manifest = read_manifest_only(source_path)
    config = parse_config(
        {
            "version": 1,
            "dataset_name": NAME,
            "output_root": str(root / "out"),
            "source": {
                "kind": "legacy_ml_csv",
                "path": str(source_path),
                "expected_content_hash": manifest["content_hash"],
            },
            "generation": {
                "mode": "extend",
                "shots": SHOTS,
                "seed": SEED,
                "chunk_size": CHUNK,
                "require_source_prefix": True,
            },
            "decoder": {"kind": "circuit_dem", "enable_correlations": False},
            "splits": {
                "method": "seeded_permutation",
                "seed": 3,
                "fractions": {"train": 0.7, "validation": 0.15, "test": 0.15},
            },
            "pipeline": {"checkpoint_rows": CHUNK, "feature_rows": 16, "spot_check_rows": 5},
            "sanity_model": {"enabled": True, "seed": 0},
        },
        root,
    )
    source = resolve_source(config)
    model = source.decoder
    graph = summarise_graph(model.dem, model.matching, model.n_detectors)
    slices = time_slices(model.circuit)
    context = FeatureContext(n_detectors=model.n_detectors, slices=slices, graph=graph)

    raw_chunks: list[tuple[np.ndarray, np.ndarray]] = []
    run_ids: list[np.ndarray] = []
    feature_chunks: list[dict[str, np.ndarray]] = []
    offset = 0
    for chunk in iter_chunks(model.circuit, SHOTS, SEED, CHUNK):
        dets = np.asarray(chunk.detectors, dtype=np.uint8)
        obs = np.asarray(chunk.observables, dtype=np.uint8)
        guess, weight = decode_chunk(model, dets)
        truth = truth_from_packed(obs, 1)
        feature_chunks.append(
            {
                "features": extract_features(dets, guess, weight, context),
                "pm_guess": guess,
                "pm_weight": weight,
                "truth": truth,
                "pm_wrong": pm_wrong(guess, truth),
            }
        )
        raw_chunks.append((dets, obs))
        run_ids.append(np.arange(offset, offset + dets.shape[0], dtype=np.int64))
        offset += dets.shape[0]
    assert offset == SHOTS

    codes = assign_splits(
        SHOTS, config.splits.method.value, dict(config.splits.fractions), config.splits.seed
    )
    dataset_dir = root / "out" / NAME
    dataset_dir.mkdir(parents=True)
    features_csv = dataset_dir / f"{NAME}_features.csv"
    write_features_csv(features_csv, feature_chunks, codes)
    write_raw_hdf5(
        dataset_dir / f"{NAME}_raw.h5",
        raw_chunks,
        run_ids,
        codes,
        {
            "n_detectors": model.n_detectors,
            "n_observables": 1,
            "source_content_hash": manifest["content_hash"],
            "config_hash": config_hash(config),
            "decoder_dem_sha256": model.dem_sha256,
        },
    )
    write_decoder_files(dataset_dir, NAME, model, graph, slices=slices)
    write_json(dataset_dir / f"{NAME}_resolved_config.json", resolved_dict(config))

    sanity: dict[str, Any]
    try:
        sanity = run_sanity_models(features_csv, seed=0)
    except SanityModelUnavailableError as error:
        sanity = {"skipped_reason": str(error)}
    write_json(dataset_dir / f"{NAME}_sanity.json", sanity)

    wrong = np.concatenate([c["pm_wrong"] for c in feature_chunks])
    failures = int(wrong.sum())
    interval = clopper_pearson(failures, SHOTS)
    identity = source.identity
    summary: dict[str, Any] = {
        "dataset_name": NAME,
        "additional": False,
        "source": f"{identity.kind.value}: {source_path.name}",
        "source_paths": identity.paths,
        "source_hashes": identity.hashes,
        "provenance_status": "verified by content hash",
        "provenance_limitations": identity.provenance_limitations,
        "distance": identity.distance,
        "rounds": identity.rounds,
        "basis": identity.basis,
        "orientation": None,
        "n_detectors": identity.n_detectors,
        "n_observables": identity.n_observables,
        "noise_model": identity.noise_model,
        "noise_parameters": identity.noise_parameters,
        "n_runs": SHOTS,
        "seed": SEED,
        "chunk_size": CHUNK,
        "versions": library_versions(),
        "decoder_method": str(model.provenance["method"]),
        "decoder_source": identity.dem_available,
        "matching_mode": str(model.provenance["matching_mode"]),
        "decoder_path": f"{NAME}_decoder.dem",
        "decoder_sha256": model.dem_sha256,
        "config_hash": config_hash(config),
        "schema_version": SCHEMA_VERSION,
        "split_method": config.splits.method.value,
        "split_seed": config.splits.seed,
        "split_fractions": dict(config.splits.fractions),
        "split_counts": {
            name: int((codes == SPLIT_CODES[name]).sum())
            for name in SPLIT_NAMES
            if name in config.splits.fractions
        },
        "pm_failures": failures,
        "pm_error_rate": interval.point,
        "pm_ci_low": interval.low,
        "pm_ci_high": interval.high,
        "always_zero_accuracy": 1.0 - interval.point,
        "sanity": sanity,
        "limitations": ["d=3: every node is boundary-adjacent"],
        "deviations": ["raw arrays under /residual"],
    }
    summary_path = dataset_dir / f"{NAME}_summary.json"
    write_json(summary_path, summary)
    note = render_note(summary, summary_sha256=file_sha256(summary_path))
    (dataset_dir / f"{NAME}_note.md").write_text(note, encoding="utf-8")
    return dataset_dir


@pytest.fixture(scope="module")
def published(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_dataset(tmp_path_factory.mktemp("residual"))


@pytest.fixture
def copy(published: Path, tmp_path: Path) -> Path:
    """A private copy per test; the source file stays where the resolved config points."""
    target = tmp_path / NAME
    shutil.copytree(published, target)
    return target


def _read_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        return header, list(reader)


def _write_rows(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _failed(report: ValidationReport) -> dict[str, str]:
    return {c.name: c.detail for c in report.checks if not c.passed}


class TestPublishedDatasetPasses:
    def test_every_check_passes_in_the_brief_order(self, published: Path) -> None:
        report = validate_dataset_dir(published, spot_rows=5)
        assert _failed(report) == {}
        assert report.ok
        assert tuple(c.name for c in report.checks) == CHECK_NAMES
        assert len(CHECK_NAMES) == 15
        assert len(report.spot_checks) >= 5
        assert all(s["ok"] for s in report.spot_checks)
        assert any(s["big_endian_differs"] for s in report.spot_checks)
        assert report.check("source_prefix_matches").passed
        assert "skipped" not in report.check("source_prefix_matches").detail

    def test_rebuild_decoder_false_uses_the_published_dem_coordinates(
        self, published: Path
    ) -> None:
        report = validate_dataset_dir(published, rebuild_decoder=False, spot_rows=6)
        assert _failed(report) == {}
        assert len(report.spot_checks) >= 6

    def test_report_serialises_through_write_json(self, published: Path, tmp_path: Path) -> None:
        report = validate_dataset_dir(published)
        payload = report.to_dict()
        write_json(tmp_path / "v.json", payload)
        loaded = json.loads((tmp_path / "v.json").read_text(encoding="utf-8"))
        assert loaded["ok"] is True
        assert [c["name"] for c in loaded["checks"]] == list(CHECK_NAMES)
        assert len(loaded["spot_checks"]) == len(report.spot_checks)
        assert loaded["dataset_name"] == NAME

    def test_spot_rows_below_one_is_refused(self, published: Path) -> None:
        with pytest.raises(ValueError, match="spot_rows"):
            validate_dataset_dir(published, spot_rows=0)


class TestCorruptionsAreCaught:
    def test_edited_pm_wrong_cell_fails_pm_wrong_consistent(self, copy: Path) -> None:
        path = copy / f"{NAME}_features.csv"
        header, rows = _read_rows(path)
        column = header.index("pm_wrong")
        rows[3][column] = "1" if rows[3][column] == "0" else "0"
        _write_rows(path, header, rows)
        report = validate_dataset_dir(copy)
        assert not report.ok
        assert not report.check("pm_wrong_consistent").passed
        assert report.check("row_count_csv_equals_raw").passed
        assert report.check("schema_exact").passed

    def test_dropped_last_row_fails_row_count(self, copy: Path) -> None:
        path = copy / f"{NAME}_features.csv"
        header, rows = _read_rows(path)
        _write_rows(path, header, rows[:-1])
        report = validate_dataset_dir(copy)
        assert not report.ok
        assert not report.check("row_count_csv_equals_raw").passed
        assert "199" in report.check("row_count_csv_equals_raw").detail

    def test_reordered_columns_fail_schema_exact(self, copy: Path) -> None:
        path = copy / f"{NAME}_features.csv"
        header, rows = _read_rows(path)
        a, b = header.index("n_fired_total"), header.index("frac_fired")
        header[a], header[b] = header[b], header[a]
        for row in rows:
            row[a], row[b] = row[b], row[a]
        _write_rows(path, header, rows)
        report = validate_dataset_dir(copy)
        assert not report.ok
        assert not report.check("schema_exact").passed
        # Columns are located by name, so the values themselves still agree.
        assert report.check("pm_wrong_consistent").passed
        assert report.check("row_count_csv_equals_raw").passed

    def test_big_endian_attr_fails_bit_packing(self, copy: Path) -> None:
        with h5py.File(copy / f"{NAME}_raw.h5", "r+") as handle:
            handle.attrs["bit_order"] = "big"
        report = validate_dataset_dir(copy)
        assert not report.check("bit_packing_little_endian").passed
        assert "bit_order" in report.check("bit_packing_little_endian").detail

    def test_big_endian_repacked_rows_fail_bit_packing(self, copy: Path) -> None:
        """The negative control: data packed with NumPy's default cannot pass."""
        with h5py.File(copy / f"{NAME}_raw.h5", "r+") as handle:
            stored = np.asarray(handle[RAW_GROUP]["detectors"], dtype=np.uint8)
            bits = unpack_bits(stored, N_DETECTORS)
            handle[RAW_GROUP]["detectors"][...] = np.packbits(bits, axis=1, bitorder="big")
        report = validate_dataset_dir(copy)
        assert not report.ok
        assert not report.check("bit_packing_little_endian").passed
        assert not report.check("source_prefix_matches").passed
        assert any(not s["ok"] for s in report.spot_checks)

    def test_tampered_dem_fails_checksums(self, copy: Path) -> None:
        path = copy / f"{NAME}_decoder.dem"
        path.write_bytes(path.read_bytes() + b"# tampered\n")
        report = validate_dataset_dir(copy)
        assert not report.check("checksums_match").passed
        assert "dem" in report.check("checksums_match").detail.lower()

    def test_stale_summary_fails_note_matches_data(self, copy: Path) -> None:
        path = copy / f"{NAME}_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["pm_failures"] = int(summary["pm_failures"]) + 1
        write_json(path, summary)
        report = validate_dataset_dir(copy)
        assert not report.check("note_matches_data").passed

    def test_note_rendered_from_other_numbers_fails_note_matches_data(self, copy: Path) -> None:
        summary_path = copy / f"{NAME}_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["pm_failures"] = int(summary["pm_failures"]) + 1
        note = render_note(summary, summary_sha256=file_sha256(summary_path))
        (copy / f"{NAME}_note.md").write_text(note, encoding="utf-8")
        report = validate_dataset_dir(copy)
        assert not report.check("note_matches_data").passed
        assert "failures" in report.check("note_matches_data").detail.lower()

    def test_corrupted_prefix_row_fails_source_prefix(self, copy: Path) -> None:
        with h5py.File(copy / f"{NAME}_raw.h5", "r+") as handle:
            detectors = handle[RAW_GROUP]["detectors"]
            row = np.asarray(detectors[5], dtype=np.uint8)
            row[1] ^= 0b0001_0100
            detectors[5] = row
        report = validate_dataset_dir(copy)
        assert not report.check("source_prefix_matches").passed

    def test_padding_bits_set_fail_raw_widths(self, copy: Path) -> None:
        with h5py.File(copy / f"{NAME}_raw.h5", "r+") as handle:
            observables = handle[RAW_GROUP]["observables"]
            row = np.asarray(observables[7], dtype=np.uint8)
            row[0] |= 0b1000_0000
            observables[7] = row
        report = validate_dataset_dir(copy)
        assert not report.check("raw_widths_match_true_widths").passed

    def test_wrong_config_hash_fails_checksums(self, copy: Path) -> None:
        path = copy / f"{NAME}_resolved_config.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["config_hash"] = "0" * 64
        write_json(path, payload)
        report = validate_dataset_dir(copy)
        assert not report.check("checksums_match").passed

    def test_fitted_decoder_claim_fails_no_fit_on_held_out(self, copy: Path) -> None:
        path = copy / f"{NAME}_decoder_metadata.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["fitted_in_this_pipeline"] = True
        write_json(path, payload)
        report = validate_dataset_dir(copy)
        assert not report.check("no_fit_on_held_out").passed

    def test_extra_feature_context_field_fails_feature_inputs_audit(self, copy: Path) -> None:
        path = copy / f"{NAME}_decoder_metadata.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["feature_context_fields"] = ["n_detectors", "slices", "graph", "truth"]
        write_json(path, payload)
        report = validate_dataset_dir(copy)
        assert not report.check("feature_inputs_audit").passed

    def test_nan_feature_fails_no_nan_inf(self, copy: Path) -> None:
        path = copy / f"{NAME}_features.csv"
        header, rows = _read_rows(path)
        rows[0][header.index("fired_per_round_std")] = "nan"
        _write_rows(path, header, rows)
        report = validate_dataset_dir(copy)
        assert not report.check("no_nan_inf").passed

    def test_fraction_out_of_range_fails_fractions_in_range(self, copy: Path) -> None:
        path = copy / f"{NAME}_features.csv"
        header, rows = _read_rows(path)
        rows[0][header.index("frac_fired")] = "1.5"
        _write_rows(path, header, rows)
        report = validate_dataset_dir(copy)
        assert not report.check("fractions_in_range").passed

    def test_non_binary_truth_fails_binary_columns(self, copy: Path) -> None:
        path = copy / f"{NAME}_features.csv"
        header, rows = _read_rows(path)
        rows[2][header.index("truth")] = "2"
        _write_rows(path, header, rows)
        report = validate_dataset_dir(copy)
        assert not report.check("binary_columns").passed

    def test_missing_features_csv_raises(self, copy: Path) -> None:
        (copy / f"{NAME}_features.csv").unlink()
        with pytest.raises(FileNotFoundError, match="features"):
            validate_dataset_dir(copy)

    def test_all_columns_constant_is_the_schema(self) -> None:
        assert ALL_COLUMNS[-4:] == ("truth", "pm_wrong", "run_id", "split")


@pytest.fixture
def isolated(tmp_path: Path) -> tuple[Path, Path]:
    """A dataset whose *source* file the test may corrupt or delete.

    The shared ``published`` fixture's resolved config points at the source beside it, so
    a test touching that file would break every other test; these tests need their own.
    """
    return _build_dataset(tmp_path), tmp_path / "source" / "legacy.ml.csv"


class TestChecksumsRecordSourceProblems:
    """A source that is missing or no longer hashes to its manifest is a checksum finding
    like any other and must be listed *beside* the decoder and config-hash findings —
    not propagate and replace the whole detail with one exception name."""

    def test_missing_source_file_is_listed_beside_the_other_problems(
        self, isolated: tuple[Path, Path]
    ) -> None:
        dataset_dir, source = isolated
        source.unlink()
        dem = dataset_dir / f"{NAME}_decoder.dem"
        dem.write_bytes(dem.read_bytes() + b"# tampered\n")
        report = validate_dataset_dir(dataset_dir)
        check = report.check("checksums_match")
        assert not check.passed
        assert "source: FileNotFoundError" in check.detail
        assert "published .dem hashes to" in check.detail

    def test_corrupted_source_row_is_listed_beside_the_other_problems(
        self, isolated: tuple[Path, Path]
    ) -> None:
        dataset_dir, source = isolated
        lines = source.read_text(encoding="utf-8").splitlines()
        cells = lines[2].split(",")
        cells[2] = "1" if cells[2] == "0" else "0"
        lines[2] = ",".join(cells)
        source.write_text("\n".join(lines) + "\n", encoding="utf-8")
        dem = dataset_dir / f"{NAME}_decoder.dem"
        dem.write_bytes(dem.read_bytes() + b"# tampered\n")
        report = validate_dataset_dir(dataset_dir)
        check = report.check("checksums_match")
        assert not check.passed
        assert "source: ValueError" in check.detail
        assert "published .dem hashes to" in check.detail
