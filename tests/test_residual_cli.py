"""The residual module CLI and the pipeline behind it, end to end on a tiny legacy source.

Everything runs in-process through ``typer.testing.CliRunner`` (the pattern of
``tests/test_cli.py``) against a 96-row d=3 source written into ``tmp_path``; nothing under
``data/`` is read. The tests pin the behaviours a crash or a wrong resume would otherwise
hide: artifacts appear only as a complete set, a second build rewrites no chunk, a config
that changed refuses the old checkpoints by name, a source that cannot be resolved never
creates its output directory, and the pilot gate runs inline when no pilot record exists.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from typer.testing import CliRunner, Result

from qecgen.environments import build_single_environment
from qecgen.exporters.ml_csv import MLCSVExporter, read_manifest_only
from qecgen.residual import pipeline
from qecgen.residual.checkpoint import CHUNK_SUFFIX, INDEX_FILENAME, CheckpointIdentityError
from qecgen.residual.cli import app
from qecgen.residual.config import ResidualConfig, load_config
from qecgen.residual.features import ALL_COLUMNS
from qecgen.residual.validation import validate_dataset_dir
from qecgen.residual.writers import RAW_GROUP
from qecgen.residual.zenodo import WillowSourceBlockedError
from qecgen.sampling import unpack_bits

runner = CliRunner()

WIDE = {"COLUMNS": "300"}
"""rich wraps tables at the detected terminal width; a wide one keeps rows greppable."""

SOURCE_SHOTS = 96
SHOTS = 192
CHUNK = 32
SEED = 1
CHECKPOINT_ROWS = 64
FEATURE_ROWS = 16

ARTIFACT_SUFFIXES = (
    "_features.csv",
    "_raw.h5",
    "_note.md",
    "_validation.json",
    "_resolved_config.json",
    "_decoder.dem",
    "_decoder_metadata.json",
    "_summary.json",
    "_sanity.json",
)


def _invoke(*args: str) -> Result:
    return runner.invoke(app, list(args), env=WIDE)


def _combined(result: Result) -> str:
    """stdout plus stderr, across click versions that split them differently."""
    try:
        return result.output + result.stderr
    except ValueError:
        return result.output


def _check_row(result: Result, check: str) -> str:
    """The rendered table row for one validation check (rich draws its own borders, so
    the row is located by the check name rather than by a separator)."""
    return next(line for line in result.output.splitlines() if check in line)


@pytest.fixture(scope="module")
def legacy_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 96-row d=3 legacy ``.ml.csv`` sampled at chunk 32 (call sizes ``[32, 32, 32]``)."""
    root = tmp_path_factory.mktemp("source")
    path = root / "legacy.ml.csv"
    MLCSVExporter().write(
        build_single_environment(
            distance=3, p=0.01, shots=SOURCE_SHOTS, seed=SEED, chunk_size=CHUNK
        ),
        path,
    )
    return path


def _config_payload(
    name: str,
    source: Path,
    output_root: Path,
    *,
    shots: int = SHOTS,
    split_seed: int = 7,
    additional: bool = False,
) -> dict[str, Any]:
    manifest = read_manifest_only(source)
    return {
        "version": 1,
        "dataset_name": name,
        "output_root": str(output_root),
        "additional": additional,
        "source": {
            "kind": "legacy_ml_csv",
            "path": str(source),
            "expected_content_hash": manifest["content_hash"],
        },
        "generation": {
            "mode": "extend",
            "shots": shots,
            "seed": SEED,
            "chunk_size": CHUNK,
            "require_source_prefix": True,
        },
        "decoder": {"kind": "circuit_dem", "enable_correlations": False},
        "splits": {
            "method": "seeded_permutation",
            "seed": split_seed,
            "fractions": {"train": 0.7, "validation": 0.15, "test": 0.15},
        },
        "pipeline": {
            "checkpoint_rows": CHECKPOINT_ROWS,
            "feature_rows": FEATURE_ROWS,
            "spot_check_rows": 5,
        },
        "sanity_model": {"enabled": False, "seed": 0},
    }


def _write_config(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _legacy_config(name: str, root: Path, legacy_source: Path) -> ResidualConfig:
    """A loaded legacy config for ``name`` whose output lands under ``root / "out"``."""
    return load_config(
        _write_config(root / f"{name}.json", _config_payload(name, legacy_source, root / "out")),
        root,
    )


def _willow_payload(name: str, root: Path) -> dict[str, Any]:
    """A syntactically valid hardware_willow config whose table does not exist."""
    return {
        "version": 1,
        "dataset_name": name,
        "output_root": str(root / "out"),
        "source": {
            "kind": "hardware_willow",
            "table": str(root / "missing" / "table.parquet"),
            "circuit": str(root / "missing" / "circuit.stim"),
            "expected": {
                "table_sha256": "0" * 64,
                "circuit_sha256": "1" * 64,
                "distance": 3,
                "basis": "Z",
                "rounds": 10,
                "orientation": "q10_7",
            },
            "zenodo": {
                "record": 13273331,
                "archive": "google_105Q_surface_code_d3_d5_d7.zip",
                "archive_md5_published": "21fa6ad35b395d838ebcdbc92e364a12",
                "cohort_prefix": "google_105Q_surface_code_d3_d5_d7/d3_at_q10_7/Z/r10/",
                "cache_dir": str(root / "cache"),
            },
        },
        "generation": {"mode": "source_rows", "chunk_size": 10000},
        "decoder": {
            "kind": "official_dem",
            "member": "decoding_results/correlated_matching_decoder_with_si1000_prior/error_model.dem",
            "expected_sha256": "2" * 64,
        },
        "splits": {
            "method": "contiguous_blocks",
            "fractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
        },
        "pipeline": {"checkpoint_rows": 10000, "feature_rows": 10000, "spot_check_rows": 8},
        "sanity_model": {"enabled": False},
    }


def _file_digests(directory: Path) -> dict[str, tuple[float, str]]:
    return {
        path.name: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix == CHUNK_SUFFIX
    }


class TestHelp:
    def test_help_exits_zero_and_lists_every_command(self) -> None:
        result = _invoke("--help")
        assert result.exit_code == 0, _combined(result)
        for command in ("inventory", "pilot", "build", "validate", "build-all", "manifest"):
            assert command in result.output


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory, legacy_source: Path) -> dict[str, Any]:
    """inventory -> pilot -> build (with --skip-pilot-gate) on the tiny legacy source."""
    root = tmp_path_factory.mktemp("build")
    name = "legacy_d3_cli"
    config_path = _write_config(
        root / "config.json", _config_payload(name, legacy_source, root / "out")
    )
    inventory = _invoke("inventory", "--config", str(config_path))
    pilot = _invoke("pilot", "--config", str(config_path), "--generated-rows", "64")
    build = _invoke("build", "--config", str(config_path), "--skip-pilot-gate")
    return {
        "root": root,
        "name": name,
        "config_path": config_path,
        "inventory": inventory,
        "pilot": pilot,
        "build": build,
        "dataset_dir": root / "out" / name,
        "checkpoints": root / "out" / ".checkpoints" / name,
    }


class TestEndToEnd:
    def test_inventory_writes_a_record_and_no_output_directory(self, built: dict[str, Any]) -> None:
        result: Result = built["inventory"]
        assert result.exit_code == 0, _combined(result)
        record = json.loads((built["checkpoints"] / "inventory.json").read_text("utf-8"))
        assert record["dataset_name"] == built["name"]
        assert record["identity"]["kind"] == "legacy_ml_csv"
        assert record["identity"]["n_observables"] == 1
        assert len(record["config_hash"]) == 64
        assert "source.kind" in result.output

    def test_pilot_records_counts_alignment_and_resources(self, built: dict[str, Any]) -> None:
        result: Result = built["pilot"]
        assert result.exit_code == 0, _combined(result)
        record = json.loads((built["checkpoints"] / "pilot.json").read_text("utf-8"))
        rows = record["existing_rows"]
        assert rows["n_rows"] == SOURCE_SHOTS
        assert 0 <= rows["pm_wrong"] <= SOURCE_SHOTS
        assert rows["pm_ci_low"] <= rows["pm_error_rate"] <= rows["pm_ci_high"]
        assert "pm_wrong:" in result.output
        alignment = record["alignment"]
        for key in (
            "self_consistency",
            "permutation_control",
            "big_endian_control",
            "marginals",
            "resample",
            "detection_event_rate",
        ):
            assert key in alignment, key
        assert alignment["self_consistency"]["equal"] is True
        assert alignment["marginals"]["n_detectors"] == 24
        assert alignment["resample"]["n_rows"] == 16_000
        assert isinstance(record["resources_sufficient"], bool)
        assert record["throughput"]["generated_rows"] == 64
        assert record["projection"]["rows"] == SHOTS

    def test_build_publishes_the_full_artifact_set(self, built: dict[str, Any]) -> None:
        result: Result = built["build"]
        assert result.exit_code == 0, _combined(result)
        dataset_dir: Path = built["dataset_dir"]
        name: str = built["name"]
        for suffix in ARTIFACT_SUFFIXES:
            assert (dataset_dir / f"{name}{suffix}").is_file(), suffix
        assert not list(dataset_dir.glob(".qecgen-partial-*"))
        assert (built["root"] / "out" / "MANIFEST.md").is_file()
        assert "pm_wrong:" in result.output
        with (dataset_dir / f"{name}_features.csv").open(encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split(",")
            n_rows = sum(1 for _ in fh)
        assert tuple(header) == ALL_COLUMNS
        assert n_rows == SHOTS
        with h5py.File(dataset_dir / f"{name}_raw.h5", "r") as handle:
            assert handle[RAW_GROUP]["detectors"].shape[0] == SHOTS
            assert handle.attrs["bit_order"] == "little"
        validation = json.loads((dataset_dir / f"{name}_validation.json").read_text("utf-8"))
        assert validation["ok"] is True
        sanity = json.loads((dataset_dir / f"{name}_sanity.json").read_text("utf-8"))
        assert "skipped_reason" in sanity

    def test_checkpoints_live_outside_the_published_directory(self, built: dict[str, Any]) -> None:
        checkpoints: Path = built["checkpoints"]
        chunks = sorted(p.name for p in checkpoints.iterdir() if p.suffix == CHUNK_SUFFIX)
        assert chunks == [
            f"{stage}_chunk_{i:05d}.chk" for stage in ("feat", "raw") for i in range(3)
        ]
        assert not list(checkpoints.glob("*.npz"))
        assert not list(built["dataset_dir"].glob("*.chk"))

    def test_validate_command_passes(self, built: dict[str, Any]) -> None:
        result = _invoke("validate", "--output", str(built["dataset_dir"]))
        assert result.exit_code == 0, _combined(result)
        assert re.search(r"^ok: True \(\d+ spot rows\)$", result.output, re.MULTILINE)
        assert " NO " not in _check_row(result, "checksums_match")

    def test_validate_fails_a_dataset_whose_resolved_seed_changed(
        self, built: dict[str, Any], tmp_path: Path
    ) -> None:
        """The recorded hash no longer matches the configuration beside it, so every
        check that needs the configuration fails and the footer says so."""
        target = tmp_path / built["name"]
        shutil.copytree(built["dataset_dir"], target)
        path = target / f"{built['name']}_resolved_config.json"
        payload = json.loads(path.read_text("utf-8"))
        payload["generation"]["seed"] = SEED + 1
        path.write_text(json.dumps(payload), encoding="utf-8")
        result = _invoke("validate", "--output", str(target))
        assert result.exit_code == 1
        assert re.search(r"^ok: False \(\d+ spot rows\)$", result.output, re.MULTILINE)
        assert " NO " in _check_row(result, "checksums_match")

    def test_summary_pilot_block_names_the_pilot_it_reports(self, built: dict[str, Any]) -> None:
        summary = json.loads(
            (built["dataset_dir"] / f"{built['name']}_summary.json").read_text("utf-8")
        )
        record = json.loads((built["checkpoints"] / "pilot.json").read_text("utf-8"))
        assert summary["pilot"]["pilot_config_hash"] == record["config_hash"]
        assert summary["pilot"]["pilot_config_hash"] == summary["config_hash"]
        assert summary["pilot"]["pm_wrong_source_rows"] == record["existing_rows"]["pm_wrong"]

    def test_manifest_lists_the_dataset(self, built: dict[str, Any]) -> None:
        text = (built["root"] / "out" / "MANIFEST.md").read_text("utf-8")
        assert built["name"] in text
        assert "1 completed, 0 blocked, 0 failed" in text

    def test_rebuild_resumes_without_rewriting_chunks(self, built: dict[str, Any]) -> None:
        before = _file_digests(built["checkpoints"])
        assert len(before) == 6
        result = _invoke("build", "--config", str(built["config_path"]), "--skip-pilot-gate")
        assert result.exit_code == 0, _combined(result)
        assert _file_digests(built["checkpoints"]) == before
        assert "resum" in result.output.lower()
        report = validate_dataset_dir(built["dataset_dir"], spot_rows=5)
        assert report.ok, [c for c in report.checks if not c.passed]

    def test_changed_config_refuses_the_old_checkpoints(
        self, built: dict[str, Any], legacy_source: Path
    ) -> None:
        payload = _config_payload(built["name"], legacy_source, built["root"] / "out", split_seed=8)
        changed = _write_config(built["root"] / "changed.json", payload)
        result = _invoke("build", "--config", str(changed), "--skip-pilot-gate")
        assert result.exit_code == 1
        text = _combined(result)
        assert "config_hash" in text
        assert "different run" in text
        with pytest.raises(CheckpointIdentityError, match="config_hash"):
            pipeline.build(load_config(changed, built["root"]), skip_pilot_gate=True)


class TestPilotGate:
    def test_build_without_a_pilot_record_runs_the_pilot_inline(
        self, tmp_path: Path, legacy_source: Path
    ) -> None:
        name = "legacy_d3_gate"
        config_path = _write_config(
            tmp_path / "config.json", _config_payload(name, legacy_source, tmp_path / "out")
        )
        checkpoints = tmp_path / "out" / ".checkpoints" / name
        assert not (checkpoints / "pilot.json").exists()
        result = _invoke("build", "--config", str(config_path), "--pilot-rows", "64")
        assert result.exit_code == 0, _combined(result)
        record = json.loads((checkpoints / "pilot.json").read_text("utf-8"))
        assert record["resources_sufficient"] is True
        assert (tmp_path / "out" / name / f"{name}_features.csv").is_file()
        assert "pilot" in result.output.lower()

    def test_insufficient_resources_stop_the_build_before_staging(
        self, tmp_path: Path, legacy_source: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        name = "legacy_d3_nogate"
        config = load_config(
            _write_config(
                tmp_path / "config.json", _config_payload(name, legacy_source, tmp_path / "out")
            ),
            tmp_path,
        )
        pipeline.pilot(config, generated_rows=64)
        monkeypatch.setattr(pipeline, "_free_disk_bytes", lambda _path: 0)
        with pytest.raises(pipeline.PilotGateError, match="free disk"):
            pipeline.build(config)
        assert not (tmp_path / "out" / name).exists()

    def test_another_configurations_pilot_record_is_not_reported(
        self, tmp_path: Path, legacy_source: Path
    ) -> None:
        """Under ``--skip-pilot-gate`` no gate replaces a stale record, so ``build`` used
        to carry a foreign pilot's source-row failure counts into the summary."""
        name = "legacy_d3_stale_pilot"
        config = _legacy_config(name, tmp_path, legacy_source)
        checkpoints = tmp_path / "out" / ".checkpoints" / name
        checkpoints.mkdir(parents=True)
        stale = {
            "dataset_name": name,
            "config_hash": "0" * 64,
            "source_hash": "1" * 64,
            "decoder_dem_sha256": "2" * 64,
            "existing_rows": {
                "n_rows": SOURCE_SHOTS,
                "pm_wrong": SOURCE_SHOTS,
                "pm_wrong_fraction": 1.0,
                "pm_error_rate": 1.0,
                "pm_ci_low": 0.9,
                "pm_ci_high": 1.0,
                "accuracy": 0.0,
                "detection_event_rate": 0.5,
            },
            "alignment": {"concerns": ["planted"]},
            "expectation": {"applied": False},
            "resources_sufficient": True,
            "projection": {"bytes": 0},
        }
        (checkpoints / "pilot.json").write_text(json.dumps(stale), encoding="utf-8")
        published = pipeline.build(config, skip_pilot_gate=True)
        summary = json.loads((published / f"{name}_summary.json").read_text("utf-8"))
        assert summary["pilot"] is None

    def test_pilot_unpacks_stored_rows_in_feature_row_blocks(
        self, tmp_path: Path, legacy_source: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A checkpoint chunk of the d=9 source unpacks to 256 MB; the pilot must never
        unpack more than ``feature_rows`` rows at once."""
        config = _legacy_config("legacy_d3_blocks", tmp_path, legacy_source)
        seen: list[int] = []

        def recording(packed: np.ndarray, n_bits: int) -> np.ndarray:
            seen.append(int(packed.shape[0]))
            return unpack_bits(packed, n_bits)

        monkeypatch.setattr(pipeline, "unpack_bits", recording)
        record = pipeline.pilot(config, generated_rows=64, resample_rows=64)
        assert record["existing_rows"]["n_rows"] == SOURCE_SHOTS
        assert seen
        assert max(seen) <= FEATURE_ROWS


class TestResolveOrBlock:
    def test_only_refusals_become_blocked(
        self, tmp_path: Path, legacy_source: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bug in resolution must not be filed under "blocked: fix the input"."""
        name = "legacy_d3_resolve"
        config = _legacy_config(name, tmp_path, legacy_source)

        def bug(*_args: Any, **_kwargs: Any) -> None:
            raise TypeError("planted bug")

        monkeypatch.setattr(pipeline, "resolve_source", bug)
        with pytest.raises(TypeError, match="planted bug"):
            pipeline.build(config, skip_pilot_gate=True)

        for planted in (
            ValueError("planted refusal"),
            FileNotFoundError("planted missing file"),
            WillowSourceBlockedError("planted zenodo failure"),
        ):

            def refuse(*_args: Any, _error: Exception = planted, **_kwargs: Any) -> None:
                raise _error

            monkeypatch.setattr(pipeline, "resolve_source", refuse)
            with pytest.raises(pipeline.SourceBlockedError) as caught:
                pipeline.build(config, skip_pilot_gate=True)
            assert caught.value.reason == f"{type(planted).__name__}: {planted}"
        assert not (tmp_path / "out" / name).exists()


class TestAtomicPublication:
    def test_planted_stage_c_exception_leaves_nothing_published(
        self, tmp_path: Path, legacy_source: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        name = "legacy_d3_crash"
        config = load_config(
            _write_config(
                tmp_path / "config.json", _config_payload(name, legacy_source, tmp_path / "out")
            ),
            tmp_path,
        )

        def explode(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("planted stage C failure")

        monkeypatch.setattr(pipeline, "write_decoder_files", explode)
        with pytest.raises(RuntimeError, match="planted"):
            pipeline.build(config, skip_pilot_gate=True)
        dataset_dir = tmp_path / "out" / name
        assert not list(dataset_dir.glob("*_features.csv"))
        assert not list(dataset_dir.glob("*_raw.h5"))
        assert not list(dataset_dir.glob(".qecgen-partial-*"))
        assert not (tmp_path / "out" / "MANIFEST.md").exists()
        # The checkpoints survive so the rerun only redoes stage C.
        chunks = tmp_path / "out" / ".checkpoints" / name
        assert (chunks / INDEX_FILENAME).is_file()
        monkeypatch.undo()
        published = pipeline.build(config, skip_pilot_gate=True)
        assert (published / f"{name}_features.csv").is_file()


class TestBuildAll:
    def test_two_configs_write_one_manifest_with_two_rows(
        self, tmp_path: Path, legacy_source: Path
    ) -> None:
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        out = tmp_path / "out"
        _write_config(
            config_dir / "b_second.json",
            _config_payload("second_ds", legacy_source, out, split_seed=11, additional=True),
        )
        _write_config(config_dir / "a_first.json", _config_payload("first_ds", legacy_source, out))
        result = _invoke("build-all", "--config-dir", str(config_dir), "--skip-pilot-gate")
        assert result.exit_code == 0, _combined(result)
        text = (out / "MANIFEST.md").read_text("utf-8")
        rows = [line for line in text.splitlines() if line.startswith("| ") and "_ds" in line]
        assert len(rows) == 2
        assert rows[0].startswith("| first_ds |")
        assert rows[1].startswith("| second_ds (additional) |")
        assert "completed (additional)" in rows[1]
        assert not list(out.glob(".qecgen-partial-*"))

    def test_missing_willow_table_is_reported_blocked_without_a_directory(
        self, tmp_path: Path
    ) -> None:
        name = "willow_missing"
        config_path = _write_config(tmp_path / "willow.json", _willow_payload(name, tmp_path))
        result = _invoke("build", "--config", str(config_path))
        assert result.exit_code == 1
        text = _combined(result)
        assert "blocked" in text.lower()
        assert "does not exist" in text
        assert not (tmp_path / "out" / name).exists()

    def test_build_all_continues_past_a_blocked_dataset(
        self, tmp_path: Path, legacy_source: Path
    ) -> None:
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        out = tmp_path / "out"
        _write_config(config_dir / "willow.json", _willow_payload("willow_blocked", tmp_path))
        _write_config(config_dir / "legacy.json", _config_payload("legacy_ok", legacy_source, out))
        results = pipeline.build_all(config_dir, repo_root=tmp_path, skip_pilot_gate=True)
        statuses = {r["dataset_name"]: r["status"] for r in results}
        assert statuses == {"willow_blocked": "blocked", "legacy_ok": "completed"}
        assert not (out / "willow_blocked").exists()
        text = (out / "MANIFEST.md").read_text("utf-8")
        assert "1 completed, 1 blocked, 0 failed" in text
        assert "| willow_blocked |" in text
        assert "blocked:" in text

    def test_build_all_records_a_refusal_as_failed_not_blocked(
        self, tmp_path: Path, legacy_source: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        out = tmp_path / "out"
        _write_config(
            config_dir / "legacy.json", _config_payload("legacy_fails", legacy_source, out)
        )

        def refuse(*_args: Any, **_kwargs: Any) -> None:
            raise ValueError("planted refusal")

        monkeypatch.setattr(pipeline, "write_decoder_files", refuse)
        results = pipeline.build_all(config_dir, repo_root=tmp_path, skip_pilot_gate=True)
        assert [r["status"] for r in results] == ["failed"]
        assert "planted refusal" in results[0]["reason"]
        assert not list((out / "legacy_fails").glob("*_features.csv"))
        assert not list((out / "legacy_fails").glob(".qecgen-partial-*"))
        text = (out / "MANIFEST.md").read_text("utf-8")
        assert "0 completed, 0 blocked, 1 failed" in text
        assert "| legacy_fails |" in text
        assert "failed: build failed: ValueError: planted refusal" in text
        assert "blocked:" not in text

    def test_build_all_lets_a_bug_propagate(
        self, tmp_path: Path, legacy_source: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        out = tmp_path / "out"
        _write_config(config_dir / "legacy.json", _config_payload("legacy_bug", legacy_source, out))

        def bug(*_args: Any, **_kwargs: Any) -> None:
            raise TypeError("planted bug")

        monkeypatch.setattr(pipeline, "write_decoder_files", bug)
        with pytest.raises(TypeError, match="planted bug"):
            pipeline.build_all(config_dir, repo_root=tmp_path, skip_pilot_gate=True)
        assert not (out / "MANIFEST.md").exists()

    def test_build_all_refuses_an_empty_config_dir(self, tmp_path: Path) -> None:
        empty = tmp_path / "configs"
        empty.mkdir()
        result = _invoke("build-all", "--config-dir", str(empty))
        assert result.exit_code == 1
        text = _combined(result)
        assert "refused: FileNotFoundError" in text
        assert "no *.json" in text
        assert isinstance(result.exception, SystemExit)

    def test_build_all_refuses_a_malformed_config(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "broken.json").write_text("{not json", encoding="utf-8")
        result = _invoke("build-all", "--config-dir", str(config_dir))
        assert result.exit_code == 1
        text = _combined(result)
        assert "refused: ConfigError" in text
        assert "not valid JSON" in text
        assert isinstance(result.exception, SystemExit)

    def test_manifest_refuses_a_malformed_summary(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        (out / "ds").mkdir(parents=True)
        (out / "ds" / "ds_summary.json").write_text("{broken", encoding="utf-8")
        result = _invoke("manifest", "--output-root", str(out))
        assert result.exit_code == 1
        assert "refused: JSONDecodeError" in _combined(result)
        assert isinstance(result.exception, SystemExit)
        assert not (out / "MANIFEST.md").exists()


class TestPeakWorkingSet:
    def test_peak_working_set_is_a_positive_byte_count(self) -> None:
        peak = pipeline.peak_working_set_bytes()
        assert peak is None or peak > 0
        if peak is not None:
            assert peak > np.zeros(1_000_000, dtype=np.uint8).nbytes
