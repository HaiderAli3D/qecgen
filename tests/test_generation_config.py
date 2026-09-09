"""Integration guards for configured generation, including frozen legacy streams."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from qecgen.circuits import Basis, build_circuit
from qecgen.cli import app
from qecgen.configuration import expand_sweep, normalize_config, prepare_generation, read_config
from qecgen.dataset import content_hash
from qecgen.exporters import EXPORTERS, get_exporter
from qecgen.hardware import load_willow_derived
from qecgen.run import ConfiguredSpec, run
from qecgen.validate import validate_dataset


def example(path: Path, mode: str = "device", fmt: str = "hdf5") -> dict[str, Any]:
    result: dict[str, Any] = {
        "version": 1,
        "mode": mode,
        "output": {"path": str(path), "format": fmt},
        "sampling": {"shots": 257, "seed": 123, "chunk_size": 31},
        "circuit": {"distance": 3, "rounds": 3},
    }
    if mode == "device":
        result["noise"] = {"version": 1, "probabilities": {"measurement": 0.03}}
        result["parameter_provenance"] = {"kind": "scenario", "description": "Test scenario"}
    else:
        result["legacy"] = {"p": 0.01}
    return result


@pytest.mark.parametrize("fmt", list(EXPORTERS))
@pytest.mark.parametrize("dynamic", [False, True])
def test_configured_exports_preserve_realized_arrays_and_config(
    tmp_path: Path, fmt: str, dynamic: bool
) -> None:
    exporter = get_exporter(fmt)
    config = example(tmp_path / ("shots" + exporter.extension), fmt=fmt)
    if dynamic:
        config["noise"]["bursts"] = {
            "enabled": True,
            "qubits": [1, 3],
            "pauli": "X",
            "onset_probability": 0.1,
            "recovery_probability": 0.2,
            "effect_probability": 0.3,
        }
    config["output"]["structure"] = "coords"
    spec = ConfiguredSpec(config)
    batches = list(prepare_generation(config).chunks)
    expected = np.concatenate([batch.detectors for batch in batches])
    files = run(spec)
    dataset = exporter.read(files[0].path)
    assert np.array_equal(dataset.detectors, expected)
    assert dataset.meta.generation_config == spec.config
    assert dataset.meta.to_json_dict()["manifest_version"] == 2
    assert dataset.meta.generation_audit is not None
    assert dataset.meta.generation_audit["has_dynamic"] is dynamic
    assert dataset.meta.environments[0].p is None
    assert validate_dataset(dataset).ok


@pytest.mark.parametrize("case_index", range(24))
def test_legacy_preset_matches_frozen_pre_overhaul_streams(tmp_path: Path, case_index: int) -> None:
    from research.realism.legacy import runtime_contract

    golden = json.loads(
        (Path(__file__).parents[1] / "research/realism/legacy_golden.json").read_text()
    )
    current = runtime_contract()
    for key in ("stim", "numpy"):
        assert current[key] == golden["runtime"][key]
    for key in ("platform", "machine", "byteorder", "stim_march"):
        if current[key] != golden["runtime"][key]:
            pytest.skip(f"Frozen Stim stream requires matching {key}")
    case = golden["cases"][case_index]
    old = case["config"]
    config = example(tmp_path / "legacy.h5", "legacy")
    config["legacy"] = {key: old[key] for key in ("p", "noise_model")}
    config["sampling"] = {
        key: old[key] for key in ("shots", "seed", "chunk_size", "emit_mechanisms")
    }
    config["circuit"] = {key: old[key] for key in ("distance", "rounds", "basis", "rotated")}
    chunks = list(prepare_generation(config).chunks)
    det = np.concatenate([batch.detectors for batch in chunks])
    obs = np.concatenate([batch.observables for batch in chunks])
    mech = (
        np.concatenate([batch.mechanisms for batch in chunks]) if old["emit_mechanisms"] else None
    )
    assert content_hash(det, obs, mechanisms=mech) == case["fingerprint"]["content_hash"]


def test_static_contract_b_is_one_consistent_dem_draw(tmp_path: Path) -> None:
    config = example(tmp_path / "b.npz", fmt="npz")
    config["sampling"]["emit_mechanisms"] = True
    config["output"]["structure"] = "full"
    artifact = run(ConfiguredSpec(config))[0]
    dataset = get_exporter("npz").read(artifact.path)
    assert dataset.mechanisms is not None
    assert dataset.meta.generation_audit is not None
    assert dataset.meta.generation_audit["contract"] == "B"
    assert dataset.meta.generation_audit["rng_streams"] == ["stim_dem"]
    sources = dataset.meta.generation_audit["engine_source_sha256"]
    assert (
        sources["noise.py"]
        == hashlib.sha256((Path(__file__).parents[1] / "qecgen/noise.py").read_bytes()).hexdigest()
    )
    assert validate_dataset(dataset).ok


def test_cancelled_stream_preserves_existing_destination(tmp_path: Path) -> None:
    path = tmp_path / "old.h5"
    path.write_bytes(b"existing user artifact")

    def cancel(_: int) -> None:
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        run(ConfiguredSpec(example(path)), cancel)
    assert path.read_bytes() == b"existing user artifact"
    assert list(tmp_path.iterdir()) == [path]


def test_unknown_fields_wrong_scalars_and_conflicting_modes_are_refused(tmp_path: Path) -> None:
    config = example(tmp_path / "a.h5")
    for field, value in (("shots", True), ("seed", -1), ("chunk_size", 0)):
        changed = copy.deepcopy(config)
        changed["sampling"][field] = value
        with pytest.raises(ValueError):
            normalize_config(changed)
    config["unknown"] = 1
    with pytest.raises(ValueError, match="Unknown"):
        normalize_config(config)
    del config["unknown"]
    config["legacy"] = {"p": 0.1}
    with pytest.raises(ValueError, match="cannot be used"):
        normalize_config(config)


def test_companion_and_configuration_paths_cannot_be_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "a.ml.csv"
    companion = get_exporter("ml_csv").companions(output)[0]
    config = example(output, fmt="ml_csv")
    config["circuit"].update(stim_file=str(companion), sha256="0" * 64)
    with pytest.raises(ValueError, match="companion"):
        normalize_config(config)
    del config["circuit"]["stim_file"]
    del config["circuit"]["sha256"]
    companion.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="configuration file"):
        read_config(companion)


def test_custom_circuit_cannot_lie_about_basis_or_distance(tmp_path: Path) -> None:
    circuit, _ = build_circuit(5, 0, rounds=3, basis=Basis.X)
    path = tmp_path / "external.stim"
    path.write_text(str(circuit))
    config = example(tmp_path / "a.h5")
    config["circuit"].update(
        stim_file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="Unsupported external circuit identity"):
        prepare_generation(config)


def test_cli_records_config_and_inspects_unknown_uniform_rate(tmp_path: Path) -> None:
    output = tmp_path / "a.h5"
    path = tmp_path / "config.json"
    config = example(output)
    description = "An uninterrupted provenance identifier: " + "abcdef0123456789" * 12
    config["parameter_provenance"]["description"] = description
    path.write_text(json.dumps(config))
    runner = CliRunner()
    generated = runner.invoke(app, ["generate-config", "--config", str(path)])
    assert generated.exit_code == 0, generated.output
    assert description in generated.output
    inspected = runner.invoke(app, ["inspect", str(output)], env={"COLUMNS": "240"})
    assert inspected.exit_code == 0, inspected.output
    assert "not applicable" in inspected.output
    path.write_text('{"version":1,"version":2}')
    assert runner.invoke(app, ["generate-config", "--config", str(path)]).exit_code != 0


def test_unlisted_user_table_does_not_inherit_google_identity(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    circuit, _ = build_circuit(3, 0, rounds=3)
    source = tmp_path / "source.stim"
    source.write_text(str(circuit))
    table = tmp_path / "source.parquet"
    pq.write_table(
        pa.table(
            {
                "shot": [0, 1],
                "detectors": [[False] * circuit.num_detectors] * 2,
                "observable": [False, True],
                "distance": [3, 3],
                "basis": ["Z", "Z"],
                "rounds": [3, 3],
                "orientation": ["custom", "custom"],
            }
        ),
        table,
    )
    expected = {
        "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        "circuit_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "distance": 3,
        "basis": "Z",
        "rounds": 3,
        "orientation": "custom",
    }
    cohort = load_willow_derived(table, source, expected)
    assert cohort.source["publisher_identity_verified"] is False
    assert cohort.source["original_doi"] is None
    assert cohort.source["mirror_revision"] is None


def test_mechanism_sweep_is_replayable_and_preserves_other_parameters(tmp_path: Path) -> None:
    config = example(tmp_path / "sweep.h5")
    first = expand_sweep(config, "noise.probabilities.measurement", [0.01, 0.02, 0.03])
    second = expand_sweep(config, "noise.probabilities.measurement", [0.01, 0.02, 0.03])
    assert first == second
    assert [item["noise"]["probabilities"]["measurement"] for item in first] == [0.01, 0.02, 0.03]
    assert len({item["sampling"]["seed"] for item in first}) == 3
    assert len({item["output"]["path"] for item in first}) == 3
    assert config["noise"]["probabilities"]["measurement"] == 0.03
    with pytest.raises(ValueError, match="dataset identity"):
        expand_sweep(config, "circuit.distance", [5])
    with pytest.raises(ValueError):
        expand_sweep(config, "noise.probabilities.measurement", [1.2])
