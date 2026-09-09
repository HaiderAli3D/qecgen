"""Versioned metadata must never turn a device or hardware file into uniform noise."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from qecgen.circuits import ChannelVector, NoiseModel
from qecgen.dataset import (
    Contract,
    DatasetMeta,
    EnvironmentModel,
    EnvironmentSpec,
    InMemoryDataset,
    StructureLevel,
)
from qecgen.environments import build_single_environment
from qecgen.exporters import EXPORTERS, get_exporter
from qecgen.qa import benchmark_dataset, estimate_environment_rates
from qecgen.validate import validate_dataset


@pytest.fixture(scope="module")
def legacy() -> InMemoryDataset:
    return build_single_environment(distance=3, p=0.05, shots=16, seed=42, chunk_size=8)


def common_config(legacy: InMemoryDataset, mode: str) -> dict[str, Any]:
    meta = legacy.meta
    return {
        "version": 1,
        "mode": mode,
        "circuit": {
            "distance": meta.distance,
            "rounds": meta.rounds,
            "basis": str(meta.basis),
            "rotated": meta.rotated,
        },
        "sampling": {
            "shots": meta.shots,
            "seed": meta.seed,
            "chunk_size": meta.chunk_size,
            "emit_mechanisms": False,
        },
        "output": {"path": "output.h5", "format": "hdf5", "structure": "none"},
    }


def configured(legacy: InMemoryDataset, model: EnvironmentModel) -> InMemoryDataset:
    mode = "device" if model is EnvironmentModel.DEVICE_PROFILE else "hardware"
    config = common_config(legacy, mode)
    if mode == "device":
        config["noise"] = {"version": 1, "probabilities": {"measurement": 0.05}}
        config["parameter_provenance"] = {
            "kind": "scenario",
            "description": "Metadata test fixture",
        }
    else:
        config["hardware"] = {
            "table": "source.parquet",
            "circuit": "source.stim",
            "offset": 0,
            "expected": {
                "table_sha256": "a" * 64,
                "circuit_sha256": "b" * 64,
                "distance": legacy.meta.distance,
                "rounds": legacy.meta.rounds,
                "basis": str(legacy.meta.basis).upper(),
                "orientation": "d3_at_q10_7",
            },
        }
    axis = "profile" if mode == "device" else "source_rows"
    env = replace(
        legacy.meta.environments[0],
        p=None,
        noise_model=model,
        channels=None,
        axis=axis,
        axis_value=0.0,
    )
    meta = replace(
        legacy.meta,
        environments=(env,),
        generation_config=config,
        generation_audit={"has_dynamic": False},
        drift_axis=axis,
    )
    return replace(legacy, meta=meta)


@pytest.mark.parametrize("name", sorted(EXPORTERS))
@pytest.mark.parametrize("model", list(EnvironmentModel))
def test_configured_metadata_roundtrips_every_exporter(
    legacy: InMemoryDataset, name: str, model: EnvironmentModel, tmp_path: Path
) -> None:
    dataset = configured(legacy, model)
    exporter = get_exporter(name)
    path = tmp_path / f"configured{exporter.extension}"
    exporter.write(dataset, path, StructureLevel.NONE)
    restored = exporter.read(path)
    assert restored.meta.to_json_dict() == dataset.meta.to_json_dict()
    assert restored.meta.environments[0].channels is None
    assert restored.meta.environments[0].p is None
    assert restored.meta.environments[0].noise_model is model
    assert np.array_equal(restored.detectors, dataset.detectors)
    assert np.array_equal(restored.observables, dataset.observables)
    assert restored.compute_content_hash() == legacy.compute_content_hash()
    assert validate_dataset(restored).ok
    # This is the legacy reader's enum conversion. It refuses rather than assigning
    # hardware a legacy model, even if an old consumer ignores manifest_version.
    with pytest.raises(ValueError):
        NoiseModel(restored.meta.to_json_dict()["environments"][0]["noise_model"])


def test_legacy_configuration_is_additive_and_preserves_arrays(legacy: InMemoryDataset) -> None:
    before = legacy.meta.to_json_dict()
    assert "manifest_version" not in before
    assert "generation_config" not in before
    assert "generation_audit" not in before
    config = common_config(legacy, "legacy")
    config["legacy"] = {"noise_model": "stim_uniform_circuit_level", "p": 0.05}
    meta = replace(legacy.meta, generation_config=config)
    stored = meta.to_json_dict()
    assert stored.pop("manifest_version") == 2
    assert stored.pop("generation_config") == config
    assert stored == before
    assert DatasetMeta.from_json(meta.to_json()).generation_config == config
    assert replace(legacy, meta=meta).compute_content_hash() == legacy.compute_content_hash()


@pytest.mark.parametrize("version", [0, 3, "2", True, None])
def test_unknown_manifest_versions_are_refused(legacy: InMemoryDataset, version: Any) -> None:
    stored = legacy.meta.to_json_dict()
    stored["manifest_version"] = version
    with pytest.raises(ValueError, match="unsupported manifest_version"):
        DatasetMeta.from_json_dict(stored)


@pytest.mark.parametrize("model", list(EnvironmentModel))
def test_configured_files_require_explicit_config_version(
    legacy: InMemoryDataset, model: EnvironmentModel
) -> None:
    stored = configured(legacy, model).meta.to_json_dict()
    stored.pop("manifest_version")
    with pytest.raises(ValueError, match="require manifest_version 2"):
        DatasetMeta.from_json_dict(stored)
    stored["manifest_version"] = 2
    stored.pop("generation_config")
    with pytest.raises(ValueError, match="requires generation_config"):
        DatasetMeta.from_json_dict(stored)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("version", 2, "unsupported generation_config version"),
        ("version", True, "unsupported generation_config version"),
        ("mode", "legacy", "disagrees"),
        ("mode", [], "mode must be"),
        ("noise", {}, "nonempty noise"),
        ("noise", "profile", "nonempty noise"),
        ("noise", {"version": 2}, "Profile version"),
    ],
)
def test_config_is_typed_and_agrees_with_model(
    legacy: InMemoryDataset, key: str, value: Any, message: str
) -> None:
    stored = configured(legacy, EnvironmentModel.DEVICE_PROFILE).meta.to_json_dict()
    stored["generation_config"][key] = value
    with pytest.raises(ValueError, match=message):
        DatasetMeta.from_json_dict(stored)


@pytest.mark.parametrize("model", list(EnvironmentModel))
def test_unknown_channels_cannot_be_reported_as_noiseless(
    legacy: InMemoryDataset, model: EnvironmentModel
) -> None:
    dataset = configured(legacy, model)
    assert np.any(dataset.detectors)
    assert validate_dataset(dataset).ok
    with pytest.raises(ValueError, match="null p and channels"):
        replace(dataset.meta.environments[0], channels=ChannelVector())
    with pytest.raises(ValueError, match="null p and channels"):
        replace(dataset.meta.environments[0], p=0.0)
    with pytest.raises(ValueError, match="require p"):
        replace(legacy.meta.environments[0], channels=None)


@pytest.mark.parametrize("model", list(EnvironmentModel))
def test_legacy_qa_refuses_new_models(legacy: InMemoryDataset, model: EnvironmentModel) -> None:
    dataset = configured(legacy, model)
    with pytest.raises(ValueError, match="legacy QA/benchmark does not support"):
        estimate_environment_rates(dataset.meta, max_shots=1)
    with pytest.raises(ValueError, match="legacy QA/benchmark does not support"):
        benchmark_dataset(dataset)


@pytest.mark.parametrize(
    "payload",
    [
        {"circuit_text": "R 0"},
        {"other": [{"dem_text": "error(0.1) D0"}]},
        {"other": {"circuit": "R 0"}},
        {"dem": "error(0.1) D0"},
        {"description": "Example\nR 0\nM 0"},
    ],
)
def test_nested_simulator_text_never_reaches_manifest(
    legacy: InMemoryDataset, payload: dict[str, Any]
) -> None:
    stored = configured(legacy, EnvironmentModel.DEVICE_PROFILE).meta.to_json_dict()
    stored["generation_config"]["extra"] = payload
    with pytest.raises(ValueError, match="belongs in provenance"):
        DatasetMeta.from_json_dict(stored)


def test_device_static_contract_b_is_allowed_but_hardware_is_not(legacy: InMemoryDataset) -> None:
    device = configured(legacy, EnvironmentModel.DEVICE_PROFILE)
    config = copy.deepcopy(device.meta.generation_config)
    assert config is not None
    config["sampling"]["emit_mechanisms"] = True
    assert (
        replace(device.meta, generation_config=config, contract=Contract.DEM_MECHANISM).contract
        is Contract.DEM_MECHANISM
    )
    hardware = configured(legacy, EnvironmentModel.HARDWARE)
    with pytest.raises(ValueError, match="cannot carry DEM mechanism"):
        replace(hardware.meta, contract=Contract.DEM_MECHANISM)


def test_dynamic_model_cannot_hide_behind_static_audit(legacy: InMemoryDataset) -> None:
    device = configured(legacy, EnvironmentModel.DEVICE_PROFILE)
    config = copy.deepcopy(device.meta.generation_config)
    assert config is not None
    config["noise"]["bursts"] = {
        "enabled": True,
        "qubits": [1],
        "pauli": "X",
        "onset_probability": 0.01,
        "recovery_probability": 0.5,
        "effect_probability": 0.1,
    }
    with pytest.raises(ValueError, match="has_dynamic disagrees"):
        replace(device.meta, generation_config=config)
    dynamic = replace(device.meta, generation_config=config, generation_audit={"has_dynamic": True})
    with pytest.raises(ValueError, match="cannot carry DEM mechanism"):
        replace(dynamic, contract=Contract.DEM_MECHANISM)
    with pytest.raises(ValueError, match="cannot claim an exact independent DEM"):
        replace(dynamic, structure_level=StructureLevel.DEM)
    # Missing audit is not permission to reinterpret a dynamic mixture as an
    # independent DEM: this decision is derived from the profile itself.
    with pytest.raises(ValueError, match="cannot carry DEM mechanism"):
        replace(dynamic, generation_audit=None, contract=Contract.DEM_MECHANISM)


def test_new_environment_without_config_is_refused(legacy: InMemoryDataset) -> None:
    env = EnvironmentSpec(0, None, EnvironmentModel.DEVICE_PROFILE, None, "", "", 16)
    with pytest.raises(ValueError, match="require generation_config"):
        replace(legacy.meta, environments=(env,))


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("circuit", "distance", 5),
        ("circuit", "rounds", 12),
        ("circuit", "basis", "x"),
        ("circuit", "rotated", False),
        ("sampling", "shots", 15),
        ("sampling", "seed", 43),
        ("sampling", "chunk_size", 9),
        ("sampling", "emit_mechanisms", True),
        ("sampling", "shots", 16.0),
    ],
)
def test_reproduction_config_must_agree_with_manifest(
    legacy: InMemoryDataset, section: str, key: str, value: Any
) -> None:
    stored = configured(legacy, EnvironmentModel.DEVICE_PROFILE).meta.to_json_dict()
    stored["generation_config"][section][key] = value
    with pytest.raises(ValueError, match="disagrees with manifest"):
        DatasetMeta.from_json_dict(stored)


@pytest.mark.parametrize("section", ["circuit", "sampling", "output"])
def test_configured_manifests_require_complete_recipe(
    legacy: InMemoryDataset, section: str
) -> None:
    stored = configured(legacy, EnvironmentModel.DEVICE_PROFILE).meta.to_json_dict()
    stored["generation_config"].pop(section)
    with pytest.raises(ValueError, match="must be an object"):
        DatasetMeta.from_json_dict(stored)


def test_config_hash_detects_mutation_even_when_scalar_fields_agree(
    legacy: InMemoryDataset,
) -> None:
    device = configured(legacy, EnvironmentModel.DEVICE_PROFILE)
    config = device.meta.generation_config
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    meta = replace(device.meta, generation_audit={"config_sha256": digest})
    stored = meta.to_json_dict()
    stored["generation_config"]["noise"]["probabilities"]["measurement"] = 0.02
    with pytest.raises(ValueError, match="config_sha256 disagrees"):
        DatasetMeta.from_json_dict(stored)


@pytest.mark.parametrize("requested", ["dem", "full"])
def test_requested_structure_may_legitimately_differ_after_exporter_downgrade(
    legacy: InMemoryDataset, requested: str
) -> None:
    meta = configured(legacy, EnvironmentModel.DEVICE_PROFILE).meta
    config = copy.deepcopy(meta.generation_config)
    assert config is not None
    config["output"]["structure"] = requested
    restored = DatasetMeta.from_json(replace(meta, generation_config=config).to_json())
    assert restored.structure_level is StructureLevel.NONE
    assert restored.generation_config is not None
    assert restored.generation_config["output"]["structure"] == requested


def test_configured_environment_axis_cannot_change_legacy_reconstruction(
    legacy: InMemoryDataset,
) -> None:
    config = common_config(legacy, "legacy")
    config["legacy"] = {"noise_model": "stim_uniform_circuit_level", "p": 0.05}
    stored = replace(legacy.meta, generation_config=config).to_json_dict()
    stored["environments"][0]["axis"] = "xz_bias"
    with pytest.raises(ValueError, match="drift axis"):
        DatasetMeta.from_json_dict(stored)


def test_configured_legacy_channels_cannot_contradict_the_recipe(legacy: InMemoryDataset) -> None:
    config = common_config(legacy, "legacy")
    config["legacy"] = {"noise_model": "stim_uniform_circuit_level", "p": 0.05}
    stored = replace(legacy.meta, generation_config=config).to_json_dict()
    stored["environments"][0]["channels"]["before_measure_flip_probability"] = 0.25
    with pytest.raises(ValueError, match="channel vector"):
        DatasetMeta.from_json_dict(stored)


def test_unknown_config_fields_are_not_silently_ignored(legacy: InMemoryDataset) -> None:
    stored = configured(legacy, EnvironmentModel.DEVICE_PROFILE).meta.to_json_dict()
    stored["generation_config"]["circuit"]["round_rate"] = 1_000_000
    with pytest.raises(ValueError, match="unknown fields"):
        DatasetMeta.from_json_dict(stored)


def test_hardware_identity_must_agree_with_manifest(legacy: InMemoryDataset) -> None:
    stored = configured(legacy, EnvironmentModel.HARDWARE).meta.to_json_dict()
    stored["generation_config"]["hardware"]["expected"]["basis"] = "X"
    with pytest.raises(ValueError, match=r"hardware\.expected\.basis disagrees"):
        DatasetMeta.from_json_dict(stored)


def test_fitted_profile_cannot_claim_test_fit_as_calibration(legacy: InMemoryDataset) -> None:
    stored = configured(legacy, EnvironmentModel.DEVICE_PROFILE).meta.to_json_dict()
    stored["generation_config"]["parameter_provenance"] = {
        "kind": "fitted",
        "description": "Wrong partition",
        "source": "hardware",
        "fit_partition": "test",
    }
    with pytest.raises(ValueError, match="fit_partition='train'"):
        DatasetMeta.from_json_dict(stored)
