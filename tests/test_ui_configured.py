"""Configured jobs preserve UI confinement and the existing worker artifact contract."""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from qecgen.dataset import DatasetMeta
from qecgen.ui.app import create_app
from qecgen.ui.datasets import PathOutsideRootError
from qecgen.ui.jobs import JobRecord, JobStore, run_input_paths, run_output_paths
from qecgen.ui.schemas import ConfiguredRequest
from qecgen.ui.settings import WebSettings


def legacy_config() -> dict[str, Any]:
    return {
        "version": 1,
        "mode": "legacy",
        "output": {"path": "configured.h5", "format": "hdf5", "structure": "none"},
        "sampling": {"shots": 32, "seed": 2, "chunk_size": 16},
        "circuit": {"distance": 3, "rounds": 3, "basis": "z", "rotated": True},
        "legacy": {"noise_model": "stim_uniform_circuit_level", "p": 0.005},
    }


def test_resolution_does_not_mutate_the_request(tmp_path: Path) -> None:
    config = legacy_config()
    before = deepcopy(config)
    spec = ConfiguredRequest(config=config).to_spec(tmp_path)
    assert config == before
    assert spec.out == tmp_path / "configured.h5"


@pytest.mark.parametrize("path", ["../escape.h5", "."])
def test_output_must_remain_below_root(tmp_path: Path, path: str) -> None:
    config = legacy_config()
    config["output"]["path"] = path
    with pytest.raises(PathOutsideRootError):
        ConfiguredRequest(config=config).to_spec(tmp_path)


@pytest.mark.parametrize(
    ("section", "key"),
    [("hardware", "table"), ("hardware", "circuit"), ("circuit", "stim_file")],
)
def test_nested_input_paths_cannot_escape(tmp_path: Path, section: str, key: str) -> None:
    config = legacy_config()
    config["hardware"] = {"table": "shots.parquet", "circuit": "circuit.stim"}
    config[section][key] = "../private-file"
    with pytest.raises(PathOutsideRootError):
        ConfiguredRequest(config=config).to_spec(tmp_path)


def test_import_inputs_are_kept_when_deleting_a_run(tmp_path: Path) -> None:
    table, circuit, output = (tmp_path / name for name in ("raw.parquet", "raw.stim", "out.h5"))
    record = JobRecord(
        id="configured",
        mode="configured",
        total_units=10,
        spec={
            "config": {
                "hardware": {"table": str(table), "circuit": str(circuit)},
                "output": {"path": str(output)},
            }
        },
        files=[{"path": str(output), "kind": "dataset"}],
    )
    assert run_input_paths(record) == [table, circuit]
    assert run_output_paths(record) == [output]


def test_live_configured_output_is_protected_from_ui_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JobStore(tmp_path / "runs")
    # Keep the job queued so this tests the protection without a scheduler race.
    monkeypatch.setattr(store, "_pump", lambda: None)
    spec = ConfiguredRequest(config=legacy_config()).to_spec(tmp_path)
    record = store.submit(spec)
    blocker = store.blocking_run(spec.out)
    assert blocker is not None
    assert blocker.job_id == record.id
    assert store.blocking_run(tmp_path / "unrelated.h5") is None


@pytest.mark.parametrize("mode", ["legacy", "device"])
def test_configured_http_job_uses_real_worker_and_dataset_flow(tmp_path: Path, mode: str) -> None:
    settings = WebSettings.create(tmp_path / "data")
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        config = legacy_config()
        if mode == "device":
            del config["legacy"]
            config["mode"] = "device"
            config["noise"] = {
                "version": 1,
                "probabilities": {"measurement": 0.003},
                "qubit_overrides": {"1": {"measurement": 0.01}},
            }
            config["parameter_provenance"] = {
                "kind": "scenario",
                "description": "UI integration test; arbitrary parameters",
            }
        body = {"mode": "configured", "config": config}
        response = client.post("/api/preview", json=body)
        assert response.status_code == 200, response.text
        preview = response.json()
        assert preview["kind"] == "configured"
        assert preview["total_shots"] == 32
        assert "estimated_bytes" not in preview
        response = client.post("/api/runs", json=body)
        assert response.status_code == 202, response.text
        job = response.json()
        assert job["mode"] == "configured"
        assert job["total_units"] == 32
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            job = client.get(f"/api/runs/{job['id']}").json()
            if job["status"] in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(0.02)
        assert job["status"] == "succeeded", job
        assert job["completed_units"] == 32
        assert job["files"][0]["kind"] == "dataset"
        assert job["files"][0]["shots"] == 32
        assert (settings.data_root / "configured.h5").is_file()
        listed = client.get("/api/datasets").json()
        assert any(row["path"] == "configured.h5" for row in listed)


def test_bad_config_is_a_client_error(tmp_path: Path) -> None:
    with TestClient(
        create_app(WebSettings.create(tmp_path / "data")), base_url="http://127.0.0.1"
    ) as client:
        for config in ({}, {**legacy_config(), "version": 999}):
            response = client.post("/api/preview", json={"mode": "configured", "config": config})
            assert response.status_code == 400, response.text


@pytest.mark.parametrize("mode", ["hardware", "device"])
def test_external_layout_correction_preview_refuses_unaudited_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    # Exercise the schema policy after parsing. No fixture claims that synthetic
    # measurements are hardware: the metadata reader alone is replaced here.
    meta = SimpleNamespace(
        generation_config={
            "mode": mode,
            "circuit": {"stim_file": "external.stim"} if mode == "device" else {},
        }
    )
    monkeypatch.setattr(DatasetMeta, "from_json_dict", lambda payload: meta)
    monkeypatch.setattr("qecgen.exporters.read_manifest", lambda *args: {})
    settings = WebSettings.create(tmp_path / "data")
    (settings.data_root / "external.h5").touch()
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        response = client.get("/api/datasets/correction-schema", params={"path": "external.h5"})
        assert response.status_code == 422, response.text
        assert "external hardware correction roles have not been audited" in response.text
