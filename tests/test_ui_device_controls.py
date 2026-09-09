"""The form's helpers preserve server authority and exact sweep seed identity."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from qecgen.configuration import expand_sweep
from qecgen.ui.app import create_app
from qecgen.ui.jobs import JobStore
from qecgen.ui.schemas import ConfiguredRequest
from qecgen.ui.settings import WebSettings


def config() -> dict[str, Any]:
    return {
        "version": 1,
        "mode": "device",
        "output": {"path": "controls.h5"},
        "sampling": {"shots": 16, "seed": 123, "chunk_size": 8},
        "circuit": {"distance": 3, "rounds": 3},
        "noise": {"version": 1, "probabilities": {"measurement": 0.01}},
        "parameter_provenance": {"kind": "scenario", "description": "Form test"},
    }


def test_layout_can_help_finish_an_incomplete_coherence_profile(tmp_path: Path) -> None:
    raw = config()
    raw["noise"]["coherence"] = {"enabled": True}
    with TestClient(
        create_app(WebSettings.create(tmp_path)), base_url="http://127.0.0.1"
    ) as client:
        assert (
            client.post("/api/preview", json={"mode": "configured", "config": raw}).status_code
            == 400
        )
        response = client.post("/api/configured/layout", json={"config": raw})
    assert response.status_code == 200, response.text
    actual = response.json()
    assert len(actual["qubits"]) == 17  # Includes checks; this is not a correction schema.
    assert actual["n_detectors"] == 24
    assert actual["n_observables"] == 1
    assert actual["layer_count"] == 22
    assert all(a < b for a, b in actual["edges"])
    assert raw["noise"]["coherence"] == {"enabled": True}


def test_layout_refuses_external_paths_and_wrong_source_hashes(tmp_path: Path) -> None:
    raw = config()
    raw["circuit"].update(stim_file="../private.stim", sha256="0" * 64)
    with TestClient(
        create_app(WebSettings.create(tmp_path)), base_url="http://127.0.0.1"
    ) as client:
        assert client.post("/api/configured/layout", json={"config": raw}).status_code == 400
        raw["circuit"]["stim_file"] = "input.stim"
        (tmp_path / "input.stim").write_text("R 0\nM 0\n")
        response = client.post("/api/configured/layout", json={"config": raw})
        assert response.status_code == 400
        assert "SHA-256" in response.text


def test_sweep_preview_and_queued_records_preserve_full_width_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = WebSettings.create(tmp_path)
    store = JobStore(settings.runs_dir)
    monkeypatch.setattr(store, "_pump", lambda: None)
    request: dict[str, Any] = {
        "config": config(),
        "field": "noise.probabilities.measurement",
        "values": [0.01, 0.03],
    }
    base = ConfiguredRequest(config=request["config"]).to_spec(tmp_path)
    expected = expand_sweep(base.config, request["field"], request["values"])
    with TestClient(create_app(settings, store), base_url="http://127.0.0.1") as client:
        response = client.post("/api/configured/sweep-preview", json=request)
        assert response.status_code == 200, response.text
        preview = response.json()
        assert preview["total_shots"] == 32
        assert store.records() == []
        assert [json.loads(row["config_json"]) for row in preview["runs"]] == expected
        assert any(int(row["seed"]) > 2**53 for row in preview["runs"])
        queued = client.post("/api/configured/sweep", json=request)
        assert queued.status_code == 202, queued.text
        for row, expected_config in zip(queued.json()["runs"], expected, strict=True):
            record = store.get(row["id"])
            assert record is not None
            assert record.spec["config"] == expected_config
            assert json.loads(row["spec_json"])["config"] == expected_config
            fetched = client.get(f"/api/runs/{row['id']}").json()
            assert fetched["spec_json"] == row["spec_json"]
        # Repeated clicks cannot queue simultaneous writers to the same outputs.
        assert client.post("/api/configured/sweep", json=request).status_code == 409


@pytest.mark.parametrize("values", [[], [0.01, 1.1], ["0.01"], [False], [0.01] * 101])
def test_invalid_sweep_never_queues_a_partial_grid(tmp_path: Path, values: list[Any]) -> None:
    settings = WebSettings.create(tmp_path)
    store = JobStore(settings.runs_dir)
    request = {"config": config(), "field": "noise.probabilities.measurement", "values": values}
    with TestClient(create_app(settings, store), base_url="http://127.0.0.1") as client:
        assert client.post("/api/configured/sweep", json=request).status_code in {400, 422}
        assert store.records() == []


def test_browser_refuses_unsafe_base_seed_but_preserves_large_decimal_rates(tmp_path: Path) -> None:
    raw = config()
    raw["sampling"]["seed"] = 2**53 + 1
    with TestClient(
        create_app(WebSettings.create(tmp_path)), base_url="http://127.0.0.1"
    ) as client:
        assert (
            client.post("/api/preview", json={"mode": "configured", "config": raw}).status_code
            == 422
        )
        raw["sampling"]["seed"] = 2**53 - 1
        assert (
            client.post("/api/preview", json={"mode": "configured", "config": raw}).status_code
            == 200
        )


def test_preview_checks_used_qubits_without_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import qecgen.configuration as configuration

    def forbidden_sampler(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        yield from ()
        raise AssertionError("Preview advanced the shot iterator")

    monkeypatch.setattr(configuration, "iter_profile_chunks", forbidden_sampler)
    raw = config()
    with TestClient(
        create_app(WebSettings.create(tmp_path)), base_url="http://127.0.0.1"
    ) as client:
        response = client.post("/api/preview", json={"mode": "configured", "config": raw})
        assert response.status_code == 200, response.text
        assert response.json()["n_detectors"] == 24
        raw["noise"]["qubit_overrides"] = {"999": {"measurement": 0.01}}
        response = client.post("/api/preview", json={"mode": "configured", "config": raw})
        assert response.status_code == 400
        assert "unused qubits" in response.text


def test_preview_reports_missing_hardware_as_input_error(tmp_path: Path) -> None:
    raw = json.loads((Path(__file__).parents[1] / "examples/willow-import.json").read_text())
    raw["output"]["path"] = "hardware.h5"
    raw["hardware"]["table"] = "missing.parquet"
    raw["hardware"]["circuit"] = "missing.stim"
    with TestClient(
        create_app(WebSettings.create(tmp_path)), base_url="http://127.0.0.1"
    ) as client:
        response = client.post("/api/preview", json={"mode": "configured", "config": raw})
        assert response.status_code == 400


@pytest.fixture
def parked_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, JobStore]]:
    settings = WebSettings.create(tmp_path)
    store = JobStore(settings.runs_dir)
    monkeypatch.setattr(store, "_pump", lambda: None)
    with TestClient(create_app(settings, store), base_url="http://127.0.0.1") as client:
        yield client, store


@pytest.mark.parametrize("configured_first", [False, True])
@pytest.mark.parametrize("overlap", ["anchor", "companion", "directory", "threshold"])
def test_configured_and_ordinary_jobs_reserve_overlapping_outputs_symmetrically(
    parked_client: tuple[TestClient, JobStore], tmp_path: Path, configured_first: bool, overlap: str
) -> None:
    client, store = parked_client
    raw = config()
    ordinary: dict[str, Any] = {
        "mode": "generate",
        "distance": 3,
        "shots": 16,
        "p": 0.02,
        "out": "shared.h5",
    }
    raw["output"]["path"] = "shared.h5"
    if overlap == "companion":
        raw["output"].update(path="shared.ml.csv", format="ml_csv")
        ordinary["out"] = "shared.ml.manifest.json"
    elif overlap == "directory":
        raw["output"]["path"] = "study/device.h5"
        ordinary = {
            "mode": "drift",
            "distance": 3,
            "shots": 16,
            "train_p": 0.01,
            "test_values": [0.02],
            "out": "study",
        }
    elif overlap == "threshold":
        raw["output"].update(path="threshold.csv", format="csv")
        ordinary = {
            "mode": "sweep",
            "distances": [3],
            "p_low": 0.01,
            "p_high": 0.02,
            "p_count": 2,
            "out": "threshold.csv",
            "decoders": ["pymatching"],
        }
    configured = {"mode": "configured", "config": raw}
    first, second = (configured, ordinary) if configured_first else (ordinary, configured)
    accepted = client.post("/api/runs", json=first)
    assert accepted.status_code == 202, accepted.text
    refused = client.post("/api/runs", json=second)
    assert refused.status_code == 409, refused.text
    assert accepted.json()["id"] in refused.text
    assert len(store.records()) == 1
    # These reservations must work before an exporter has created any companions.
    assert not (tmp_path / raw["output"]["path"]).exists()
    assert not (tmp_path / ordinary["out"]).exists()


@pytest.mark.parametrize("configured_first", [False, True])
def test_configured_parameter_sweep_and_ordinary_generation_share_reservations(
    parked_client: tuple[TestClient, JobStore], configured_first: bool
) -> None:
    client, store = parked_client
    request = {
        "config": config(),
        "field": "noise.probabilities.measurement",
        "values": [0.01, 0.02],
    }
    ordinary = {
        "mode": "generate",
        "distance": 3,
        "shots": 16,
        "p": 0.02,
        "out": "controls-001.h5",
    }
    first = ("/api/configured/sweep", request) if configured_first else ("/api/runs", ordinary)
    second = ("/api/runs", ordinary) if configured_first else ("/api/configured/sweep", request)
    accepted = client.post(first[0], json=first[1])
    assert accepted.status_code == 202, accepted.text
    refused = client.post(second[0], json=second[1])
    assert refused.status_code == 409, refused.text
    # A collision at the second point must not leave the first point queued.
    assert len(store.records()) == (2 if configured_first else 1)
