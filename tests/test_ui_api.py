"""The HTTP surface, in-process through TestClient. No uvicorn, no browser.

Two themes run through these. Choice sets must be *derived* from the registries rather
than restated, so a new exporter or drift axis reaches the UI without a second edit. And
paths arrive from a browser, so every one of them has to be confined to the data root
before it is used.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from qecgen.dataset import DriftCondition, StructureLevel
from qecgen.environments import DriftAxis
from qecgen.exporters import EXPORTERS, StreamingHDF5Writer
from qecgen.ui.app import create_app
from qecgen.ui.settings import WebSettings

TERMINAL = {"succeeded", "failed", "cancelled"}

GENERATE: dict[str, Any] = {
    "mode": "generate",
    "distance": 3,
    "p": 0.01,
    "shots": 200,
    "seed": 1,
    "out": "dataset.h5",
    "chunk_size": 100,
}


@pytest.fixture
def client(tmp_path: Path) -> Any:
    settings = WebSettings.create(tmp_path / "data")
    # base_url matters: TestClient's default host is "testserver", which is
    # deliberately absent from the production TrustedHost allowlist. Pointing the
    # client at 127.0.0.1 means the suite exercises the real allowlist.
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as test_client:
        yield test_client


def settle(client: TestClient, job_id: str, timeout: float = 60.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    record: dict[str, Any] = {"status": "never polled"}
    while time.monotonic() < deadline:
        record = client.get(f"/api/runs/{job_id}").json()
        if record["status"] in TERMINAL:
            return record
        time.sleep(0.05)
    pytest.fail(f"run never settled; last status {record['status']}")


class TestCapabilities:
    def test_choice_sets_come_from_the_registries(self, client: TestClient) -> None:
        caps = client.get("/api/capabilities").json()
        assert caps["drift_axes"] == [str(axis) for axis in DriftAxis]
        assert caps["structure_levels"] == [str(level) for level in StructureLevel]
        assert {entry["name"] for entry in caps["formats"]} == set(EXPORTERS)

    def test_format_flags_are_read_not_hardcoded(self, client: TestClient) -> None:
        caps = client.get("/api/capabilities").json()
        by_name = {entry["name"]: entry for entry in caps["formats"]}
        for name, exporter in EXPORTERS.items():
            assert by_name[name]["streaming"] is exporter.streaming
            assert by_name[name]["structure_round_trip"] is exporter.structure_round_trip
            assert by_name[name]["extension"] == exporter.extension

    def test_not_applicable_is_not_offered_as_a_drift_condition(self, client: TestClient) -> None:
        # build_drift_environments always refuses it; a dropdown should not repeat a
        # choice that can only ever fail.
        caps = client.get("/api/capabilities").json()
        assert str(DriftCondition.NOT_APPLICABLE) not in caps["drift_conditions"]
        assert str(DriftCondition.FROZEN_PRIOR) in caps["drift_conditions"]


class TestPreview:
    def test_reports_the_real_geometry_without_sampling(self, client: TestClient) -> None:
        preview = client.post("/api/preview", json=GENERATE).json()
        # d=3 rotated, 3 rounds: 24 detectors, 1 observable, 286 DEM mechanisms.
        assert preview["n_detectors"] == 24
        assert preview["n_observables"] == 1
        assert preview["n_mechanisms"] == 286
        assert preview["rounds"] == 3
        assert preview["row_bytes"] == 4  # ceil(24/8) + ceil(1/8)

    def test_says_whether_the_run_will_stream(self, client: TestClient) -> None:
        streamed = client.post(
            "/api/preview", json={**GENERATE, "shots": 5000, "chunk_size": 1000}
        ).json()
        assert streamed["will_stream"] is True
        assert streamed["chunks"] == 5

        materialised = client.post("/api/preview", json={**GENERATE, "fmt": "npz"}).json()
        assert materialised["will_stream"] is False
        assert materialised["materialises"] is True

    def test_counts_every_file_in_a_drift_study(self, client: TestClient) -> None:
        preview = client.post(
            "/api/preview",
            json={
                "mode": "drift",
                "distance": 3,
                "train_p": 0.005,
                "test_values": [0.007, 0.01],
                "shots": 100,
                "seed": 0,
                "out": "drift",
            },
        ).json()
        assert preview["n_files"] == 3
        assert preview["total_shots"] == 300


class TestSubmitAndRun:
    def test_a_run_produces_a_valid_file(self, client: TestClient, tmp_path: Path) -> None:
        response = client.post("/api/runs", json=GENERATE)
        assert response.status_code == 202
        record = settle(client, response.json()["id"])
        assert record["status"] == "succeeded", record["error"]
        assert record["completed_units"] == 200
        assert record["progress_unit"] == "shots"

        written = Path(record["files"][0]["path"])
        assert written.is_file()
        report = client.post("/api/datasets/validate", json={"path": "dataset.h5"}).json()
        assert report["ok"], [c for c in report["checks"] if not c["passed"]]

    def test_the_resolved_config_is_stored_with_the_run(self, client: TestClient) -> None:
        # The CLI's promise is that a log is a complete record of the run, defaults
        # included. The web equivalent is this, and it has to survive on disk.
        record = settle(client, client.post("/api/runs", json=GENERATE).json()["id"])
        assert record["spec"]["rotated"] is True
        assert record["spec"]["noise_model"] == "stim_uniform_circuit_level"
        assert record["spec"]["structure_level"] == "none"

    def test_unknown_run_is_404(self, client: TestClient) -> None:
        assert client.get("/api/runs/nope").status_code == 404
        assert client.post("/api/runs/nope/cancel").status_code == 404


class TestBadInput:
    def test_a_field_violation_is_attributable(self, client: TestClient) -> None:
        response = client.post("/api/runs", json={**GENERATE, "p": 1.5})
        assert response.status_code == 422
        assert any("p" in entry["loc"] for entry in response.json()["detail"])

    def test_an_unregistered_format_is_refused_at_submit(self, client: TestClient) -> None:
        response = client.post("/api/runs", json={**GENERATE, "fmt": "pickle"})
        assert response.status_code == 422

    def test_a_non_p_axis_without_base_p_is_refused(self, client: TestClient) -> None:
        response = client.post(
            "/api/runs",
            json={
                "mode": "multi-env",
                "distance": 3,
                "axis": "xz_bias",
                "axis_values": [0.3, 0.7],
                "shots_per_env": 50,
                "seed": 0,
                "out": "m.h5",
            },
        )
        assert response.status_code == 422

    def test_a_domain_rule_surfaces_as_a_failed_run_not_a_500(self, client: TestClient) -> None:
        # code_capacity with rounds != 1 cannot be checked without building a circuit, so
        # it is caught by the domain and reported through the run record verbatim.
        response = client.post(
            "/api/runs", json={**GENERATE, "noise_model": "code_capacity", "rounds": 3}
        )
        assert response.status_code == 202
        record = settle(client, response.json()["id"])
        assert record["status"] == "failed"
        assert record["error_kind"] == "input"
        assert "CODE_CAPACITY" in record["error"]


# Escape paths are platform-specific: "C:/Windows/..." and "..\\.." are only absolute
# or traversal on Windows — on POSIX they are ordinary relative names that resolve
# INSIDE the data root, so asserting a 400 on them would fail on a correct server.
_WRITE_ESCAPES = ["../../evil.h5"] + (
    ["C:/Windows/System32/evil.h5", "..\\..\\evil.h5"]
    if sys.platform == "win32"
    else ["/tmp/evil.h5"]
)
_READ_ESCAPES = ["../../../etc/passwd"] + (
    ["C:/Windows/win.ini"] if sys.platform == "win32" else ["/etc/passwd"]
)


class TestPathConfinement:
    @pytest.mark.parametrize("escape", _WRITE_ESCAPES)
    def test_writes_outside_the_data_root_are_refused(
        self, client: TestClient, escape: str
    ) -> None:
        response = client.post("/api/runs", json={**GENERATE, "out": escape})
        assert response.status_code == 400
        assert "outside the data root" in response.json()["detail"]

    @pytest.mark.parametrize("escape", _READ_ESCAPES)
    def test_reads_outside_the_data_root_are_refused(self, client: TestClient, escape: str) -> None:
        assert client.get("/api/datasets/manifest", params={"path": escape}).status_code == 400
        assert client.get("/api/datasets/download", params={"path": escape}).status_code == 400

    @pytest.mark.parametrize("alias", [".", "x/..", "./."])
    def test_the_data_root_itself_is_refused_as_a_target(
        self, client: TestClient, alias: str
    ) -> None:
        """Review regression: `out="."` passed validation, and the run writers then
        staged into ``out.parent`` — one level OUTSIDE the confined root, where a killed
        worker's scratch directory could never be swept."""
        response = client.post("/api/runs", json={**GENERATE, "out": alias})
        assert response.status_code == 400
        assert "data root itself" in response.json()["detail"]


class TestReviewRegressionsUI:
    def test_testserver_is_not_in_the_production_host_allowlist(self, tmp_path: Path) -> None:
        """DNS-rebinding surface: TestClient's default 'testserver' host used to ship in
        the production allowlist of an unauthenticated API that writes files and spawns
        processes."""
        settings = WebSettings.create(tmp_path / "data")
        with TestClient(create_app(settings)) as default_host:
            assert default_host.get("/api/capabilities").status_code == 400

    def test_drift_preview_probes_the_axis_unbiased_point(self, tmp_path: Path) -> None:
        """The preview hardcoded probe value 0.5 — xz_bias's unbiased point but HALF of
        measurement_ratio's — so it reported before_measure_flip_probability at half the
        value the training file actually uses."""
        from qecgen.run import DriftSpec
        from qecgen.ui.app import _preview

        spec = DriftSpec(
            distance=3,
            train_p=0.01,
            test_values=(1.5,),
            shots=10,
            seed=0,
            condition=DriftCondition.FROZEN_PRIOR,
            out=tmp_path / "d",
            axis=DriftAxis.MEASUREMENT_RATIO,
        )
        preview = _preview(spec)
        assert preview["channels"]["before_measure_flip_probability"] == pytest.approx(0.01)

    def test_a_corrupt_history_record_does_not_stop_the_server(self, tmp_path: Path) -> None:
        """One unrecognised status string in a persisted record raised ValueError out of
        the FastAPI lifespan, and the server never started."""
        settings = WebSettings.create(tmp_path / "data")
        (settings.runs_dir / "bad.json").write_text(
            json.dumps({"id": "bad", "status": "exploded"}), encoding="utf-8"
        )
        (settings.runs_dir / "good.json").write_text(
            json.dumps({"id": "good", "status": "succeeded"}), encoding="utf-8"
        )
        with TestClient(create_app(settings), base_url="http://127.0.0.1") as survivor:
            ids = {record["id"] for record in survivor.get("/api/runs").json()}
        assert "good" in ids
        assert "bad" not in ids


class TestScore:
    """Scoring a supplied correction, driven entirely from the browser.

    The correction is an input, not a target: nothing here is Contract C, and no file
    gains a label column. See DATA_CONTRACT.md.
    """

    @staticmethod
    def _identity_correction(root: Path, shots: int, width: int, name: str = "zero.npz") -> str:
        import numpy as np

        root.mkdir(parents=True, exist_ok=True)
        np.savez(
            root / name,
            correction_x=np.zeros((shots, width), dtype=np.uint8),
            correction_z=np.zeros((shots, width), dtype=np.uint8),
        )
        return name

    def test_the_correction_schema_is_derived_not_guessed(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        settle(client, client.post("/api/runs", json=GENERATE).json()["id"])
        schema = client.get("/api/datasets/correction-schema", params={"path": "dataset.h5"}).json()
        # d=3 rotated: 9 data qubits, so 2 packed bytes. Neither number is in the
        # manifest -- n_data_qubits comes from the final non-resetting measurement layer.
        assert schema["n_data_qubits"] == 9
        assert schema["packed_width"] == 2
        assert schema["n_observables"] == 1
        assert schema["shots"] == GENERATE["shots"]
        assert schema["bit_order"] == "little"
        assert len(schema["schema_digest"]) > 0

    def test_a_correction_file_is_listed_apart_from_datasets(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        settle(client, client.post("/api/runs", json=GENERATE).json()["id"])
        self._identity_correction(tmp_path / "data", shots=200, width=2)

        corrections = client.get("/api/corrections").json()
        assert [entry["name"] for entry in corrections] == ["zero.npz"]
        assert corrections[0]["shots"] == 200
        assert corrections[0]["width"] == 2
        assert corrections[0]["unpacked"] is False

        # And it is not mistaken for a broken dataset in the browser it shares an
        # extension with. That flag means "a worker died mid-write".
        entry = next(e for e in client.get("/api/datasets").json() if e["name"] == "zero.npz")
        assert entry["unreadable"] is None
        assert "correction" in (entry["not_a_dataset"] or "")

    def test_a_width_mismatch_is_refused_before_the_dataset_is_read(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """The whole point of previewing a score. Left to the run, this surfaces from
        inside the scorer after the file has been materialised."""
        settle(client, client.post("/api/runs", json=GENERATE).json()["id"])
        self._identity_correction(tmp_path / "data", shots=200, width=5, name="wrong.npz")

        preview = client.post(
            "/api/preview",
            json={"mode": "score", "dataset": "dataset.h5", "correction": "wrong.npz"},
        ).json()
        assert preview["compatible"] is False
        assert any("needs" in problem for problem in preview["problems"])

    def test_an_identity_correction_scores_the_raw_error_rate(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """An all-zero correction changes nothing, so its logical error rate must equal
        the uncorrected rate -- the same identity the CLI's own test asserts."""
        settle(client, client.post("/api/runs", json=GENERATE).json()["id"])
        self._identity_correction(tmp_path / "data", shots=200, width=2)

        submitted = client.post(
            "/api/runs",
            json={"mode": "score", "dataset": "dataset.h5", "correction": "zero.npz"},
        )
        assert submitted.status_code == 202
        record = settle(client, submitted.json()["id"])
        assert record["status"] == "succeeded", record["error"]

        result = record["result"]
        assert result["kind"] == "score"
        assert result["shots"] == 200
        assert result["n_data_qubits"] == 9
        assert 0.0 <= result["ci_low"] <= result["logical_error_rate"] <= result["ci_high"] <= 1.0
        assert result["schema_digest"] and result["content_hash"]
        # A read-only job writes no dataset and no artifacts.
        assert record["files"] == []
        assert record["artifacts"] == []
        # Its total is unknown by construction, which the browser draws as indeterminate.
        assert record["progress_unit"] == ""

        # The identity: scoring against the file's own observables reproduces the
        # uncorrected failure count exactly.
        import numpy as np

        from qecgen.exporters import get_exporter
        from qecgen.sampling import unpack_bits

        dataset = get_exporter("hdf5").read(tmp_path / "data" / "dataset.h5")
        raw_failures = int(np.count_nonzero(unpack_bits(dataset.observables, 1).any(axis=1)))
        assert result["failures"] == raw_failures

    def test_a_correction_outside_the_data_root_is_refused(self, client: TestClient) -> None:
        response = client.post(
            "/api/runs",
            json={"mode": "score", "dataset": "dataset.h5", "correction": "../escape.npz"},
        )
        assert response.status_code == 400
        assert "outside the data root" in response.json()["detail"]


class TestProvenance:
    """`inspect --show-text` in the browser, behind its own route."""

    def test_the_manifest_route_still_never_carries_the_text(self, client: TestClient) -> None:
        """The separation is the point. A provenance endpoint that made the manifest
        endpoint looser would have defeated it."""
        settle(
            client,
            client.post("/api/runs", json={**GENERATE, "structure_level": "full"}).json()["id"],
        )
        manifest = client.get("/api/datasets/manifest", params={"path": "dataset.h5"}).json()
        assert "QUBIT_COORDS" not in json.dumps(manifest)

        provenance = client.get("/api/datasets/provenance", params={"path": "dataset.h5"}).json()
        assert provenance["stored"] is True
        assert "QUBIT_COORDS" in provenance["environments"][0]["circuit"]
        assert "error(" in provenance["environments"][0]["dem"]

    def test_it_says_which_condition_the_file_was_written_under(self, client: TestClient) -> None:
        """Warned about, not refused. The response has to carry the one fact that makes
        the warning meaningful, so a caller is never guessing which file this is."""
        settle(
            client,
            client.post("/api/runs", json={**GENERATE, "structure_level": "full"}).json()["id"],
        )
        provenance = client.get("/api/datasets/provenance", params={"path": "dataset.h5"}).json()
        assert provenance["drift_condition"] == "not_applicable"
        assert provenance["structure_level"] == "full"
        assert "hdf5" in provenance["formats_that_store_it"]

    def test_a_file_without_provenance_says_so_rather_than_failing(
        self, client: TestClient
    ) -> None:
        settle(client, client.post("/api/runs", json=GENERATE).json()["id"])
        provenance = client.get("/api/datasets/provenance", params={"path": "dataset.h5"}).json()
        assert provenance["stored"] is False
        assert provenance["environments"] == []


SWEEP: dict[str, Any] = {
    "mode": "sweep",
    "distances": [3, 5],
    "p_low": 0.01,
    "p_high": 0.02,
    "p_count": 2,
    "out": "sweeps/s.csv",
    "max_errors": 20,
    "max_shots": 4000,
    "workers": 2,
}


class TestSweep:
    """Threshold sweeps in the browser."""

    def test_the_preview_reports_the_grid_and_decoder_availability(
        self, client: TestClient
    ) -> None:
        preview = client.post("/api/sweeps/preview", json=SWEEP).json()
        assert preview["n_tasks"] == 4  # 2 distances x 2 rates x 1 decoder
        assert preview["error_rates"] == [0.01, 0.02]
        assert preview["decoders"][0]["name"] == "pymatching"
        assert preview["decoders"][0]["usable"] is True
        assert preview["usable"] is True
        # No shot estimate is offered: max_shots is a ceiling max_errors usually
        # short-circuits, so a number here would be wrong by orders of magnitude.
        assert "shot estimate" in preview["note"]

    def test_a_png_out_path_is_refused_on_the_field_that_caused_it(
        self, client: TestClient
    ) -> None:
        response = client.post("/api/runs", json={**SWEEP, "out": "sweeps/s.png"})
        assert response.status_code == 422
        assert any("out" in entry["loc"] for entry in response.json()["detail"])

    def test_a_missing_decoder_backend_is_named_before_anything_is_collected(
        self, client: TestClient
    ) -> None:
        """sinter discovers this only inside a worker, after building the whole grid.

        Reported rather than refused: the preview's job is to say *which* decoder and
        *why*, on a form the user is still filling in. Submitting one is refused --
        see `test_an_unknown_decoder_is_refused_before_the_run_starts`.
        """
        response = client.post("/api/sweeps/preview", json={**SWEEP, "decoders": ["nonsense"]})
        assert response.status_code == 200
        preview = response.json()
        assert preview["usable"] is False
        assert "nonsense" in (preview["decoders"][0]["problem"] or "")

    @pytest.mark.slow
    def test_a_sweep_writes_three_files_and_reports_its_summary(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        record = settle(client, client.post("/api/runs", json=SWEEP).json()["id"], 600)
        assert record["status"] == "succeeded", record["error"]
        assert record["progress_unit"] == "tasks"
        assert record["total_units"] == 4

        # Three files, and none of them filed as a dataset: they have no shot count, no
        # content hash and no drift condition to put in one.
        assert record["files"] == []
        kinds = {artifact["kind"] for artifact in record["artifacts"]}
        assert kinds == {"results table", "plot", "summary"}

        result = record["result"]
        assert result["kind"] == "sweep"
        assert result["n_points"] == 4
        assert "pymatching" in result["decoders"]
        assert "reported_not_asserted" in result

        # The plot renders inline rather than downloading, which is what an <img> needs.
        plot = next(a for a in record["artifacts"] if a["kind"] == "plot")
        relative = str(Path(plot["path"]).relative_to(tmp_path / "data")).replace("\\", "/")
        response = client.get("/api/sweeps/plot", params={"path": relative})
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert "content-disposition" not in response.headers

        # And the sweep is listed from disk, not only from the run record.
        [listed] = client.get("/api/sweeps").json()
        assert listed["results_path"] == "sweeps/s.csv"
        assert listed["plot_path"] == "sweeps/s.png"
        assert listed["stem"] == "sweeps/s"
        # The listing reads the sidecar rather than the run record, so it also knows which
        # decoders ran and where each one crossed -- including from a terminal sweep this
        # server never supervised.
        assert listed["decoders"] == ["pymatching"]
        assert "pymatching" in listed["crossings"]
        assert listed["unreadable"] is None

    def test_the_plot_route_refuses_anything_that_is_not_a_png(self, client: TestClient) -> None:
        assert client.get("/api/sweeps/plot", params={"path": "s.csv"}).status_code == 400
        assert client.get("/api/sweeps/plot", params={"path": "../escape.png"}).status_code == 400


class TestStatisticalQa:
    """QA in the browser, running the same checks in the same order as `validate --qa`."""

    def test_the_preview_says_the_shot_count_is_a_ceiling(self, client: TestClient) -> None:
        settle(client, client.post("/api/runs", json=GENERATE).json()["id"])
        preview = client.post(
            "/api/preview", json={"mode": "qa", "dataset": "dataset.h5", "max_shots": 400}
        ).json()
        assert preview["n_environments"] == 1
        assert preview["max_total_shots"] == 400
        assert preview["resamples"] is True
        assert "ceiling" in preview["note"]

    def test_qa_reports_a_rate_with_an_interval_per_environment(self, client: TestClient) -> None:
        settle(client, client.post("/api/runs", json=GENERATE).json()["id"])
        record = settle(
            client,
            client.post(
                "/api/runs",
                json={
                    "mode": "qa",
                    "dataset": "dataset.h5",
                    "max_shots": 400,
                    "target_errors": 20,
                },
            ).json()["id"],
        )
        assert record["status"] == "succeeded", record["error"]
        result = record["result"]
        assert result["kind"] == "qa"
        assert result["ok"] is True
        assert result["skipped"] is None
        assert all(check["passed"] for check in result["checks"])
        [environment] = result["environments"]
        # A rate is never reported without its interval, here as anywhere else.
        assert environment["ci_low"] <= environment["logical_error_rate"] <= environment["ci_high"]
        assert environment["shots"] > 0
        assert "never asserted" in result["reported_not_asserted"]

    def test_statistics_are_skipped_when_the_structure_fails(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """The order is the design, and it is `validate --qa`'s order. A rate measured
        against a file whose arrays disagree with its manifest describes nothing, and
        producing one costs minutes before the disagreement is even mentioned."""
        import h5py

        settle(client, client.post("/api/runs", json=GENERATE).json()["id"])
        # Corrupt the recorded hash: cheap, and exactly the kind of disagreement that
        # makes a measured rate meaningless.
        with h5py.File(tmp_path / "data" / "dataset.h5", "r+") as handle:
            manifest = json.loads(str(handle.attrs["manifest"]))
            manifest["content_hash"] = "0" * 64
            handle.attrs["manifest"] = json.dumps(manifest)

        record = settle(
            client,
            client.post(
                "/api/runs", json={"mode": "qa", "dataset": "dataset.h5", "max_shots": 400}
            ).json()["id"],
        )
        assert record["status"] == "succeeded"
        result = record["result"]
        assert result["ok"] is False
        assert result["environments"] == []
        assert "structural check" in result["skipped"]


class TestDatasets:
    def test_listing_is_empty_before_anything_is_written(self, client: TestClient) -> None:
        assert client.get("/api/datasets").json() == []

    def test_a_manifest_never_carries_circuit_or_dem_text(self, client: TestClient) -> None:
        # The provenance firewall. Under FROZEN_PRIOR that text is exactly what the
        # condition withholds from a decoder, and a browser is a decoder-facing surface.
        settle(
            client,
            client.post("/api/runs", json={**GENERATE, "structure_level": "full"}).json()["id"],
        )
        manifest = client.get("/api/datasets/manifest", params={"path": "dataset.h5"}).json()
        assert "circuit" not in manifest
        assert "dem" not in manifest
        for environment in manifest["environments"]:
            assert "circuit" not in environment
            assert "dem" not in environment
        assert "QUBIT_COORDS" not in json.dumps(manifest)

    def test_an_aborted_file_is_listed_as_unreadable_not_hidden(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        # StreamingHDF5Writer.abort() leaves a file with no manifest attribute. Hiding it
        # conceals corruption; showing it as a dataset is worse.
        root = tmp_path / "data"
        writer = StreamingHDF5Writer(root / "broken.h5", 24, 1)
        import numpy as np

        writer.append(np.zeros((4, 3), dtype=np.uint8), np.zeros((4, 1), dtype=np.uint8))
        writer.abort()

        entries = client.get("/api/datasets").json()
        broken = next(entry for entry in entries if entry["name"] == "broken.h5")
        assert broken["unreadable"] is not None
        assert broken["manifest"] is None

    def test_validate_catches_a_truncated_jsonl(self, client: TestClient, tmp_path: Path) -> None:
        # JSONL puts its manifest on line 1, so a truncated file still *reads* and its
        # listing looks healthy. Only validation notices the rows are missing.
        settle(
            client,
            client.post("/api/runs", json={**GENERATE, "fmt": "jsonl", "out": "d.jsonl"}).json()[
                "id"
            ],
        )
        path = tmp_path / "data" / "d.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text("".join(lines[: len(lines) // 2]), encoding="utf-8")

        entry = next(e for e in client.get("/api/datasets").json() if e["name"] == "d.jsonl")
        assert entry["unreadable"] is None  # it reads cleanly; that is the trap
        assert entry["manifest"]["shots"] == 200  # and still claims every shot

        report = client.post("/api/datasets/validate", json={"path": "d.jsonl"}).json()
        assert report["ok"] is False
        assert "manifest.shots" in {c["name"] for c in report["checks"] if not c["passed"]}

    def test_validate_catches_a_truncated_csv(self, client: TestClient, tmp_path: Path) -> None:
        # CSV puts its manifest in the header comments, so it shares JSONL's trap: a file
        # cut short still reads and its listing looks healthy. Truncated on a *line*
        # boundary deliberately -- a mid-line cut leaves a short final row, which read()
        # refuses for column width and would test something else entirely.
        settle(
            client,
            client.post("/api/runs", json={**GENERATE, "fmt": "csv", "out": "d.csv"}).json()["id"],
        )
        path = tmp_path / "data" / "d.csv"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text("".join(lines[: len(lines) // 2]), encoding="utf-8")

        entry = next(e for e in client.get("/api/datasets").json() if e["name"] == "d.csv")
        assert entry["unreadable"] is None  # it reads cleanly; that is the trap
        assert entry["not_a_dataset"] is None
        assert entry["manifest"]["shots"] == 200  # and still claims every shot

        report = client.post("/api/datasets/validate", json={"path": "d.csv"}).json()
        assert report["ok"] is False
        assert "manifest.shots" in {c["name"] for c in report["checks"] if not c["passed"]}

    def test_a_sweep_results_csv_is_not_reported_as_a_broken_dataset(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        # `.csv` is both a dataset extension and the extension `qecgen sweep` writes its
        # threshold table with. Flagging an intact sweep table red teaches the reader to
        # skip the flag that means a worker died mid-write.
        from qecgen.sweep import write_csv

        write_csv([], tmp_path / "data" / "sweep.csv")

        entry = next(e for e in client.get("/api/datasets").json() if e["name"] == "sweep.csv")
        assert entry["unreadable"] is None
        assert entry["manifest"] is None
        assert entry["not_a_dataset"] is not None
        assert "qecgen sweep" in entry["not_a_dataset"]


class TestManifestReaders:
    @pytest.mark.parametrize("name", sorted(EXPORTERS))
    def test_every_registered_format_lists_with_a_manifest(
        self, name: str, client: TestClient, tmp_path: Path
    ) -> None:
        """`read_manifest`'s "no manifest reader" branch carried a
        `# pragma: no cover - unreachable while the registry is covered` comment that
        nothing enforced. Registering an exporter without adding a reader made every file
        of that format list as `unreadable` -- corruption's own signal, on a healthy
        file, with no test to catch it.

        Behavioural rather than a set comparison: `set(_MANIFEST_READERS) ==
        set(EXPORTERS)` would pass with a reader wired to the wrong format.
        """
        from qecgen.environments import build_single_environment
        from qecgen.exporters import get_exporter

        exporter = get_exporter(name)
        dataset = build_single_environment(distance=3, p=0.01, shots=40, seed=1, chunk_size=40)
        exporter.write(dataset, tmp_path / "data" / f"m{exporter.extension}")

        entries = client.get("/api/datasets").json()
        entry = next(e for e in entries if e["format"] == name)
        assert entry["unreadable"] is None, entry["unreadable"]
        assert entry["not_a_dataset"] is None
        assert entry["manifest"]["shots"] == 40


SWEEP_GRID: dict[str, Any] = {
    "mode": "sweep",
    "distances": [3, 5],
    "p_low": 0.005,
    "p_high": 0.02,
    "p_count": 4,
    "out": "sweeps/s.csv",
}

# A minimal valid PNG. The plot route only has to prove it streams bytes inline; running
# matplotlib to produce a real figure would make a structural test a slow one.
_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)


def _write_sweep(root: Path, stem: str = "sweeps/s", *, plot: bool = True) -> Path:
    """Lay down a sweep triple by hand, in the exact shape `run_sweep_job` commits."""
    target = root / stem
    target.parent.mkdir(parents=True, exist_ok=True)
    target.with_suffix(".csv").write_text(
        "decoder,distance,p,rounds,noise_model,basis,shots,errors,discards,"
        "logical_error_rate,ci_low,ci_high\n"
        "pymatching,3,0.005,3,stim_uniform_circuit_level,z,1000,40,0,0.04,0.0287,0.0539\n"
        "pymatching,3,0.02,3,stim_uniform_circuit_level,z,1000,300,0,0.3,0.2716,0.3297\n"
        "pymatching,5,0.005,5,stim_uniform_circuit_level,z,1000,0,0,0,0,0.0037\n",
        encoding="utf-8",
    )
    target.with_suffix(".threshold.json").write_text(
        json.dumps(
            {
                "qecgen_version": "0.1.0",
                "alpha": 0.05,
                "reported_not_asserted": "Threshold crossing and Lambda are results.",
                "noise_model": "stim_uniform_circuit_level",
                "basis": "z",
                "max_errors": 500,
                "max_shots": 100000,
                "censored_points": [],
                "decoders": {"pymatching": {"crossing_p": 0.011, "suppression": []}},
            }
        ),
        encoding="utf-8",
    )
    if plot:
        target.with_suffix(".png").write_bytes(_PNG)
    return target.with_suffix(".threshold.json")


class TestSweeps:
    def test_capabilities_report_decoder_availability_from_the_registry(
        self, client: TestClient
    ) -> None:
        from qecgen.decoders import check_decoder, known_decoder_names

        caps = client.get("/api/capabilities").json()
        by_name = {entry["name"]: entry for entry in caps["decoders"]}
        assert set(by_name) == set(known_decoder_names())
        for name, entry in by_name.items():
            availability = check_decoder(name)
            assert entry["usable"] is availability.usable
            assert entry["problem"] == availability.problem()
        # Unusable decoders are reported, never filtered out: a user who came looking for
        # one needs to be told which pip install is missing.
        assert all(entry["problem"] is None for entry in by_name.values() if entry["usable"])

    def test_the_grid_is_previewed_without_collecting(self, client: TestClient) -> None:
        preview = client.post("/api/sweeps/preview", json=SWEEP_GRID).json()
        assert preview["error_rates"] == [0.005, 0.01, 0.015, 0.02]
        assert preview["n_tasks"] == 8  # 2 distances x 4 rates x 1 decoder
        assert preview["overwrites"] is False
        assert preview["plot_path"].endswith(".png")
        assert preview["summary_path"].endswith(".threshold.json")

    def test_the_dataset_preview_refuses_a_sweep_and_says_where_to_go(
        self, client: TestClient
    ) -> None:
        # The two answer different questions and share no fields; silently returning a
        # dataset-shaped estimate for a sweep would be worse than a 400.
        response = client.post("/api/preview", json=SWEEP_GRID)
        assert response.status_code == 400
        assert "/api/sweeps/preview" in response.json()["detail"]

    def test_an_unknown_decoder_is_refused_before_the_run_starts(self, client: TestClient) -> None:
        # sinter only discovers this inside a worker, after the whole task grid is built.
        response = client.post("/api/runs", json={**SWEEP_GRID, "decoders": ["nope"]})
        assert response.status_code == 422
        assert "nope" in json.dumps(response.json())

    def test_a_sweep_is_queued_with_a_task_denominator(self, tmp_path: Path) -> None:
        """A scripted child, not the real worker.

        `submit()` pumps synchronously, so posting here with the default worker command
        spawns `python -m qecgen.ui.worker` immediately and the only thing stopping it
        collecting for real is that fixture teardown kills it mid-import. Nothing in this
        test needs a child at all, and a fast-suite test whose safety rests on losing a
        race with an import is a fast-suite test that will one day sample shots.
        """
        from qecgen.ui.jobs import JobStore

        settings = WebSettings.create(tmp_path / "data")
        store = JobStore(
            settings.runs_dir,
            worker_command=(sys.executable, "-c", 'print(\'{"event": "done", "files": []}\')'),
        )
        with TestClient(create_app(settings, store=store), base_url="http://127.0.0.1") as scripted:
            record = scripted.post("/api/runs", json={**SWEEP_GRID, "distances": [3]}).json()
        assert record["mode"] == "sweep"
        assert record["progress_unit"] == "tasks"
        # 1 distance x 4 rates x 1 decoder, known before anything runs.
        assert record["total_units"] == 4

    def test_listing_finds_a_sweep_the_cli_wrote(self, client: TestClient, tmp_path: Path) -> None:
        _write_sweep(tmp_path / "data")
        entries = client.get("/api/sweeps").json()
        assert len(entries) == 1
        assert entries[0]["stem"] == "sweeps/s"
        assert entries[0]["decoders"] == ["pymatching"]
        assert entries[0]["crossings"] == {"pymatching": 0.011}
        assert entries[0]["plot_path"] == "sweeps/s.png"
        assert entries[0]["unreadable"] is None

    def test_a_sweep_without_its_plot_is_listed_with_the_plot_missing(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        _write_sweep(tmp_path / "data", plot=False)
        entry = client.get("/api/sweeps").json()[0]
        assert entry["plot_path"] is None
        assert entry["results_path"] == "sweeps/s.csv"

    def test_detail_returns_points_read_from_the_results_table(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        summary = _write_sweep(tmp_path / "data")
        detail = client.get(f"/api/sweeps/detail?path={summary.relative_to(tmp_path / 'data')}")
        payload = detail.json()
        assert detail.status_code == 200
        series = {(s["decoder"], s["distance"]): s["points"] for s in payload["series"]}
        assert set(series) == {("pymatching", 3), ("pymatching", 5)}
        # Every drawn number comes from a column the CSV already carried; nothing is
        # recomputed here or in the browser.
        first = series[("pymatching", 3)][0]
        assert (first["p"], first["rate"], first["ci_low"], first["ci_high"]) == (
            0.005,
            0.04,
            0.0287,
            0.0539,
        )
        # The zero-error point survives with its one-sided bound: it is drawn as a caret,
        # not dropped, because it is usually the most suppressed point of a sweep.
        assert series[("pymatching", 5)][0]["errors"] == 0
        assert series[("pymatching", 5)][0]["ci_high"] == 0.0037
        assert payload["summary"]["decoders"]["pymatching"]["crossing_p"] == 0.011

    def test_detail_without_a_results_table_is_a_404_naming_the_reason(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        summary = _write_sweep(tmp_path / "data")
        summary.with_suffix("").with_suffix(".csv").unlink()
        response = client.get("/api/sweeps/detail?path=sweeps/s.threshold.json")
        assert response.status_code == 404
        assert "nothing to plot" in response.json()["detail"]

    def test_the_plot_is_served_for_display_not_download(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """The whole point of a separate route.

        `/api/datasets/download` passes `filename=`, which makes Starlette send
        `Content-Disposition: attachment` and the browser save the file instead of
        rendering it.
        """
        _write_sweep(tmp_path / "data")
        response = client.get("/api/sweeps/plot?path=sweeps/s.png")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert "content-disposition" not in response.headers
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.content == _PNG

    def test_the_plot_route_serves_only_png(self, client: TestClient, tmp_path: Path) -> None:
        # This is the one route that asks a browser to render a file out of the data root,
        # so it must not be able to hand back something that executes.
        _write_sweep(tmp_path / "data")
        response = client.get("/api/sweeps/plot?path=sweeps/s.csv")
        assert response.status_code == 400
        assert "only serves sweep plots" in response.json()["detail"]

    def test_the_plot_route_is_confined_to_the_data_root(self, client: TestClient) -> None:
        assert client.get("/api/sweeps/plot?path=../escape.png").status_code == 400

    def test_a_results_table_is_still_listed_as_not_a_dataset(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        # The sweep CSV carries a dataset extension. It must stay visible in the dataset
        # browser as "not ours" rather than wearing the corruption flag.
        _write_sweep(tmp_path / "data")
        entry = next(e for e in client.get("/api/datasets").json() if e["name"] == "s.csv")
        assert entry["unreadable"] is None
        assert entry["not_a_dataset"] is not None


class TestFrontendNotBuilt:
    def test_the_api_stays_up_and_the_page_names_the_fix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A blank page would leave the user guessing. Naming the command is the fix, and
        # the API has to keep working so a run started from an open tab is not killed by
        # a frontend that has not been compiled.
        from qecgen.ui import app as app_module

        monkeypatch.setattr(app_module, "static_is_built", lambda: False)
        settings = WebSettings.create(tmp_path / "data")
        with TestClient(create_app(settings), base_url="http://127.0.0.1") as bare:
            page = bare.get("/")
            assert page.status_code == 503
            assert "npm run build" in page.text
            assert bare.get("/api/capabilities").status_code == 200
