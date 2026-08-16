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
        assert record["completed_shots"] == 200

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
