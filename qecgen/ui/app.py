"""The FastAPI application.

The only module in the package that imports FastAPI, and nothing imports it at package
level, so a worker never pays for it.

Two things here are security-relevant rather than cosmetic. Every path that arrives from
the browser goes through :func:`~qecgen.ui.datasets.resolve_within` before it is used, so
a run cannot write outside the data root and a download cannot read outside it. And CORS
is emitted only under ``--dev``, for exactly the Vite origin: the default posture is that
the only page allowed to drive this API is the one this server served.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from qecgen import __version__
from qecgen.circuits import Basis, NoiseModel, default_rounds
from qecgen.dataset import StructureLevel
from qecgen.environments import DriftAxis, build_environment, unbiased_point
from qecgen.exporters import EXPORTERS
from qecgen.run import DriftSpec, GenerateSpec, MultiEnvSpec, RunSpec, should_stream, total_shots
from qecgen.sampling import DEFAULT_CHUNK_SIZE, packed_width
from qecgen.ui.datasets import (
    PathOutsideRootError,
    full_manifest,
    list_datasets,
    resolve_within,
    validate_at,
)
from qecgen.ui.jobs import DEFAULT_WORKER_COMMAND, JobStore
from qecgen.ui.schemas import SELECTABLE_DRIFT_CONDITIONS, RunRequest
from qecgen.ui.settings import WebSettings

__all__ = ["STATIC_DIR", "create_app", "static_is_built"]

STATIC_DIR = Path(__file__).parent / "static"
"""Where ``npm run build`` puts the frontend. Gitignored; built on demand."""

BUILD_HINT = "cd frontend && npm ci && npm run build"

SSE_POLL_SECONDS = 0.1
SSE_KEEPALIVE_SECONDS = 15.0

VITE_DEV_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


def static_is_built() -> bool:
    """Whether a built frontend is present."""
    return (STATIC_DIR / "index.html").is_file()


def _enum_options(values: Any) -> list[str]:
    return [str(value) for value in values]


def _capabilities() -> dict[str, Any]:
    """Choice sets, read from the live registries rather than restated.

    A new exporter or drift axis reaches the UI with no frontend change — the same
    automatic pickup the exporter round-trip tests get from iterating ``EXPORTERS``.
    """
    return {
        "version": __version__,
        "noise_models": _enum_options(NoiseModel),
        "bases": _enum_options(Basis),
        "structure_levels": _enum_options(StructureLevel),
        "drift_axes": _enum_options(DriftAxis),
        "drift_conditions": _enum_options(SELECTABLE_DRIFT_CONDITIONS),
        "default_chunk_size": DEFAULT_CHUNK_SIZE,
        "formats": [
            {
                "name": name,
                "extension": exporter.extension,
                "streaming": exporter.streaming,
                "structure_round_trip": exporter.structure_round_trip,
            }
            for name, exporter in sorted(EXPORTERS.items())
        ],
    }


def _preview(spec: RunSpec) -> dict[str, Any]:
    """Cost estimate for a run, without sampling a single shot.

    ``build_environment`` builds the circuit and its decomposed DEM and stops there, so
    this costs milliseconds and answers the questions the terminal cannot answer before
    committing: how many detectors, how big the file, will it stream.
    """
    probe_p: float | None
    match spec:
        case GenerateSpec():
            probe_p, probe_axis, probe_value = spec.p, DriftAxis.P, spec.p
        case MultiEnvSpec():
            first = spec.axis_values[0]
            probe_p = spec.base_p if spec.axis is not DriftAxis.P else first
            probe_axis, probe_value = spec.axis, first
        case DriftSpec():
            probe_p, probe_axis = spec.train_p, spec.axis
            # The training file is built at the axis's unbiased point — the same rule
            # build_drift_environments applies. A hardcoded 0.5 was xz_bias's point but
            # HALF of measurement_ratio's, so the preview reported
            # before_measure_flip_probability at half the training file's real value.
            probe_value = spec.train_p if spec.axis is DriftAxis.P else unbiased_point(spec.axis)

    build = build_environment(
        environment_id=0,
        distance=spec.distance,
        base_p=probe_p if probe_p is not None else 0.0,
        axis=probe_axis,
        axis_value=probe_value,
        shots=1,
        noise_model=spec.noise_model,
        rounds=spec.rounds,
        basis=spec.basis,
        rotated=spec.rotated,
    )
    n_detectors = build.circuit.num_detectors
    n_observables = build.circuit.num_observables
    n_mechanisms = build.dem.num_errors

    shots = total_shots(spec)
    row_bytes = packed_width(n_detectors) + packed_width(n_observables)
    if spec.emit_mechanisms:
        row_bytes += packed_width(n_mechanisms)
    if isinstance(spec, MultiEnvSpec):
        row_bytes += 4  # int32 environment_id per shot

    streams = isinstance(spec, GenerateSpec) and should_stream(
        spec.fmt, spec.shots, spec.chunk_size
    )
    per_file = spec.shots if isinstance(spec, DriftSpec) else shots
    return {
        "total_shots": shots,
        "n_detectors": n_detectors,
        "n_observables": n_observables,
        "n_mechanisms": n_mechanisms,
        "row_bytes": row_bytes,
        "estimated_bytes": row_bytes * shots,
        "n_files": 1 + len(spec.test_values) if isinstance(spec, DriftSpec) else 1,
        "chunks": -(-per_file // spec.chunk_size) if spec.chunk_size else 0,
        "will_stream": streams,
        "materialises": not streams,
        "rounds": default_rounds(spec.noise_model, spec.distance, spec.rounds),
        "channels": build.spec.channels.as_dict(),
    }


def create_app(settings: WebSettings, store: JobStore | None = None) -> FastAPI:
    """Build the application. A factory, so tests get an isolated instance per case."""
    jobs = store or JobStore(
        settings.runs_dir,
        worker_command=DEFAULT_WORKER_COMMAND,
        max_concurrent=settings.max_concurrent_jobs,
    )

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        del app
        jobs.load_history()
        removed = jobs.clean_partials(settings.data_root)
        for path in removed:
            # Reported rather than tidied away silently: a staging directory that
            # outlived its run means a worker was killed.
            print(f"qecgen ui: removed orphaned staging directory {path}")
        yield
        jobs.shutdown()

    api = FastAPI(
        title="qecgen",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    # Guards against DNS rebinding: a page on another origin can reach a loopback port,
    # but only by sending its own hostname in Host. Starlette compares the host without
    # its port, so these are bare names -- a "127.0.0.1:*" pattern is rejected outright.
    # "testserver" (TestClient's default) is deliberately NOT here: shipping a test
    # hostname in the production allowlist widens the accepted set for no user benefit;
    # the suite points its client at http://127.0.0.1 instead.
    api.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost"],
    )
    if settings.dev:
        api.add_middleware(
            CORSMiddleware,
            allow_origins=list(VITE_DEV_ORIGINS),
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @api.middleware("http")
    async def no_store_api_responses(request: Request, call_next: Any) -> Any:
        """Keep the browser from caching API reads.

        Everything here is live state — run progress, what is on disk, which directory
        the server was started with. A cached ``/api/capabilities`` had the page still
        naming the previous ``--data-root`` after a restart, which is exactly the kind of
        quiet wrongness this tool is supposed to avoid.
        """
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    def _resolve(path: str) -> Path:
        try:
            return resolve_within(settings.data_root, path)
        except PathOutsideRootError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @api.get("/api/capabilities")
    def capabilities() -> dict[str, Any]:
        """Choice sets and server configuration the forms are built from."""
        return {
            **_capabilities(),
            "data_root": str(settings.data_root),
            "runs_dir": str(settings.runs_dir),
            "max_concurrent_jobs": settings.max_concurrent_jobs,
            "static_built": static_is_built(),
        }

    @api.post("/api/preview")
    def preview(request: Annotated[RunRequest, Body()]) -> dict[str, Any]:
        """Cost estimate for a run that has not been submitted."""
        try:
            return _preview(request.to_spec(settings.data_root))
        except PathOutsideRootError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @api.post("/api/runs", status_code=202)
    def submit_run(request: Annotated[RunRequest, Body()]) -> dict[str, Any]:
        """Queue a run and return its record. Nothing has been sampled yet."""
        try:
            spec = request.to_spec(settings.data_root)
        except PathOutsideRootError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return jobs.submit(spec).to_json_dict()

    @api.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        """Every run this server knows about, newest first."""
        return [record.to_json_dict() for record in jobs.records()]

    @api.get("/api/runs/{job_id}")
    def get_run(job_id: str) -> dict[str, Any]:
        """One run's record."""
        record = jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"no run {job_id!r}")
        return record.to_json_dict()

    @api.post("/api/runs/{job_id}/cancel")
    def cancel_run(job_id: str) -> dict[str, Any]:
        """Ask a run to stop at its next chunk boundary."""
        if jobs.get(job_id) is None:
            raise HTTPException(status_code=404, detail=f"no run {job_id!r}")
        if not jobs.cancel(job_id):
            raise HTTPException(status_code=409, detail="run has already finished")
        record = jobs.get(job_id)
        assert record is not None
        return record.to_json_dict()

    @api.get("/api/runs/{job_id}/events")
    def run_events(job_id: str, request: Request) -> StreamingResponse:
        """Server-sent events for one run, replayable from ``Last-Event-ID``.

        SSE rather than a WebSocket: the traffic is one-way, ``EventSource`` reconnects
        by itself, and cancelling is an ordinary POST. A WebSocket would add a dependency
        and a handshake to buy nothing.
        """
        if jobs.get(job_id) is None:
            raise HTTPException(status_code=404, detail=f"no run {job_id!r}")

        header = request.headers.get("last-event-id")
        try:
            after = int(header) if header else 0
        except ValueError:
            after = 0

        async def stream() -> AsyncIterator[str]:
            cursor = after
            idle = 0.0
            while True:
                if await request.is_disconnected():
                    return
                events = jobs.events_since(job_id, cursor)
                for event in events:
                    cursor = event.id
                    idle = 0.0
                    yield f"id: {event.id}\nevent: {event.kind}\n"
                    yield f"data: {json.dumps(event.to_json_dict())}\n\n"
                record = jobs.get(job_id)
                if record is not None and record.status.terminal and not events:
                    return
                await asyncio.sleep(SSE_POLL_SECONDS)
                idle += SSE_POLL_SECONDS
                if idle >= SSE_KEEPALIVE_SECONDS:
                    idle = 0.0
                    yield ": keepalive\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @api.get("/api/datasets")
    def datasets() -> list[dict[str, Any]]:
        """Every dataset under the data root, with a cheap manifest summary."""
        return [entry.to_json_dict() for entry in list_datasets(settings.data_root)]

    @api.get("/api/datasets/manifest")
    def dataset_manifest(path: Annotated[str, Query()]) -> dict[str, Any]:
        """One dataset's full manifest. Never its circuit or DEM text."""
        target = _resolve(path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"no dataset at {path!r}")
        try:
            return full_manifest(target)
        # Corruption arrives as OSError, KeyError, BadZipFile or ArrowInvalid depending
        # on the format, so the catch has to be broad to report rather than 500.
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"{type(exc).__name__}: {exc}") from None

    @api.post("/api/datasets/validate")
    def dataset_validate(path: Annotated[str, Body(embed=True)]) -> dict[str, Any]:
        """Fully read a dataset and run the structural checks.

        The expensive call, and the only one that catches a truncated JSONL.
        """
        target = _resolve(path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"no dataset at {path!r}")
        try:
            return validate_at(target)
        # Corruption arrives as OSError, KeyError, BadZipFile or ArrowInvalid depending
        # on the format, so the catch has to be broad to report rather than 500.
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"{type(exc).__name__}: {exc}") from None

    @api.get("/api/datasets/download")
    def dataset_download(path: Annotated[str, Query()]) -> FileResponse:
        """Stream a dataset file to the browser."""
        target = _resolve(path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"no dataset at {path!r}")
        return FileResponse(target, filename=target.name)

    _mount_frontend(api)
    return api


def _mount_frontend(api: FastAPI) -> None:
    """Serve the built SPA, or explain how to build it.

    A missing build is answered with 503 and the exact command, not a blank page. The API
    stays up either way, so a job started from a tab that is still open is not killed by
    a frontend that has not been compiled.
    """
    if not static_is_built():

        @api.get("/{full_path:path}", include_in_schema=False)
        def not_built(full_path: str) -> PlainTextResponse:
            del full_path
            return PlainTextResponse(
                f"The qecgen web UI has not been built yet.\n\n    {BUILD_HINT}\n\n"
                "The API is running and usable at /api/docs.\n",
                status_code=503,
            )

        return

    api.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @api.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> FileResponse:
        """Serve real files, and index.html for client-side routes.

        ``StaticFiles(html=True)`` 404s on a deep link like ``/runs/abc`` because no such
        file exists; the router owns that path, not the filesystem.
        """
        candidate = STATIC_DIR / full_path
        if full_path and candidate.is_file() and STATIC_DIR in candidate.resolve().parents:
            return FileResponse(candidate)
        return FileResponse(STATIC_DIR / "index.html")
