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
import functools
import json
import os
import threading
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
from qecgen.dataset import DatasetMeta, StructureLevel
from qecgen.deletion import (
    RECYCLE_CAVEAT,
    DeletionPlan,
    DeletionRefusedError,
    RefusalReason,
    deletion_support,
    execute,
    plan_deletion,
)
from qecgen.environments import DriftAxis, build_environment, unbiased_point
from qecgen.exporters import EXPORTERS, get_exporter
from qecgen.run import (
    DEFAULT_SWEEP_DECODERS,
    PARTIAL_PREFIX,
    BenchmarkSpec,
    ConfiguredSpec,
    DriftSpec,
    GenerateSpec,
    JobSpec,
    MultiEnvSpec,
    QaSpec,
    ScoreSpec,
    SweepSpec,
    should_stream,
    sweep_tasks,
    total_shots,
)
from qecgen.sampling import DEFAULT_CHUNK_SIZE, packed_width
from qecgen.ui.datasets import (
    PathOutsideRootError,
    full_manifest,
    list_datasets,
    resolve_within,
    validate_at,
)
from qecgen.ui.jobs import (
    DEFAULT_WORKER_COMMAND,
    JobRecord,
    JobStore,
    run_input_paths,
    run_output_paths,
)
from qecgen.ui.protocol import spec_from_json
from qecgen.ui.schemas import (
    SELECTABLE_DRIFT_CONDITIONS,
    ConfiguredRequest,
    ConfiguredSweepRequest,
    JobRequest,
    SweepRequest,
)
from qecgen.ui.settings import WebSettings
from qecgen.ui.sweeps import list_sweeps, sweep_detail

__all__ = ["STATIC_DIR", "create_app", "static_is_built"]

STATIC_DIR = Path(__file__).parent / "static"
"""Where ``npm run build`` puts the frontend. Gitignored; built on demand."""

BUILD_HINT = "cd frontend && npm ci && npm run build"


def _submission_paths(spec: JobSpec) -> tuple[Path, ...]:
    """Reserve companions before they exist, including a drift run's whole directory."""
    match spec:
        case GenerateSpec() | MultiEnvSpec() | ConfiguredSpec():
            return (spec.out, *get_exporter(spec.fmt).companions(spec.out))
        case DriftSpec():
            return (spec.out,)
        case SweepSpec():
            return (spec.out, spec.plot_path, spec.summary_path)
        case ScoreSpec() | QaSpec() | BenchmarkSpec():
            return ()


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
                # Which formats can carry circuit and DEM text at --structure full. The
                # provenance view needs it, and `qecgen formats` prints the same column
                # from the same property.
                "carries_provenance": exporter.carries_provenance,
            }
            for name, exporter in sorted(EXPORTERS.items())
        ],
        "decoders": _decoder_options(),
        "default_sweep_decoders": list(DEFAULT_SWEEP_DECODERS),
        # The sweep form defaults `workers` to cpu_count - 2. Sent rather than
        # guessed in the browser, which has no way to know the host's core count.
        "cpu_count": os.cpu_count() or 1,
    }


@functools.cache
def _decoder_options() -> list[dict[str, Any]]:
    """Every sinter decoder name, with whether its backend is actually installed.

    Cached because ``find_spec`` walks ``sys.path`` per name and this runs on every
    capabilities request. The answer only changes if a package is installed while the
    server is running, which is not a case worth paying for on every page load.

    Still a *probe*, never an import: nothing here imports ``mwpf`` or ``fusion_blossom``.
    An ``import`` of either would be the first brick of the decoder adapter layer the
    README puts out of scope, and it is checkable — an ``mwpf.*`` entry in the mypy
    overrides means the boundary has been crossed.
    """
    from qecgen.decoders import check_decoder, known_decoder_names

    return [
        {
            "name": entry.name,
            "installed": entry.installed,
            "usable": entry.usable,
            "backing_package": entry.backing_package,
            "problem": entry.problem(),
        }
        for entry in (check_decoder(name) for name in known_decoder_names())
    ]


def correction_schema(dataset: Path, format_name: str | None = None) -> dict[str, Any]:
    """The correction shape this dataset expects, derived rather than guessed.

    A browser cannot check a correction file against a dataset on its own: the width it
    needs is ``packed_width(n_data_qubits)``, and ``n_data_qubits`` is not a manifest
    field — it comes from the circuit's final non-resetting measurement layer. Hardcoding
    ``d**2`` or ``2*d**2 - 2*d + 1`` in the frontend would be a second source of truth for
    a number this package already derives, and it would be wrong for any layout it did
    not anticipate.

    Built from a **noiseless** circuit, the same way :func:`~qecgen.run.score_correction`
    builds it, so asking this question about a ``frozen_prior`` test file reveals nothing
    about that file's error model.
    """
    from qecgen.circuits import build_circuit
    from qecgen.correction import extract_logical_operators
    from qecgen.exporters import read_manifest

    meta = DatasetMeta.from_json_dict(read_manifest(dataset, format_name))
    if meta.generation_config is not None and (
        meta.generation_config["mode"] == "hardware"
        or "stim_file" in meta.generation_config["circuit"]
    ):
        # Matching detector widths do not establish the meaning of correction columns.
        # An imported extraction circuit can reuse/reset qubits differently, so the
        # canonical layout would give a plausible but unaudited correction schema.
        raise ValueError(
            "Supplied-correction scoring requires the canonical generated layout; "
            "external hardware correction roles have not been audited"
        )
    circuit, _ = build_circuit(
        meta.distance, 0.0, rounds=meta.rounds, basis=meta.basis, rotated=meta.rotated
    )
    operators = extract_logical_operators(circuit, strict_single_basis=True)
    return {
        "n_data_qubits": operators.schema.n_data_qubits,
        "packed_width": operators.schema.packed_width,
        "n_observables": operators.n_observables,
        "shots": meta.shots,
        "schema_digest": operators.schema.digest(),
        "bit_order": operators.schema.bit_order,
        "content_hash": meta.content_hash,
        "drift_condition": str(meta.drift_condition),
    }


def _score_preview(spec: ScoreSpec) -> dict[str, Any]:
    """Check a correction against a dataset before either is read in full.

    This is the whole value of a preview for scoring. The commonest mistake is a width
    mismatch, and left to the run it surfaces from inside the scorer — after the dataset
    has been materialised, which for a million-shot CSV is minutes of pure-Python parsing
    before the user learns their array was the wrong shape.
    """
    from qecgen.correction import describe_correction_file

    schema = correction_schema(spec.dataset, spec.fmt)
    found = describe_correction_file(spec.correction)
    expected_width = schema["n_data_qubits"] if found.looks_unpacked else schema["packed_width"]
    problems: list[str] = []
    if found.width != expected_width:
        problems.append(
            f"the correction is {found.shots}x{found.width} but this dataset needs "
            f"{schema['shots']}x{expected_width}"
        )
    if found.shots != schema["shots"]:
        problems.append(
            f"the correction covers {found.shots} shots and the dataset holds "
            f"{schema['shots']}; every shot needs a correction"
        )
    return {
        **schema,
        "correction_shots": found.shots,
        "correction_width": found.width,
        "correction_dtype": found.dtype,
        # Detected, not asked. A user who answers this question wrong gets a plausible
        # number for a correction nobody proposed, rather than an error.
        "unpacked": found.looks_unpacked,
        "compatible": not problems,
        "problems": problems,
    }


def _qa_preview(spec: QaSpec) -> dict[str, Any]:
    """What statistical QA will cost, before any of it is spent.

    The shot count is an **upper bound** and is labelled as one. QA stops early in each
    environment once it has seen ``target_errors`` failures, so the real cost is usually
    far lower — reporting the bound as an estimate would make a fast job look like a slow
    one and vice versa.
    """
    from qecgen.exporters import read_manifest

    raw = read_manifest(spec.dataset, spec.fmt)
    environments = list(raw.get("environments") or [])
    return {
        "n_environments": len(environments),
        "max_shots_per_environment": spec.max_shots,
        "target_errors": spec.target_errors,
        "max_total_shots": spec.max_shots * max(len(environments), 1),
        "shots_in_file": raw.get("shots"),
        "resamples": True,
        "note": (
            "QA re-samples and decodes each environment; it does not read the file's "
            "shots. The shot count is a ceiling — each environment stops early once it "
            "has seen enough failures."
        ),
    }


def _decoder_problems(names: tuple[str, ...]) -> list[str]:
    """Why each named decoder cannot be used, empty when they all can.

    A probe, never an import: `check_decoder` resolves the name against sinter's registry
    and asks `find_spec` whether the backing package is present. Nothing here imports
    `mwpf` or `fusion_blossom` -- that would be the first brick of the adapter layer the
    README puts out of scope.
    """
    from qecgen.decoders import check_decoder

    return [problem for name in names if (problem := check_decoder(name).problem())]


def _sweep_preview(spec: SweepSpec) -> dict[str, Any]:
    """What a sweep will attempt, before it attempts any of it.

    Decoder availability is the part worth previewing. sinter discovers a missing backend
    only inside a worker, after every circuit in the grid has been built, so a name whose
    package is absent otherwise costs the whole construction before saying so. Reported
    here rather than refused at validation, so the form can say *which* decoder and *why*
    instead of rejecting the field.

    Deliberately offers no duration estimate, and that is not an omission: a sweep stops
    on ``max_errors`` with ``max_shots`` only as a ceiling, so the honest answer to "how
    long will this take" depends on the logical error rate the sweep exists to measure.
    ``max_shots_total`` is stated as the worst case it is, not as a forecast.
    """
    from qecgen.decoders import check_decoder

    availability = [check_decoder(name) for name in spec.decoders]
    tasks = sweep_tasks(spec)
    return {
        "distances": list(spec.distances),
        "error_rates": list(spec.error_rates),
        "decoders": [
            {
                "name": entry.name,
                "usable": entry.usable,
                "problem": entry.problem(),
                "backing_package": entry.backing_package,
            }
            for entry in availability
        ],
        "usable": all(entry.usable for entry in availability),
        "n_tasks": tasks,
        "max_errors": spec.max_errors,
        "max_shots_per_task": spec.max_shots,
        "max_shots_total": spec.max_shots * tasks,
        "workers": spec.workers,
        "results_path": str(spec.out),
        "plot_path": str(spec.plot_path),
        "summary_path": str(spec.summary_path),
        # A sweep overwrites its whole triple, and `staged` commits all three together, so
        # a rerun onto an existing stem replaces a complete set with a complete set. Worth
        # saying before the run rather than after it.
        "overwrites": spec.out.exists() or spec.summary_path.exists(),
        "note": (
            "No shot estimate: a sweep stops on max_errors, so max_shots is only a "
            "ceiling. max_shots_total is the worst case, not a forecast."
        ),
    }


def _benchmark_preview(spec: BenchmarkSpec) -> dict[str, Any]:
    """What a benchmark will decode, before it decodes any of it.

    Unlike QA this is an exact figure, not a ceiling: a benchmark decodes every shot
    the file holds and stops. Answered rather than refused -- a benchmark has a real
    dataset to describe, so declining the way a sweep does would withhold the one
    thing the caller can check before committing.
    """
    from qecgen.exporters import read_manifest

    raw = read_manifest(spec.dataset, spec.fmt)
    return {
        "shots": raw.get("shots"),
        "n_environments": len(list(raw.get("environments") or [])) or 1,
        "n_detectors": raw.get("n_detectors"),
        "drift_condition": raw.get("drift_condition"),
        "structure_level": raw.get("structure_level"),
        "resamples": False,
        "note": (
            "Decodes the shots in the file with PyMatching, using the error model rebuilt "
            "from each environment's recorded parameters. An oracle-calibrated ceiling."
        ),
    }


def _preview(spec: JobSpec) -> dict[str, Any]:
    """Cost estimate for a job, without sampling a single shot.

    ``build_environment`` builds the circuit and its decomposed DEM and stops there, so
    this costs milliseconds and answers the questions the terminal cannot answer before
    committing: how many detectors, how big the file, will it stream.
    """
    if isinstance(spec, ScoreSpec):
        return _score_preview(spec)
    if isinstance(spec, ConfiguredSpec):
        from qecgen.configuration import prepare_generation

        prepared = prepare_generation(spec.config)
        # Preparing checks circuit/profile compatibility and source bytes. Never
        # advance its iterator here: a preview must not generate training shots.
        audit = prepared.meta.generation_audit or {}
        return {
            "kind": "configured",
            "config": spec.config,
            "total_shots": spec.shots,
            "output_path": str(spec.out),
            "format": spec.fmt,
            "n_detectors": prepared.meta.n_detectors,
            "n_observables": prepared.meta.n_observables,
            "timing": {
                "physical_duration_s": audit.get("physical_duration_s"),
                "round_durations_s": audit.get("round_durations_s", []),
                "round_rates_hz": audit.get("round_rates_hz", []),
            },
            "note": (
                "Configuration, circuit compatibility and source files checked. "
                "Source files are checked again when the run starts. "
                "File size is not estimated for configured runs."
            ),
        }
    if isinstance(spec, QaSpec):
        return _qa_preview(spec)
    if isinstance(spec, BenchmarkSpec):
        return _benchmark_preview(spec)
    if isinstance(spec, SweepSpec):
        return _sweep_preview(spec)
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
    assert build.spec.channels is not None
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
    submission_lock = threading.Lock()

    def refuse_active_outputs(spec: JobSpec) -> None:
        # Every submitting route holds the same lock through check and enqueue.
        # This protects this server's queue; independent CLI processes remain outside it.
        candidates = _submission_paths(spec)
        for record in jobs.records():
            if record.status.terminal:
                continue
            for active in _submission_paths(spec_from_json(record.spec)):
                for candidate in candidates:
                    if (
                        active == candidate
                        or active in candidate.parents
                        or candidate in active.parents
                    ):
                        raise HTTPException(
                            status_code=409,
                            detail=f"Output is in use by run {record.id}: {candidate}",
                        )

    def browser_record(record: JobRecord) -> dict[str, Any]:
        # Derived sweep seeds use all 64 bits. Keep an exact textual copy for
        # browsers, whose JSON numeric representation cannot preserve them.
        return {**record.to_json_dict(), "spec_json": json.dumps(record.spec, indent=2)}

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
            "deletion": deletion_support(),
        }

    @api.post("/api/preview")
    def preview(request: Annotated[JobRequest, Body()]) -> dict[str, Any]:
        """Cost estimate for a run that has not been submitted.

        A sweep is refused here rather than answered. It has no shots, no detectors
        and no file size to estimate, so it shares no field with the reply this route
        gives -- and a preview that silently returned a different shape for one mode
        is how a caller ends up reading a key that is never there.
        """
        try:
            spec = request.to_spec(settings.data_root)
        except PathOutsideRootError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        if isinstance(spec, SweepSpec):
            raise HTTPException(
                status_code=400,
                detail="a sweep has no dataset to estimate; POST to /api/sweeps/preview",
            )
        try:
            return _preview(spec)
        except PathOutsideRootError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except OSError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @api.post("/api/configured/layout")
    def configured_layout(request: ConfiguredRequest) -> dict[str, Any]:
        from qecgen.ui.configured import layout

        try:
            return layout(request.config, settings.data_root)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @api.post("/api/configured/sweep-preview")
    def configured_sweep_preview(request: ConfiguredSweepRequest) -> dict[str, Any]:
        try:
            specs = request.specs(settings.data_root)
            for spec in specs:
                _preview(spec)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return {
            "field": request.field,
            "values": request.values,
            "total_shots": sum(spec.shots for spec in specs),
            "runs": [
                {
                    "output_path": str(spec.out),
                    "seed": str(spec.config["sampling"]["seed"]),
                    "config_json": json.dumps(spec.config, indent=2),
                }
                for spec in specs
            ],
            "note": (
                "Each point is an independent dataset job with a derived seed and indexed "
                "filename. "
                "Covariate-only sweeps do not introduce a physical response."
            ),
        }

    @api.post("/api/configured/sweep", status_code=202)
    def configured_sweep(request: ConfiguredSweepRequest) -> dict[str, Any]:
        try:
            specs = request.specs(settings.data_root)
            for spec in specs:
                _preview(spec)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        with submission_lock:
            for spec in specs:
                refuse_active_outputs(spec)
            records = []
            try:
                for spec in specs:
                    records.append(browser_record(jobs.submit(spec)))
            except OSError as exc:
                # Queueing independent jobs is not an atomic filesystem transaction.
                # Name accepted jobs rather than telling a caller that nothing ran.
                accepted = ", ".join(record["id"] for record in records) or "none confirmed"
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Could not queue the complete sweep: {exc}. "
                        f"Accepted run IDs: {accepted}. Check Runs before retrying."
                    ),
                ) from None
        return {"runs": records}

    @api.post("/api/sweeps/preview")
    def sweep_preview(request: Annotated[SweepRequest, Body()]) -> dict[str, Any]:
        """The grid a sweep will collect, and where its three files will land."""
        try:
            return _sweep_preview(request.to_spec(settings.data_root))
        except PathOutsideRootError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @api.post("/api/runs", status_code=202)
    def submit_run(request: Annotated[JobRequest, Body()]) -> dict[str, Any]:
        """Queue a run and return its record. Nothing has been sampled yet."""
        try:
            spec = request.to_spec(settings.data_root)
        except PathOutsideRootError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        if isinstance(spec, SweepSpec):
            # Checked here rather than on `SweepRequest`, because the same model backs
            # `/api/sweeps/preview` -- and there an unusable decoder is the answer, not an
            # error: the preview's job is to say which one and why while the form is still
            # being filled in. Submitting one is refused, because sinter would otherwise
            # discover it inside a worker after building the entire task grid.
            problems = _decoder_problems(spec.decoders)
            if problems:
                raise HTTPException(
                    status_code=422,
                    detail=[
                        {"loc": ["body", "decoders"], "msg": problem, "type": "value_error"}
                        for problem in problems
                    ],
                )
        with submission_lock:
            refuse_active_outputs(spec)
            return browser_record(jobs.submit(spec))

    @api.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        """Every run this server knows about, newest first."""
        return [browser_record(record) for record in jobs.records()]

    @api.get("/api/runs/{job_id}")
    def get_run(job_id: str) -> dict[str, Any]:
        """One run's record."""
        record = jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"no run {job_id!r}")
        return browser_record(record)

    @api.post("/api/runs/{job_id}/cancel")
    def cancel_run(job_id: str) -> dict[str, Any]:
        """Ask a run to stop at its next chunk boundary."""
        if jobs.get(job_id) is None:
            raise HTTPException(status_code=404, detail=f"no run {job_id!r}")
        if not jobs.cancel(job_id):
            raise HTTPException(status_code=409, detail="run has already finished")
        record = jobs.get(job_id)
        assert record is not None
        return browser_record(record)

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
                if record is None:
                    # The run was deleted while this stream was open. There will never be
                    # another event, and the terminal test below can only fire for a record
                    # that still exists -- so without this the generator polls forever
                    # behind a client that has gone, emitting keepalives into a dead socket.
                    return
                if record.status.terminal and not events:
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

    def record_path_of(job_id: str) -> Path:
        """Where the durable record for one run lives."""
        return settings.runs_dir / f"{job_id}.json"

    def _relative_or_absolute(root: Path, target: Path) -> str:
        """``target`` as the browser should show it: root-relative, or absolute if outside.

        Every listing route emits root-relative paths, but a run record carries absolute
        ones and may name a file under a previous ``--data-root``. Printing a bare filename
        for one of those would suggest it sits in the directory being browsed.
        """
        try:
            return str(target.relative_to(root)).replace("\\", "/")
        except ValueError:
            return str(target)

    def _plan_or_refuse(target: Path) -> DeletionPlan:
        """Plan a deletion, translating the core's refusals into status codes.

        One mapping, stated once. **400 means the server will not touch that path, ever** --
        outside the root, a reserved name, a run record, a directory it did not write. **409
        means not right now** and is the same meaning ``cancel_run`` already gives it: a
        staging directory a live run is writing into, which becomes deletable the moment that
        run stops. Splitting them matters because the browser offers a different next step
        for each, and folding both into 400 would put "cancel that run" behind a refusal that
        says the path is permanently off limits.
        """
        try:
            return plan_deletion(target, reserved=(settings.runs_dir,))
        except DeletionRefusedError as exc:
            if exc.reason is RefusalReason.NOT_FOUND:
                raise HTTPException(status_code=404, detail=str(exc)) from None
            if exc.reason is RefusalReason.STAGING_LIVE:
                raise HTTPException(status_code=409, detail=str(exc)) from None
            raise HTTPException(status_code=400, detail=str(exc)) from None

    def _refuse_if_live(plan: DeletionPlan) -> None:
        """409 when an unfinished run in this process is about to write one of these files."""
        for entry in plan.files:
            blocking = jobs.blocking_run(entry.path)
            if blocking is not None:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"{entry.path.name} is the output of run {blocking.job_id!r}, which "
                        f"is still {blocking.status}. Cancel that run first."
                    ),
                )

    def _orphaned_payload(plan: DeletionPlan) -> list[dict[str, Any]]:
        """Run records this deletion would leave describing nothing."""
        return [
            {
                "id": record.id,
                "mode": record.mode,
                "created_at": record.created_at,
                "status": str(record.status),
            }
            for record in jobs.orphaned_runs([entry.path for entry in plan.files])
        ]

    @api.get("/api/datasets/delete-preview")
    def dataset_delete_preview(path: Annotated[str, Query()]) -> dict[str, Any]:
        """Exactly what deleting this file would remove, before anything moves.

        The confirmation dialog's whole content, and the same precedent as ``/api/preview``
        and the CLI's rule that every command prints its resolved config before doing work:
        the destructive act is never the first time the user sees the list. It matters more
        here than anywhere, because the set is not guessable from the row that was clicked --
        an ``.ml.csv`` carries sidecars the listing never shows, a sweep's results table
        drags two siblings, and a drift member takes its whole study.
        """
        plan = _plan_or_refuse(_resolve(path))
        payload = plan.to_json_dict(settings.data_root)
        payload["orphaned_runs"] = _orphaned_payload(plan)
        return payload

    @api.delete("/api/datasets")
    def delete_dataset(
        path: Annotated[str, Query()],
        delete_runs: Annotated[bool, Query()],
    ) -> dict[str, Any]:
        """Send one artifact -- and everything that travels with it -- to the recycle bin.

        ``delete_runs`` is **required and has no server-side default**. Its "on by default"
        lives in the browser, in the CLI and in the dialog; a server that forgot run records
        because a future caller omitted a query parameter would be deciding something it was
        never asked about, and FastAPI's 422 makes that unreachable.

        Returns 200 with per-file outcomes rather than failing on a file it could not remove.
        A locked sidecar is a row in ``failed``, not a status code -- the same shape
        ``/api/datasets/validate`` uses when it returns ``ok: false``.
        """
        plan = _plan_or_refuse(_resolve(path))
        _refuse_if_live(plan)
        orphaned = jobs.orphaned_runs([entry.path for entry in plan.files]) if delete_runs else []
        report = execute(plan)
        payload = report.to_json_dict(settings.data_root)
        forgotten: list[str] = []
        # After the files, never before: a record removed first would leave orphan files with
        # nothing left to explain where they came from.
        for record in orphaned:
            outcome = jobs.discard(record.id)
            if outcome is not None:
                forgotten.append(record.id)
        payload["forgotten_runs"] = forgotten
        return payload

    @api.get("/api/runs/{job_id}/delete-preview")
    def run_delete_preview(job_id: str) -> dict[str, Any]:
        """What deleting this run would remove, and what it would deliberately keep.

        Always reports the full output expansion regardless of the checkbox, because the
        checkbox is presentational: toggling it must not cost a round trip, and a *read* with
        a flag is a flag that can be got wrong on the safe operation.

        ``inputs_kept`` earns its place. A ``score`` run's input dataset is named all over
        the run detail page, and a user deleting that run has no other way to learn it is not
        about to go with it.
        """
        record = jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"no run {job_id!r}")
        plans: list[DeletionPlan] = []
        outside: list[str] = []
        blocked: list[dict[str, str]] = []
        for target in run_output_paths(record):
            try:
                resolved = resolve_within(settings.data_root, target)
            except PathOutsideRootError:
                # Kept, not deleted, and the record is still removable. Honouring an absolute
                # path out of a run record would launder one past `resolve_within`; refusing
                # the whole delete would strand every record adopted from a previous
                # --data-root, permanently.
                outside.append(str(target))
                continue
            try:
                plans.append(plan_deletion(resolved, reserved=(settings.runs_dir,)))
            except DeletionRefusedError as exc:
                blocked.append({"path": str(target), "reason": str(exc)})
        merged = DeletionPlan.merge(plans)
        payload: dict[str, Any] = (
            merged.to_json_dict(settings.data_root)
            if merged is not None
            else {
                "files": [],
                "missing": [],
                "n_files": 0,
                "total_bytes": 0,
                "caveat": RECYCLE_CAVEAT,
            }
        )
        payload["run"] = {
            "id": record.id,
            "mode": record.mode,
            "status": str(record.status),
            "created_at": record.created_at,
            "record_path": _relative_or_absolute(settings.data_root, record_path_of(record.id)),
        }
        payload["inputs_kept"] = [
            _relative_or_absolute(settings.data_root, target) for target in run_input_paths(record)
        ]
        payload["outside_root"] = outside
        payload["blocked"] = blocked
        return payload

    @api.delete("/api/runs/{job_id}")
    def delete_run(
        job_id: str,
        delete_files: Annotated[bool, Query()],
    ) -> dict[str, Any]:
        """Forget a finished run, and by request the files it wrote.

        ``delete_files`` is **required and has no server-side default**, for the same reason
        as ``delete_runs`` above.

        Order is load-bearing: files first, record last. A 409 raised after the record was
        already forgotten would leave orphan files with no run left to explain them, and
        nothing to retry against.
        """
        record = jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"no run {job_id!r}")
        if not record.status.terminal:
            raise HTTPException(
                status_code=409,
                detail=f"run {job_id!r} is {record.status}; cancel it first, then delete",
            )

        payload: dict[str, Any] = {
            "id": job_id,
            "deleted_files": delete_files,
            "files": [],
            "removed": [],
            "failed": [],
            "n_removed": 0,
            "bytes_removed": 0,
            "complete": True,
            "kept": [],
            "caveat": RECYCLE_CAVEAT,
        }
        if delete_files:
            plans: list[DeletionPlan] = []
            kept: list[str] = []
            for target in run_output_paths(record):
                try:
                    resolved = resolve_within(settings.data_root, target)
                except PathOutsideRootError:
                    kept.append(str(target))
                    continue
                try:
                    plans.append(plan_deletion(resolved, reserved=(settings.runs_dir,)))
                except DeletionRefusedError as exc:
                    # One unplannable member -- a hand-edited record can name anything --
                    # must not make the run undeletable.
                    kept.append(f"{target}: {exc}")
            merged = DeletionPlan.merge(plans)
            if merged is not None:
                _refuse_if_live(merged)
                payload.update(execute(merged).to_json_dict(settings.data_root))
            payload["kept"] = kept

        outcome = jobs.discard(job_id)
        payload["record_removed"] = outcome.removed if outcome is not None else False
        payload["record_problem"] = outcome.problem if outcome is not None else None
        return payload

    @api.get("/api/datasets/provenance")
    def dataset_provenance(path: Annotated[str, Query()]) -> dict[str, Any]:
        """Circuit and DEM text, for a file that stores it. Its own route, on purpose.

        This is the browser's ``qecgen inspect --show-text``, and the rules around it are
        part of the data contract rather than presentation preferences:

        * **Never folded into the manifest.** ``/api/datasets/manifest`` returns exactly
          what a decoder may read, and under ``FROZEN_PRIOR`` this text is precisely what
          the condition withholds. A manifest that carried it would hand a test file's own
          error model to anything that reads manifests.
        * **Never fetched without being asked for.** The Datasets page must not load this
          alongside the manifest; the whole separation is defeated by a UI that renders
          them side by side because both were there.
        * **Warned about, not refused.** The CLI prints it on request and the bytes are in
          the file either way. Refusing here while allowing it there would be an
          inconsistency between two front ends over the same file — and it is the reader's
          discipline this protects, not the file's secrecy. So the response states the
          file's own ``drift_condition`` and lets the caller decide.
        """
        target = _resolve(path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"no dataset at {path!r}")
        try:
            from qecgen.exporters import provenance_formats, read_manifest, read_provenance

            meta = DatasetMeta.from_json_dict(read_manifest(target))
            payload = read_provenance(target)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"{type(exc).__name__}: {exc}") from None
        return {
            "path": path,
            "structure_level": str(meta.structure_level),
            "drift_condition": str(meta.drift_condition),
            "structure_source_environment_id": meta.structure_source_environment_id,
            "environments": (payload or {}).get("environments", []),
            "stored": payload is not None,
            "formats_that_store_it": list(provenance_formats()),
        }

    @api.get("/api/datasets/correction-schema")
    def dataset_correction_schema(path: Annotated[str, Query()]) -> dict[str, Any]:
        """The correction shape this dataset expects, so a form can check a file."""
        target = _resolve(path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"no dataset at {path!r}")
        try:
            return correction_schema(target)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"{type(exc).__name__}: {exc}") from None

    @api.get("/api/sweeps")
    def sweeps() -> list[dict[str, Any]]:
        """Every sweep result set under the data root, newest first.

        Indexed on ``*.threshold.json`` because it is the only one of a sweep's three
        files that **names itself**: a ``.csv`` is ambiguous with the dataset format and a
        ``.png`` is ambiguous with anything. Listing sweeps out of the dataset browser
        instead would have meant either widening that browser's contract past "every
        dataset file" or teaching it to recognise a results table by its columns.

        Includes sweeps this server did not run — one from a previous session, or from
        `qecgen sweep` in a terminal. The run record knows what *a run* produced; this
        knows what is on disk.
        """
        return [entry.to_json_dict() for entry in list_sweeps(settings.data_root)]

    @api.get("/api/sweeps/detail")
    def sweep_at(path: Annotated[str, Query()]) -> dict[str, Any]:
        """One sweep's points and summary. ``path`` names its ``.threshold.json``."""
        target = _resolve(path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"no sweep summary at {path!r}")
        try:
            return sweep_detail(settings.data_root, target)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        # A hand-edited or half-written sidecar arrives as JSONDecodeError or KeyError, and
        # a results table with the wrong columns as ValueError. Reported, not 500ed.
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"{type(exc).__name__}: {exc}") from None

    @api.get("/api/sweeps/plot")
    def sweep_plot(path: Annotated[str, Query()]) -> FileResponse:
        """Serve a sweep plot **inline**, so an ``<img>`` can render it.

        Deliberately not ``/api/datasets/download``. That route passes ``filename=``,
        which Starlette turns into ``Content-Disposition: attachment`` — its contract is
        "a dataset is a file you save, never something that renders in the tab", and
        weakening it so a picture displays would be the wrong trade. This one names a
        media type instead and passes no filename, so no disposition header is set at all.

        A re-run sweep writes the **same path**, so a cached response would show the
        previous run's plot. The ``/api/*`` middleware sets ``Cache-Control: no-store``,
        which is exactly the failure already recorded for a cached capabilities GET.
        """
        target = _resolve(path)
        if target.suffix.lower() != ".png":
            raise HTTPException(
                status_code=400,
                detail=f"{path!r} is not a .png; this route only serves sweep plots",
            )
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"no plot at {path!r}")
        # This is the one route that asks a browser to *render* a file out of the data
        # root rather than download it, so it must not be able to serve something that
        # executes. The suffix check stops the argument arising; `nosniff` stops the
        # browser second-guessing the media type if it ever did.
        return FileResponse(
            target,
            media_type="image/png",
            headers={"X-Content-Type-Options": "nosniff"},
        )

    @api.get("/api/corrections")
    def corrections() -> list[dict[str, Any]]:
        """Every ``.npz`` under the data root that holds a proposed correction.

        Listed separately from datasets rather than folded into that view. They share an
        extension and nothing else: a correction is an *input* to scoring, not a thing
        with shots and a content hash, and the dataset browser's columns describe none
        of it.

        Each file's shape comes from the ``.npy`` header inside the zip — about 128 bytes
        — so listing a directory of them costs no decompression.
        """
        from qecgen.correction import describe_correction_file

        found: list[dict[str, Any]] = []
        for candidate in sorted(settings.data_root.rglob("*.npz")):
            if any(part.startswith(PARTIAL_PREFIX) for part in candidate.parts):
                continue
            try:
                described = describe_correction_file(candidate)
            except Exception:
                # Not a correction file. Almost always a dataset, which the dataset
                # browser already lists; either way it is not this endpoint's business
                # and reporting it here would put every dataset in the correction picker.
                continue
            found.append(
                {
                    "path": str(candidate.relative_to(settings.data_root)).replace("\\", "/"),
                    "name": candidate.name,
                    "shots": described.shots,
                    "width": described.width,
                    "dtype": described.dtype,
                    "unpacked": described.looks_unpacked,
                    "size_bytes": candidate.stat().st_size,
                }
            )
        return found

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
