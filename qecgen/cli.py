"""Command line interface.

Every command prints its fully resolved configuration before doing any work, so a
terminal log is a complete record of what was run — including the values that were
defaulted rather than passed.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from qecgen import __version__
from qecgen import run as runner
from qecgen.circuits import Basis, NoiseModel
from qecgen.dataset import DatasetMeta, DriftCondition, StructureLevel
from qecgen.environments import DriftAxis
from qecgen.exporters import (
    EXPORTERS,
    Exporter,
    NotAQecgenDatasetError,
    get_exporter,
    infer_format,
    provenance_formats,
    read_manifest,
    read_provenance,
)
from qecgen.sampling import DEFAULT_CHUNK_SIZE
from qecgen.ui import DEFAULT_PORT
from qecgen.validate import validate_dataset

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Generate and validate surface code syndrome-to-logical-frame datasets.",
)
console = Console()


def _print_config(command: str, config: dict[str, Any]) -> None:
    """Print the fully resolved configuration as a table."""
    table = Table(title=f"qecgen {command}  (v{__version__})", show_header=True)
    table.add_column("setting", style="cyan", no_wrap=True)
    table.add_column("value", style="white")
    for key, value in config.items():
        table.add_row(key, str(value))
    console.print(table)


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )


@contextlib.contextmanager
def _run_progress(spec: runner.RunSpec) -> Iterator[tuple[runner.ProgressHook, runner.PhaseHook]]:
    """Yield the hooks that drive a live progress bar for one run.

    The bar counts *sampled* shots, and the description switches to "writing" once the
    last chunk lands, because concatenation, hashing and gzip export all happen after
    the bar fills. A bar that sits at 100% with no explanation reads as a hang.
    """
    with _progress() as progress:
        task = progress.add_task("sampling", total=runner.total_shots(spec))
        yield (
            lambda n: progress.advance(task, n),
            lambda phase: progress.update(task, description=phase),
        )


def _report_run(files: list[runner.WrittenFile]) -> None:
    for written in files:
        _report_written(written.path, written.shots, written.content_hash)


def _cli_exporter(fmt: str) -> Exporter:
    """Registry lookup, restated as a CLI argument error.

    A raw ValueError traceback for a typo in ``--format`` is inconsistent with
    :func:`_infer_format`, which already restates the same registry failure.
    """
    try:
        return get_exporter(fmt)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None


def _require_format_extension_agreement(fmt: str, out: Path) -> None:
    """Refuse an ``--out`` whose extension names a *different* registered format.

    ``--format jsonl --out data.h5`` writes JSONL bytes into a ``.h5`` file, and
    ``validate``/``inspect`` then infer HDF5 from the extension and fail on a file that
    is not corrupt. Extensions that map to no registered format are left alone — they
    are unusual, not contradictory, and ``--format`` disambiguates them on read.
    """
    exporter = _cli_exporter(fmt)
    try:
        inferred = infer_format(out)
    except ValueError:
        return
    if inferred != exporter.format_name:
        raise typer.BadParameter(
            f"--out {out} has the extension of the {inferred!r} format but --format is "
            f"{exporter.format_name!r}; validate and inspect would misread the file. "
            f"Use a {exporter.extension} path or change --format."
        )


def _require_existing_file(path: Path) -> Path:
    """Refuse a path that is not there, as a parameter error rather than a traceback.

    The commonest mistake a read command sees, and until now the least well handled: a
    mistyped path reached whichever library the format uses and came back as an h5py
    OSError, a ``BadZipFile`` or a ``FileNotFoundError`` traceback — three different
    unhelpful answers to one ordinary question. ``_cli_exporter`` and ``_infer_format``
    already restate their failures this way; this completes the set.
    """
    if not path.exists():
        raise typer.BadParameter(f"{path} does not exist")
    if not path.is_file():
        raise typer.BadParameter(f"{path} is a directory, not a dataset file")
    return path


def _resolved_config(command: str, spec: runner.JobSpec) -> None:
    """Resolve and print a run's configuration, restating a refusal as a parameter error.

    ``resolved_config`` raises for a combination the domain will not honour (chiefly
    ``CODE_CAPACITY`` with ``rounds != 1``), which is why it is called *before* the table
    is printed rather than while building it: a config that cannot be honoured must never
    reach the log as though it were the record of a run.
    """
    try:
        config = runner.resolved_config(spec)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    _print_config(command, config)


@app.command()
def generate(
    distance: Annotated[int, typer.Option(help="Code distance.")] = 5,
    p: Annotated[float, typer.Option(help="Physical error rate.")] = 0.005,
    shots: Annotated[int, typer.Option(help="Number of shots.")] = 1_000_000,
    noise: Annotated[NoiseModel, typer.Option(help="Noise model.")] = (
        NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL
    ),
    rounds: Annotated[int | None, typer.Option(help="Rounds. Defaults to distance.")] = None,
    basis: Annotated[Basis, typer.Option(help="Memory basis.")] = Basis.Z,
    rotated: Annotated[bool, typer.Option(help="Rotated surface code layout.")] = True,
    fmt: Annotated[str, typer.Option("--format", help="Output format.")] = "hdf5",
    structure: Annotated[StructureLevel, typer.Option(help="Structural detail.")] = (
        StructureLevel.NONE
    ),
    emit_mechanisms: Annotated[
        bool, typer.Option(help="Also record DEM mechanisms (Contract B).")
    ] = False,
    seed: Annotated[int, typer.Option(help="Master seed.")] = 0,
    chunk_size: Annotated[int, typer.Option(help="Shots per sample call.")] = DEFAULT_CHUNK_SIZE,
    out: Annotated[Path, typer.Option(help="Output file.")] = Path("data/dataset.h5"),
) -> None:
    """Generate a single-environment dataset."""
    spec = runner.GenerateSpec(
        distance=distance,
        p=p,
        shots=shots,
        seed=seed,
        out=out,
        fmt=fmt,
        noise_model=noise,
        rounds=rounds,
        basis=basis,
        rotated=rotated,
        structure_level=structure,
        emit_mechanisms=emit_mechanisms,
        chunk_size=chunk_size,
    )
    # Reject an unknown format or a contradictory extension before printing a config we
    # cannot honour, and resolve rounds through the real rule so the printed value is
    # the one the run uses (CODE_CAPACITY resolves to 1, not to distance).
    _require_format_extension_agreement(fmt, out)
    _resolved_config("generate", spec)

    # Above one chunk, HDF5 goes through the streaming writer so memory stays flat
    # regardless of --shots. Other formats have no incremental writer, so they still
    # materialise; that limit is stated in the README rather than implied away.
    if runner.should_stream(fmt, shots, chunk_size):
        console.print("[dim]streaming to HDF5; memory stays bounded at one chunk[/dim]")

    with _run_progress(spec) as (advance, phase):
        _report_run(runner.generate_single(spec, progress=advance, on_phase=phase))


@app.command(name="multi-env")
def multi_env(
    distance: Annotated[int, typer.Option(help="Code distance.")] = 5,
    p: Annotated[list[float] | None, typer.Option(help="One rate per environment.")] = None,
    shots_per_env: Annotated[int, typer.Option(help="Shots per environment.")] = 250_000,
    noise: Annotated[NoiseModel, typer.Option()] = NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL,
    axis: Annotated[
        DriftAxis, typer.Option("--drift-axis", "--axis", help="Which property varies.")
    ] = DriftAxis.P,
    base_p: Annotated[float | None, typer.Option(help="Base rate for non-p axes.")] = None,
    shuffle: Annotated[
        bool,
        typer.Option(
            help=(
                "Interleave environments with a seeded shuffle. Turn this off only to "
                "debug: unshuffled, every shot of environment 0 precedes every shot of "
                "environment 1, so the row index alone tells a model which environment a "
                "shot came from and an index-ordered split tears along that boundary."
            )
        ),
    ] = True,
    rounds: Annotated[int | None, typer.Option()] = None,
    basis: Annotated[Basis, typer.Option()] = Basis.Z,
    rotated: Annotated[bool, typer.Option(help="Rotated surface code layout.")] = True,
    fmt: Annotated[str, typer.Option("--format")] = "hdf5",
    structure: Annotated[StructureLevel, typer.Option()] = StructureLevel.NONE,
    emit_mechanisms: Annotated[bool, typer.Option()] = False,
    seed: Annotated[int, typer.Option()] = 0,
    chunk_size: Annotated[int, typer.Option()] = DEFAULT_CHUNK_SIZE,
    out: Annotated[Path, typer.Option()] = Path("data/train_multi.h5"),
) -> None:
    """Generate a pooled multi-environment dataset with shuffled, labelled shots."""
    if p is None and axis is not DriftAxis.P:
        # The default list is a list of physical error rates. Reusing it as xz_bias
        # etas or measurement ratios silently runs a study at nonsense coordinates.
        raise typer.BadParameter(
            f"--axis {axis} requires explicit --p values: the default list "
            "[0.003, 0.005, 0.008, 0.012] is a list of physical error rates, not "
            f"coordinates on the {axis} axis."
        )
    rates = list(p) if p else [0.003, 0.005, 0.008, 0.012]
    spec = runner.MultiEnvSpec(
        distance=distance,
        axis_values=tuple(rates),
        shots_per_env=shots_per_env,
        seed=seed,
        out=out,
        fmt=fmt,
        axis=axis,
        base_p=base_p,
        shuffle=shuffle,
        noise_model=noise,
        rounds=rounds,
        basis=basis,
        rotated=rotated,
        structure_level=structure,
        emit_mechanisms=emit_mechanisms,
        chunk_size=chunk_size,
    )
    _require_format_extension_agreement(fmt, out)
    _resolved_config("multi-env", spec)

    with _run_progress(spec) as (advance, phase):
        _report_run(runner.generate_multi(spec, progress=advance, on_phase=phase))


@app.command()
def drift(
    distance: Annotated[int, typer.Option()] = 5,
    train_p: Annotated[float, typer.Option(help="Training environment rate.")] = 0.005,
    test_p: Annotated[list[float] | None, typer.Option(help="Test rates.")] = None,
    condition: Annotated[
        DriftCondition, typer.Option(help="Where test-file structure comes from.")
    ] = DriftCondition.FROZEN_PRIOR,
    axis: Annotated[
        DriftAxis, typer.Option("--drift-axis", "--axis", help="Which property varies.")
    ] = DriftAxis.P,
    shots: Annotated[int, typer.Option(help="Shots per file.")] = 100_000,
    noise: Annotated[NoiseModel, typer.Option()] = NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL,
    rounds: Annotated[int | None, typer.Option()] = None,
    basis: Annotated[Basis, typer.Option()] = Basis.Z,
    rotated: Annotated[bool, typer.Option(help="Rotated surface code layout.")] = True,
    fmt: Annotated[str, typer.Option("--format")] = "hdf5",
    structure: Annotated[StructureLevel, typer.Option()] = StructureLevel.DEM,
    emit_mechanisms: Annotated[
        bool,
        typer.Option(
            help=(
                "Also record DEM mechanisms (Contract B). Under frozen_prior the domain "
                "refuses this when the training and test DEMs enumerate their mechanisms "
                "differently, because column k would denote different mechanisms in each."
            )
        ),
    ] = False,
    seed: Annotated[int, typer.Option()] = 0,
    chunk_size: Annotated[int, typer.Option()] = DEFAULT_CHUNK_SIZE,
    out: Annotated[Path, typer.Option(help="Output directory.")] = Path("data/drift"),
) -> None:
    """Generate a training set plus drifted test sets under an explicit condition."""
    values = list(test_p) if test_p else [0.007, 0.010, 0.014]
    spec = runner.DriftSpec(
        distance=distance,
        train_p=train_p,
        test_values=tuple(values),
        shots=shots,
        seed=seed,
        condition=condition,
        out=out,
        fmt=fmt,
        axis=axis,
        noise_model=noise,
        rounds=rounds,
        basis=basis,
        rotated=rotated,
        structure_level=structure,
        emit_mechanisms=emit_mechanisms,
        chunk_size=chunk_size,
    )
    _cli_exporter(fmt)
    _resolved_config("drift", spec)

    try:
        with _run_progress(spec) as (advance, phase):
            written = runner.generate_drift(spec, progress=advance, on_phase=phase)
    except ValueError as exc:
        # The filename-collision check now runs before sampling rather than after every
        # dataset was already built.
        raise typer.BadParameter(str(exc)) from None

    for file in written:
        console.print(
            f"  [green]wrote[/green] {file.path}  shots={file.shots:,}  "
            f"condition={file.drift_condition}  "
            f"structure_from_env={file.structure_source_environment_id}"
        )


@app.command()
def sweep(
    distances: Annotated[list[int] | None, typer.Option(help="Distances to sweep.")] = None,
    p_range: Annotated[
        str, typer.Option(help="Range as low:high:count, e.g. 0.001:0.020:8.")
    ] = "0.001:0.020:8",
    max_errors: Annotated[int, typer.Option(help="Primary stopping condition.")] = 500,
    max_shots: Annotated[int, typer.Option(help="Ceiling.")] = 100_000_000,
    workers: Annotated[int, typer.Option()] = 4,
    noise: Annotated[NoiseModel, typer.Option()] = NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL,
    basis: Annotated[Basis, typer.Option()] = Basis.Z,
    decoder: Annotated[
        list[str] | None,
        typer.Option("--decoder", help="Decoder to run; repeatable. Default: pymatching."),
    ] = None,
    out: Annotated[Path, typer.Option(help="CSV output path.")] = Path("results/sweep.csv"),
) -> None:
    """Run a sinter-driven threshold sweep and emit CSV, a plot and a threshold summary.

    The orchestration is :func:`qecgen.run.run_threshold_sweep`. What stays here is
    argument resolution -- expanding ``--p-range``, defaulting the distances and the
    decoder list -- and presentation.
    """
    from qecgen.decoders import resolve_decoders

    low, high, count = _parse_range(p_range)
    try:
        spec = runner.SweepSpec(
            distances=tuple(distances) if distances else (3, 5, 7),
            error_rates=tuple(runner.expand_range(low, high, count)),
            out=out,
            max_errors=max_errors,
            max_shots=max_shots,
            workers=workers,
            decoders=tuple(decoder) if decoder else runner.DEFAULT_SWEEP_DECODERS,
            noise_model=noise,
            basis=basis,
        )
    except ValueError as exc:
        # Structural refusals from the spec itself -- a .png target, a rate outside [0, 1],
        # a duplicate on any axis. Restated as a parameter error rather than reaching the
        # user as a traceback out of a dataclass constructor.
        raise typer.BadParameter(str(exc)) from None
    _resolved_config("sweep", spec)

    # Before any collection starts. sinter discovers a bad decoder name only inside a
    # worker, after every circuit in the grid has been built.
    try:
        resolve_decoders(spec.decoders)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None

    with _progress() as bar:
        task = bar.add_task("collecting", total=runner.job_total(spec)[0])

        def advance(done: int) -> None:
            bar.advance(task, done)

        def phase(text: str) -> None:
            bar.update(task, description=text[:60])

        try:
            result = runner.analyse(spec, progress=advance, on_phase=phase)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from None

    console.print(f"[green]wrote[/green] {spec.out}, {spec.plot_path} and {spec.summary_path}")
    _report_sweep(result.summary)


def _report_sweep(summary: dict[str, Any]) -> None:
    """Print the crossing and suppression summary, as results rather than verdicts.

    Reads the same payload that goes into the `.threshold.json` sidecar, so the terminal
    and the file cannot disagree about a crossing.
    """
    for decoder, entry in sorted(summary["decoders"].items()):
        crossing = entry["crossing_p"]
        location = "not visible in the sampled range" if crossing is None else f"p ~ {crossing:g}"
        console.print(f"\n[bold]{decoder}[/bold]  crossing: {location}")
        rows = entry["suppression"]
        if not rows:
            console.print("  [dim]no suppression fit: fewer than two distances with errors[/dim]")
            continue
        for row in rows:
            # Lambda only means anything below threshold, so say which side each row is
            # on rather than letting a super-threshold number read like a suppression
            # result.
            regime = (
                ""
                if crossing is None or row["p"] < crossing
                else "  [yellow](at or above crossing)[/yellow]"
            )
            excluded = (
                f" excluded(k=0)={row['excluded_zero_error']}" if row["excluded_zero_error"] else ""
            )
            chi = (
                f" chi2/dof={row['reduced_chi_square']:.2f}"
                if row["reduced_chi_square"] is not None
                else ""
            )
            console.print(
                f"  p={row['p']:<8g} Lambda={row['lambda']:.3g} "
                f"[{row['lambda_low']:.3g}, {row['lambda_high']:.3g}] "
                f"d={','.join(str(d) for d in row['distances_used'])}{excluded}{chi}{regime}"
            )

    censored = summary["censored"]
    if censored:
        console.print(
            f"\n[yellow]{len(censored)} point(s) hit the shot ceiling before the error "
            "target; their intervals are the widest in the file:[/yellow]"
        )
        for point in censored:
            console.print(
                f"  {point['decoder']} d={point['distance']} p={point['p']:g} "
                f"shots={point['shots']:,} errors={point['errors']}"
            )
    console.print(f"\n[dim]{summary['reported_not_asserted']}[/dim]")


@app.command()
def validate(
    path: Annotated[Path, typer.Argument(help="Dataset file to validate.")],
    fmt: Annotated[str | None, typer.Option("--format", help="Override format.")] = None,
    qa: Annotated[bool, typer.Option(help="Also run slow statistical checks.")] = False,
) -> None:
    """Run fast structural validation, and optionally slow statistical QA.

    ``--qa`` routes through :func:`qecgen.run.qa_report` rather than calling the estimator
    here, so the browser and the terminal run the same checks in the same order. Without
    ``--qa`` this stays a direct structural read: there is no reason to pay for the
    analysis layer to answer a question the exporter and the validator already answer.
    """
    _require_existing_file(path)
    if qa:
        _validate_with_qa(path, fmt)
        return

    exporter = _cli_exporter(fmt) if fmt else _cli_exporter(_infer_format(path))
    dataset = _read_dataset(exporter, path)
    report = validate_dataset(dataset)

    for result in report.results:
        colour = "green" if result.passed else "red"
        console.print(f"[{colour}]{result}[/{colour}]")

    if not report.ok:
        console.print(f"[red]{len(report.failures)} check(s) FAILED[/red]")
        raise typer.Exit(code=1)

    console.print("[green]all structural checks passed[/green]")


def _validate_with_qa(path: Path, fmt: str | None) -> None:
    """Structural checks then statistics, through the shared analysis layer."""
    spec = runner.QaSpec(dataset=path, fmt=fmt)
    _resolved_config("validate --qa", spec)
    try:
        result = runner.analyse(spec)
    except NotAQecgenDatasetError as exc:
        raise typer.BadParameter(str(exc)) from None

    summary = result.summary
    for check in summary["checks"]:
        colour = "green" if check["passed"] else "red"
        detail = check["detail"]
        if not check["passed"] and check["requirement"]:
            detail = f"{detail} -- expected {check['requirement']}"
        verdict = "PASS" if check["passed"] else "FAIL"
        console.print(f"[{colour}][{verdict}] {check['name']}: {detail}[/{colour}]")

    if not summary["ok"]:
        # Structural failures are reported before minutes are spent on statistics that
        # would be meaningless against a malformed file. The domain enforces the order;
        # this only says so.
        console.print(
            f"[red]{sum(1 for c in summary['checks'] if not c['passed'])} check(s) FAILED[/red]"
        )
        console.print(f"[yellow]{summary['skipped']}[/yellow]")
        raise typer.Exit(code=1)

    console.print("[green]all structural checks passed[/green]")
    console.print("\n[bold]statistical QA (slow, opt-in)[/bold]")
    for env in summary["environments"]:
        console.print(
            f"  axis={env['axis']}={env['axis_value']:g}  "
            f"logical={env['logical_error_rate']:.5f} "
            f"[{env['ci_low']:.5f}, {env['ci_high']:.5f}] "
            f"({env['failures']}/{env['shots']}) det_rate={env['detection_event_rate']:.5f}"
        )
    console.print(
        "[dim]Reported as results, not asserted. The commonly quoted 0.5-1% threshold "
        "depends on channel convention, rounds, basis and decoder.[/dim]"
    )


@app.command()
def inspect(
    path: Annotated[Path, typer.Argument(help="Dataset file to inspect.")],
    fmt: Annotated[str | None, typer.Option("--format")] = None,
    show_text: Annotated[bool, typer.Option(help="Include circuit and DEM text.")] = False,
) -> None:
    """Print a dataset's manifest without loading any shots at all.

    Every registered format now has a cheap manifest reader, so this never materialises a
    file. It used to for npz, parquet and jsonl, purely to add an ``arrays`` row of array
    shapes — a row that was therefore present for exactly the three formats where it cost
    the most and absent for the two where it would have been nearly free. The manifest
    already carries ``shots``, ``n_detectors``, ``n_observables`` and ``n_mechanisms``,
    which is everything that row restated, and ``validate`` checks the real shapes
    against them.
    """
    _require_existing_file(path)
    resolved = fmt or _infer_format(path)
    try:
        meta = DatasetMeta.from_json_dict(read_manifest(path, resolved))
    except NotAQecgenDatasetError as exc:
        raise typer.BadParameter(str(exc)) from None

    table = Table(title=f"{path}", show_header=True)
    table.add_column("field", style="cyan", no_wrap=True)
    table.add_column("value", style="white")
    payload = meta.to_json_dict()
    environments = payload.pop("environments")
    for key, value in payload.items():
        table.add_row(key, json.dumps(value) if isinstance(value, dict) else str(value))
    table.add_row("n_environments", str(len(environments)))
    console.print(table)

    env_table = Table(title="environments", show_header=True)
    for column in ("id", "axis", "axis_value", "p", "noise_model", "shots", "channels"):
        env_table.add_column(column)
    for env in meta.environments:
        env_table.add_row(
            str(env.environment_id),
            env.axis,
            f"{env.axis_value:g}",
            f"{env.p:g}",
            str(env.noise_model),
            f"{env.shots:,}",
            json.dumps(env.channels.as_dict()),
        )
    console.print(env_table)

    if show_text:
        # Circuit/DEM text never travels in the manifest (the FROZEN_PRIOR firewall),
        # so it must be read from the format's provenance block. The old check on
        # `meta.environments[0].circuit` could never fire: no read path populates the
        # spec's text fields, so --show-text printed nothing, silently, forever.
        provenance = read_provenance(path, resolved)
        if provenance is None:
            console.print(
                f"\n[yellow]no provenance stored[/yellow] "
                f"(structure_level={meta.structure_level}). Circuit and DEM text are "
                "written only at --structure full, and only by formats with a "
                f"provenance block ({', '.join(provenance_formats())})."
            )
        else:
            for env_payload in provenance.get("environments", []):
                env_id = env_payload.get("environment_id")
                console.print(f"\n[bold]circuit (environment {env_id})[/bold]")
                console.print(str(env_payload.get("circuit", "")))
                console.print(f"\n[bold]dem (environment {env_id})[/bold]")
                console.print(str(env_payload.get("dem", "")))


@app.command(
    help=(
        "Serve the web UI on localhost: every command this tool has, in a browser. "
        "Loopback only, and not configurable: the API writes files and spawns "
        "processes for anyone who can reach it, with no authentication, so the only "
        "safe audience is the person at this machine. A non-loopback --host is "
        "refused rather than quietly accepted."
    )
)
def ui(
    data_root: Annotated[
        Path, typer.Option(help="The only directory the UI may write to or browse.")
    ] = Path("data"),
    host: Annotated[
        str, typer.Option(help="Bind address. Loopback only, by design.")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to serve on.")] = DEFAULT_PORT,
    jobs: Annotated[int, typer.Option(help="Runs executed at once.")] = 1,
    dev: Annotated[bool, typer.Option(help="Allow the Vite dev server origin.")] = False,
    open_browser: Annotated[bool, typer.Option("--open/--no-open")] = True,
) -> None:
    """Serve the web UI. The user-facing help lives in the ``help=`` argument: this
    docstring is developer documentation, and typer renders raw docstring markup
    (``double backticks``, **bold**) literally into the terminal."""
    import uvicorn

    from qecgen.ui.app import BUILD_HINT, create_app, static_is_built
    from qecgen.ui.settings import LOOPBACK_HOSTS, WebSettings

    if host not in LOOPBACK_HOSTS:
        raise typer.BadParameter(
            f"refusing to bind {host!r}. This API writes files and starts processes on "
            "request and has no authentication, so it is loopback-only. Use an SSH "
            f"tunnel if you need it from elsewhere. Allowed: {', '.join(sorted(LOOPBACK_HOSTS))}"
        )

    settings = WebSettings.create(
        data_root, host=host, port=port, max_concurrent_jobs=jobs, dev=dev
    )
    url = f"http://{host}:{port}"
    _print_config(
        "ui",
        {
            "url": url,
            "data_root": settings.data_root,
            "runs_dir": settings.runs_dir,
            "concurrent_jobs": jobs,
            "frontend": "built" if static_is_built() else f"NOT BUILT — run: {BUILD_HINT}",
            "dev_origin_allowed": "http://localhost:5173 (Vite)" if dev else "no",
        },
    )
    if not static_is_built():
        console.print(
            f"[yellow]The frontend is not built.[/yellow] The API works and is documented "
            f"at {url}/api/docs, but the pages will return 503 until you run:\n"
            f"    [cyan]{BUILD_HINT}[/cyan]"
        )
    elif open_browser:
        _open_browser_when_listening(host, port, url)

    uvicorn.run(create_app(settings), host=host, port=port, log_level="warning")


def _open_browser_when_listening(host: str, port: int, url: str) -> None:
    """Open the browser once the server actually accepts connections.

    ``webbrowser.open`` before ``uvicorn.run`` races the bind: the first page load can
    land on connection-refused and the user sees a dead tab for a working server. A
    daemon thread polls the socket and opens the page only when a connect succeeds; it
    gives up quietly after 15 seconds, because a server that failed to start has
    already printed its own error and a browser window would only bury it.
    """
    import socket
    import threading
    import time
    import webbrowser

    def _poll_then_open() -> None:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=0.25):
                    pass
            except OSError:
                time.sleep(0.1)
                continue
            webbrowser.open(url)
            return

    threading.Thread(target=_poll_then_open, name="qecgen-ui-open-browser", daemon=True).start()


@app.command()
def formats() -> None:
    """List the registered export formats."""
    table = Table(title="registered exporters", show_header=True)
    table.add_column("format", style="cyan")
    table.add_column("extension")
    table.add_column("streaming")
    table.add_column("round-trips structure")
    table.add_column("carries provenance")
    for name, exporter in sorted(EXPORTERS.items()):
        # Read the protocol property rather than hardcoding a name set: a duplicated
        # source of truth here would start lying the moment a format gains structure
        # round-tripping.
        structure_ok = "yes" if exporter.structure_round_trip else "no (arrays + manifest only)"
        # Read from the protocol property, like the other two columns. This one decides
        # whether `inspect --show-text` has anything to show, and it used to be restated
        # by hand in a second table in this file.
        provenance_ok = (
            "yes (circuit + DEM text at --structure full)"
            if exporter.carries_provenance
            else "no (--structure full records dem)"
        )
        table.add_row(
            name,
            exporter.extension,
            "yes" if exporter.streaming else "no",
            structure_ok,
            provenance_ok,
        )
    console.print(table)
    console.print(
        "\n[dim]No Nexus exporter is registered. The Nexus input format is not yet known; "
        "adding it is one module plus one line in qecgen/exporters/__init__.py.[/dim]"
    )


@app.command(
    help=(
        "Score a proposed physical Pauli correction by its logical effect: apply the "
        "correction, check the logical qubit matches. The correction is an input, not "
        "a target, so this is not Contract C and no ground-truth fault label is needed "
        "(see DATA_CONTRACT.md). The logical operators are rebuilt from a noiseless "
        "circuit generated from the dataset's manifest parameters, so scoring a "
        "frozen_prior test file never reads that file's own error model."
    )
)
def score(
    path: Annotated[Path, typer.Argument(help="Dataset file holding the true observables.")],
    correction: Annotated[
        Path, typer.Option(help="NPZ with correction_x and correction_z arrays.")
    ],
    fmt: Annotated[str | None, typer.Option("--format", help="Override dataset format.")] = None,
    unpacked: Annotated[
        bool, typer.Option(help="Correction arrays are bool (shots, n_data_qubits).")
    ] = False,
    alpha: Annotated[float, typer.Option(help="1 - confidence level.")] = 0.05,
) -> None:
    """Score a supplied correction. The user-facing help lives in ``help=`` — typer
    renders docstring markup literally, so this stays developer documentation.

    The orchestration itself is :func:`qecgen.run.score_correction`, not here. The
    noiseless-rebuild rule is the property that makes scoring a ``frozen_prior`` file
    safe, and a copy of it in a second front end could drift to ``p=meta.p`` and still
    produce numbers that looked reasonable.
    """
    _require_existing_file(path)
    _require_existing_file(correction)
    spec = runner.ScoreSpec(
        dataset=path, correction=correction, fmt=fmt, unpacked=unpacked, alpha=alpha
    )
    _resolved_config("score", spec)

    try:
        result = runner.analyse(spec)
    except NotAQecgenDatasetError as exc:
        raise typer.BadParameter(str(exc)) from None
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None

    summary = result.summary
    console.print(
        f"\n[bold]logical error rate under the supplied correction[/bold]\n"
        f"  {summary['logical_error_rate']:.5f} "
        f"[{summary['ci_low']:.5f}, {summary['ci_high']:.5f}] "
        f"({summary['failures']}/{summary['shots']})  "
        f"n_data={summary['n_data_qubits']} n_obs={summary['n_observables']} "
        f"schema={str(summary['schema_digest'])[:12]}"
    )
    console.print(
        "\n[dim]A shot succeeds when the correction's induced observable flip equals the "
        "true flip on every observable. The schema digest identifies the qubit ordering "
        "the score was computed under; the same array under a different ordering gives a "
        "different, equally plausible number.[/dim]"
    )


def _read_dataset(exporter: Exporter, path: Path) -> Any:
    """Read a dataset, restating "this file is not ours" as an argument error.

    ``.csv`` is both a dataset extension and the extension ``qecgen sweep`` writes its
    results table with, so pointing ``validate`` at a sweep table is an ordinary mistake
    this tool itself makes easy to reach. A traceback for it would be the inconsistency
    :func:`_cli_exporter` and :func:`_infer_format` already exist to avoid.
    """
    try:
        return exporter.read(path)
    except NotAQecgenDatasetError as exc:
        raise typer.BadParameter(str(exc)) from None


def _infer_format(path: Path) -> str:
    """Registry inference, restated as a CLI argument error."""
    try:
        return infer_format(path)
    except ValueError as exc:
        raise typer.BadParameter(f"{exc}; pass --format") from None


def _parse_range(text: str) -> tuple[float, float, int]:
    """Parse ``low:high:count``, restating the domain's refusal as a parameter error.

    The rules themselves live in :func:`qecgen.run.parse_range` so the web UI expands a
    range through the same code. A count below 1 silently coercing to a single rate at
    ``low`` is the refusal that matters: it runs a different sweep than the flag
    describes, and a second front end re-deriving the parse could reintroduce it.
    """
    try:
        return runner.parse_range(text)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None


def _report_written(path: Path, shots: int, content_hash: str | None) -> None:
    console.print(
        f"[green]wrote[/green] {path}  shots={shots:,}  "
        f"content_hash={(content_hash or 'none')[:16]}..."
    )


if __name__ == "__main__":  # pragma: no cover
    app()
