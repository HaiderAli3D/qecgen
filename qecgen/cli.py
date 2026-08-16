"""Command line interface.

Every command prints its fully resolved configuration before doing any work, so a
terminal log is a complete record of what was run — including the values that were
defaulted rather than passed.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

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
)
from qecgen.sampling import DEFAULT_CHUNK_SIZE
from qecgen.ui import DEFAULT_PORT
from qecgen.validate import validate_dataset

if TYPE_CHECKING:
    # Imported for annotations only. `qecgen.sweep` pulls in matplotlib and sinter, and
    # the sweep command imports it lazily so that `qecgen generate` does not pay for them.
    from qecgen.qa import SuppressionFit
    from qecgen.sweep import SweepPoint

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


def _resolved_config(command: str, spec: runner.RunSpec) -> None:
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
    """Run a sinter-driven threshold sweep and emit CSV, a plot and a threshold summary."""
    from qecgen.decoders import DEFAULT_DECODERS, resolve_decoders
    from qecgen.sweep import (
        censored_points,
        crossing_from_sweep,
        plot_threshold,
        run_sweep,
        suppression_from_sweep,
        write_csv,
        write_threshold_json,
    )

    ds = list(distances) if distances else [3, 5, 7]
    low, high, count = _parse_range(p_range)
    rates = runner.linear_rates(low, high, count)
    requested = list(decoder) if decoder else list(DEFAULT_DECODERS)

    _print_config(
        "sweep",
        {
            "distances": ds,
            "p_range": f"{low} .. {high} in {count} steps",
            "rates": [f"{r:.4g}" for r in rates],
            "max_errors": max_errors,
            "max_shots": f"{max_shots:,}",
            "workers": workers,
            "noise_model": noise,
            "rounds": "per task: distance (the memory-experiment default)",
            "basis": basis,
            "rotated": True,
            "decoders": ", ".join(requested),
            # The sweep decoder and the QA oracle are different things, and the natural
            # wrong instinct once --decoder exists is to make the oracle configurable too.
            # An oracle that can be set to a decoder under test is not an oracle.
            "qa_oracle": "pymatching (qa.py only; --decoder does not change it)",
            "dem_seen_by_decoders": "decomposed (sinter derives it with decompose_errors=True)",
            "out_csv": out,
            "out_plot": out.with_suffix(".png"),
            "note": "sinter timing is throughput, NOT decoder latency",
        },
    )

    if out.suffix.lower() == ".png":
        raise typer.BadParameter(
            f"--out {out} would make the CSV and the plot the same path, so the plot "
            "would overwrite the data. Pass a .csv path; the plot is written beside it."
        )

    try:
        decoder_names = resolve_decoders(requested)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None

    points = run_sweep(
        distances=ds,
        error_rates=rates,
        max_errors=max_errors,
        max_shots=max_shots,
        workers=workers,
        decoders=decoder_names,
        noise_model=noise,
        basis=basis,
    )
    plot = out.with_suffix(".png")
    summary = out.with_suffix(".threshold.json")
    # Staged like every dataset write: the three sweep outputs truncate on open, so an
    # interrupted write would otherwise destroy the previous sweep's results and could
    # leave a cleanly-parsing truncated CSV.
    with runner.staged(out.parent) as staging:
        write_csv(points, staging.scratch / out.name)
        plot_threshold(points, staging.scratch / plot.name)
        write_threshold_json(
            points,
            staging.scratch / summary.name,
            max_errors=max_errors,
            max_shots=max_shots,
            noise_model=str(noise),
            basis=str(basis),
        )
    console.print(f"[green]wrote[/green] {out}, {plot} and {summary}")

    _report_threshold(
        crossings=crossing_from_sweep(points),
        fits=suppression_from_sweep(points),
        censored=censored_points(points, max_errors, max_shots),
    )


def _report_threshold(
    crossings: dict[str, float | None],
    fits: dict[tuple[str, float], SuppressionFit],
    censored: list[SweepPoint],
) -> None:
    """Print the crossing and suppression summary, as results rather than verdicts."""
    for decoder, crossing in sorted(crossings.items()):
        location = "not visible in the sampled range" if crossing is None else f"p ~ {crossing:g}"
        console.print(f"\n[bold]{decoder}[/bold]  crossing: {location}")
        rows = sorted((p, fit) for (d, p), fit in fits.items() if d == decoder)
        if not rows:
            console.print("  [dim]no suppression fit: fewer than two distances with errors[/dim]")
            continue
        for p, fit in rows:
            # Lambda only means anything below threshold, so say which side each row is on
            # rather than letting a super-threshold number read like a suppression result.
            regime = (
                ""
                if crossing is None or p < crossing
                else "  [yellow](at or above crossing)[/yellow]"
            )
            console.print(f"  p={p:<8g} {fit}{regime}")

    if censored:
        console.print(
            f"\n[yellow]{len(censored)} point(s) hit the shot ceiling before the error "
            "target; their intervals are the widest in the file:[/yellow]"
        )
        for point in censored:
            console.print(
                f"  {point.decoder} d={point.distance} p={point.p:g} "
                f"shots={point.shots:,} errors={point.errors}"
            )

    console.print(
        "\n[dim]Reported as results, never asserted. Crossing and Lambda both depend on "
        "the channel convention, rounds, basis and decoder.[/dim]"
    )


@app.command()
def validate(
    path: Annotated[Path, typer.Argument(help="Dataset file to validate.")],
    fmt: Annotated[str | None, typer.Option("--format", help="Override format.")] = None,
    qa: Annotated[bool, typer.Option(help="Also run slow statistical checks.")] = False,
) -> None:
    """Run fast structural validation, and optionally slow statistical QA."""
    exporter = _cli_exporter(fmt) if fmt else _cli_exporter(_infer_format(path))
    dataset = _read_dataset(exporter, path)
    report = validate_dataset(dataset)

    for result in report.results:
        colour = "green" if result.passed else "red"
        console.print(f"[{colour}]{result}[/{colour}]")

    if not report.ok:
        # Report structural failures before spending minutes on statistics that are
        # meaningless if the file is malformed.
        console.print(f"[red]{len(report.failures)} check(s) FAILED[/red]")
        if qa:
            console.print("[yellow]skipping --qa: structural checks failed first[/yellow]")
        raise typer.Exit(code=1)

    console.print("[green]all structural checks passed[/green]")
    if qa:
        _run_qa(dataset)


def _run_qa(dataset: Any) -> None:
    """Print the opt-in statistical checks; the rebuild rule itself lives in qa.py.

    Argument resolution and presentation are all a front end owns. The
    rebuild-from-recorded-axis logic moved to :func:`qecgen.qa.estimate_environment_rates`
    so a second front end wanting QA cannot re-derive it differently.
    """
    from qecgen.qa import estimate_environment_rates

    console.print("\n[bold]statistical QA (slow, opt-in)[/bold]")
    for env, estimate in estimate_environment_rates(dataset.meta):
        console.print(f"  axis={env.axis}={env.axis_value:g}  {estimate}")
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
    """Print a dataset's manifest without loading more than necessary."""
    resolved = fmt or _infer_format(path)
    cheap = _read_manifest_only(path, resolved)
    if cheap is not None:
        meta = DatasetMeta.from_json_dict(cheap)
        dataset = None
    else:
        dataset = _read_dataset(_cli_exporter(resolved), path)
        meta = dataset.meta

    table = Table(title=f"{path}", show_header=True)
    table.add_column("field", style="cyan", no_wrap=True)
    table.add_column("value", style="white")
    payload = meta.to_json_dict()
    environments = payload.pop("environments")
    for key, value in payload.items():
        table.add_row(key, json.dumps(value) if isinstance(value, dict) else str(value))
    table.add_row("n_environments", str(len(environments)))
    if dataset is not None:
        table.add_row("arrays", _describe_arrays(dataset))
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
        provenance = _read_provenance(path, resolved)
        if provenance is None:
            console.print(
                f"\n[yellow]no provenance stored[/yellow] "
                f"(structure_level={meta.structure_level}). Circuit and DEM text are "
                "written only at --structure full, and only by formats with a "
                f"provenance block ({', '.join(sorted(_PROVENANCE_READERS))})."
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
        "Serve the web UI for generate, multi-env and drift on localhost. "
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
    for name, exporter in sorted(EXPORTERS.items()):
        # Read the protocol property rather than hardcoding a name set: a duplicated
        # source of truth here would start lying the moment a format gains structure
        # round-tripping.
        structure_ok = "yes" if exporter.structure_round_trip else "no (arrays + manifest only)"
        table.add_row(name, exporter.extension, "yes" if exporter.streaming else "no", structure_ok)
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

    Rebuilding operators from a **noiseless** circuit is sound because the correction
    schema is a property of the code rather than of the noise — asserted by
    ``test_schema_digest_is_stable_and_noise_independent``.
    """
    import numpy as np

    from qecgen.circuits import build_circuit
    from qecgen.correction import (
        estimate_correction_logical_error_rate,
        extract_logical_operators,
        pack_correction,
    )

    exporter = _cli_exporter(fmt) if fmt else _cli_exporter(_infer_format(path))
    dataset = _read_dataset(exporter, path)
    meta = dataset.meta

    _print_config(
        "score",
        {
            "dataset": path,
            "correction": correction,
            "distance": meta.distance,
            "rounds": meta.rounds,
            "basis": meta.basis,
            "rotated": meta.rotated,
            "shots": f"{dataset.n_shots:,}",
            "drift_condition": meta.drift_condition,
            "operators_from": "noiseless circuit rebuilt from manifest parameters",
            "note": "scores a supplied correction; this is not Contract C",
        },
    )

    circuit, _ = build_circuit(
        meta.distance, 0.0, rounds=meta.rounds, basis=meta.basis, rotated=meta.rotated
    )
    operators = extract_logical_operators(circuit, strict_single_basis=True)

    with np.load(correction) as payload:
        missing = [key for key in ("correction_x", "correction_z") if key not in payload]
        if missing:
            raise typer.BadParameter(
                f"{correction} is missing {missing}; it must hold correction_x and "
                "correction_z arrays over the data qubits."
            )
        corr_x = np.asarray(payload["correction_x"])
        corr_z = np.asarray(payload["correction_z"])

    if unpacked:
        corr_x = pack_correction(corr_x)
        corr_z = pack_correction(corr_z)

    try:
        estimate = estimate_correction_logical_error_rate(
            corr_x, corr_z, dataset.observables, operators, alpha=alpha
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None

    console.print(f"\n[bold]logical error rate under the supplied correction[/bold]\n  {estimate}")
    console.print(
        "\n[dim]A shot succeeds when the correction's induced observable flip equals the "
        "true flip on every observable. The schema digest identifies the qubit ordering "
        "the score was computed under; the same array under a different ordering gives a "
        "different, equally plausible number.[/dim]"
    )


def _describe_arrays(dataset: Any) -> str:
    parts = [f"detectors{dataset.detectors.shape}", f"observables{dataset.observables.shape}"]
    if dataset.environment_ids is not None:
        parts.append(f"environment_ids{dataset.environment_ids.shape}")
    if dataset.mechanisms is not None:
        parts.append(f"mechanisms{dataset.mechanisms.shape}")
    if dataset.structure is not None:
        parts.append(f"dem(H{dataset.structure.h.shape})")
    return ", ".join(parts)


def _provenance_hdf5(path: Path) -> dict[str, Any] | None:
    import h5py

    with h5py.File(path, "r") as handle:
        if "provenance" not in handle:
            return None
        return dict(json.loads(str(handle["provenance"].attrs["environments"])))


def _provenance_npz(path: Path) -> dict[str, Any] | None:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        if "provenance" not in archive.files:
            return None
        return dict(json.loads(str(archive["provenance"].item())))


def _provenance_csv(path: Path) -> dict[str, Any] | None:
    from qecgen.exporters.csv_table import read_provenance_only

    payload = read_provenance_only(path)
    return None if payload is None else dict(payload)


_PROVENANCE_READERS: dict[str, Callable[[Path], dict[str, Any] | None]] = {
    "hdf5": _provenance_hdf5,
    "npz": _provenance_npz,
    "csv": _provenance_csv,
}
"""Formats that carry a provenance block, keyed by format name.

A dict rather than a chain of ``if resolved == ...`` so the "no provenance stored"
message can name the formats from the same source that decides whether to look. The
hardcoded ``(hdf5, npz)`` in that message was a second source of truth one edit away
from lying, and CSV is the edit that would have made it lie.
"""


def _read_provenance(path: Path, resolved: str) -> dict[str, Any] | None:
    """Read the provenance block from formats that carry one, or None.

    Kept read-only and separate from the manifest readers on purpose: provenance is
    not decoder-visible, and folding it into a manifest loader would put a frozen-prior
    test file's own DEM one attribute away from every decoder that reads manifests.
    """
    reader = _PROVENANCE_READERS.get(resolved)
    return None if reader is None else reader(path)


def _read_manifest_only(path: Path, resolved: str) -> dict[str, object] | None:
    """The manifest alone, for formats that can produce it without reading the shots.

    Materialising a million-shot file just to print its settings defeats the purpose of
    ``inspect``, and CSV is the worst case of all — text, one column per bit.

    ``cli.py`` cannot reach for ``qecgen.ui.datasets.read_manifest``: one front end
    importing another is a dependency this tree does not have, and the UI layer pulls in
    a web stack that ``qecgen inspect`` must run without.
    """
    if resolved == "hdf5":
        from qecgen.exporters.hdf5 import read_manifest_only

        return read_manifest_only(path)
    if resolved == "csv":
        from qecgen.exporters.csv_table import read_manifest_only as read_csv_manifest

        return read_csv_manifest(path)
    return None


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
