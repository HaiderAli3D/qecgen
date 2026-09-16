"""``python -m qecgen.residual.cli``: the residual pipeline's module CLI.

Run from the repository root with ``python -m`` — an editable ``qecgen`` install can
resolve to a different checkout while looking identical in the terminal (``CLAUDE.md``,
"Verify a change with python -m qecgen.cli"), and this module lives beside the code it
drives for the same reason. Every command prints the fully resolved configuration before
doing work so a terminal log is a complete record of the run; ``build`` prints the
``pm_wrong`` counts the brief asks for. No domain logic lives here: each command loads a
config, calls :mod:`qecgen.residual.pipeline`, and turns its refusals into exit code 1
with the message intact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from qecgen import __version__
from qecgen.residual import pipeline
from qecgen.residual.checkpoint import CheckpointError
from qecgen.residual.config import ConfigError, ResidualConfig, load_config, resolved_dict
from qecgen.residual.validation import validate_dataset_dir
from qecgen.residual.writers import write_json

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Build, validate and summarise PyMatching residual-error datasets.",
)
console = Console()

_REFUSALS: tuple[type[Exception], ...] = (
    pipeline.SourceBlockedError,
    pipeline.PilotGateError,
    pipeline.ValidationFailedError,
    CheckpointError,
    ConfigError,
    FileNotFoundError,
    ValueError,
)
"""Errors the pipeline raises on purpose; printed verbatim and exited with status 1.
Anything else is a bug and keeps its traceback."""


def _log(message: str) -> None:
    """Progress lines go through rich with markup off: a Clopper-Pearson interval prints as
    ``[0.1, 0.2]``, which rich would otherwise read as a style tag."""
    console.print(message, markup=False, highlight=False)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for key in sorted(payload):
        value = payload[key]
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = str(value)
    return flat


def _print_config(command: str, config: ResidualConfig) -> None:
    """Print the fully resolved configuration as a table (the ``qecgen.cli`` convention)."""
    table = Table(title=f"qecgen residual {command}  (v{__version__})", show_header=True)
    table.add_column("setting", style="cyan", no_wrap=True)
    table.add_column("value", style="white")
    for key, value in _flatten(resolved_dict(config)).items():
        table.add_row(key, value)
    console.print(table)


def _load(config_path: Path, repo_root: Path | None) -> ResidualConfig:
    return load_config(config_path, repo_root or _repo_root())


def _refuse(error: Exception) -> None:
    label = "blocked" if isinstance(error, pipeline.SourceBlockedError) else "refused"
    _log(f"{label}: {type(error).__name__}: {error}")
    raise typer.Exit(code=1)


ConfigOption = Annotated[
    Path, typer.Option("--config", exists=True, dir_okay=False, help="Reviewed residual config.")
]
RepoRootOption = Annotated[
    Path | None,
    typer.Option(
        "--repo-root",
        help="Checkout that relative config paths resolve under (default: this checkout).",
    ),
]


@app.command()
def inventory(config: ConfigOption, repo_root: RepoRootOption = None) -> None:
    """Resolve the source and decoder, decode nothing, write inventory.json."""
    cfg = _load(config, repo_root)
    _print_config("inventory", cfg)
    try:
        record = pipeline.inventory(cfg, log=_log)
    except _REFUSALS as error:
        _refuse(error)
        return
    _log(f"wrote {pipeline.checkpoint_root(cfg) / pipeline.INVENTORY_FILENAME}")
    _log(f"config_hash {record['config_hash']}")


@app.command()
def pilot(
    config: ConfigOption,
    repo_root: RepoRootOption = None,
    generated_rows: Annotated[
        int, typer.Option("--generated-rows", min=1, help="Fresh rows to time the stages on.")
    ] = 10_000,
    resample_rows: Annotated[
        int, typer.Option("--resample-rows", min=1, help="Rows in the re-sample CI check.")
    ] = 16_000,
    edge_prototype_rows: Annotated[
        int,
        typer.Option(
            "--edge-prototype-rows",
            min=0,
            help="Rows for the optional decode_to_edges_array timing (0 = skip).",
        ),
    ] = 0,
) -> None:
    """Decode the existing rows, run the alignment investigation, project the full run."""
    cfg = _load(config, repo_root)
    _print_config("pilot", cfg)
    try:
        record = pipeline.pilot(
            cfg,
            generated_rows=generated_rows,
            resample_rows=resample_rows,
            edge_prototype_rows=edge_prototype_rows,
            log=_log,
        )
    except _REFUSALS as error:
        _refuse(error)
        return
    _log(f"wrote {pipeline.checkpoint_root(cfg) / pipeline.PILOT_FILENAME}")
    _log(f"resources_sufficient: {record['resources_sufficient']}")


@app.command()
def build(
    config: ConfigOption,
    repo_root: RepoRootOption = None,
    fresh: Annotated[
        bool, typer.Option("--fresh", help="Discard this dataset's checkpoints first.")
    ] = False,
    skip_pilot_gate: Annotated[
        bool,
        typer.Option("--skip-pilot-gate", help="Do not require a pilot record (tests only)."),
    ] = False,
    pilot_rows: Annotated[
        int, typer.Option("--pilot-rows", min=1, help="Rows for an inline pilot, if one runs.")
    ] = 10_000,
) -> None:
    """Build and publish one dataset atomically, resuming from verified checkpoints."""
    cfg = _load(config, repo_root)
    _print_config("build", cfg)
    try:
        pipeline.build(
            cfg, fresh=fresh, skip_pilot_gate=skip_pilot_gate, pilot_rows=pilot_rows, log=_log
        )
    except _REFUSALS as error:
        _refuse(error)


@app.command()
def validate(
    output: Annotated[
        Path,
        typer.Option("--output", exists=True, file_okay=False, help="Published dataset directory."),
    ],
    spot_rows: Annotated[
        int, typer.Option("--spot-rows", min=1, help="Rows to re-decode and re-feature.")
    ] = 5,
    rebuild_decoder: Annotated[
        bool,
        typer.Option(
            "--rebuild-decoder/--no-rebuild-decoder",
            help="Also rebuild the decoder from the resolved configuration.",
        ),
    ] = True,
    report_path: Annotated[
        Path | None,
        typer.Option("--report", dir_okay=False, help="Also write the report as JSON here."),
    ] = None,
) -> None:
    """Run the fifteen named checks (the brief's fourteen assertions) plus spot checks."""
    try:
        report = validate_dataset_dir(output, rebuild_decoder=rebuild_decoder, spot_rows=spot_rows)
    except _REFUSALS as error:
        _refuse(error)
        return
    table = Table(title=f"qecgen residual validate {report.dataset_name}", show_header=True)
    table.add_column("check", style="cyan", no_wrap=True)
    table.add_column("passed")
    table.add_column("detail", style="white")
    for check in report.checks:
        table.add_row(check.name, "yes" if check.passed else "NO", check.detail)
    console.print(table)
    if report_path is not None:
        write_json(report_path, report.to_dict())
        _log(f"wrote {report_path}")
    _log(f"ok: {report.ok} ({len(report.spot_checks)} spot rows)")
    if not report.ok:
        raise typer.Exit(code=1)


@app.command(name="build-all")
def build_all(
    config_dir: Annotated[
        Path,
        typer.Option("--config-dir", exists=True, file_okay=False, help="Directory of configs."),
    ],
    repo_root: RepoRootOption = None,
    fresh: Annotated[bool, typer.Option("--fresh", help="Discard every checkpoint first.")] = False,
    skip_pilot_gate: Annotated[
        bool,
        typer.Option("--skip-pilot-gate", help="Do not require pilot records (tests only)."),
    ] = False,
    pilot_rows: Annotated[
        int, typer.Option("--pilot-rows", min=1, help="Rows for inline pilots, if any run.")
    ] = 10_000,
) -> None:
    """Build every config: required datasets first, additional last; blocked ones recorded."""
    try:
        results = pipeline.build_all(
            config_dir,
            repo_root=repo_root or _repo_root(),
            fresh=fresh,
            skip_pilot_gate=skip_pilot_gate,
            pilot_rows=pilot_rows,
            log=_log,
        )
    except _REFUSALS as error:
        # An empty config directory or one malformed config refuses the whole batch
        # before any dataset is attempted; that is a refusal, not a bug with a traceback.
        _refuse(error)
        return
    _log(pipeline.as_json(results))
    if any(r["status"] != "completed" for r in results):
        raise typer.Exit(code=2)


@app.command()
def manifest(
    output_root: Annotated[
        Path,
        typer.Option("--output-root", exists=True, file_okay=False, help="Residual output root."),
    ],
) -> None:
    """Re-render MANIFEST.md from the published summaries and blocked records."""
    try:
        path = pipeline.write_manifest(output_root)
    except _REFUSALS as error:
        _refuse(error)
        return
    _log(f"wrote {path}")


if __name__ == "__main__":
    app()
