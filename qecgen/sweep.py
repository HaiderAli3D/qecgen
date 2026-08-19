"""Sinter-driven parameter sweeps and threshold plots.

Stopping is governed by ``max_errors`` (default 500 logical failures) with ``max_shots``
as a ceiling. A fixed shot count either wastes days at low ``p`` or returns numbers with
no statistical content.

**Sinter's parallel collection timing is a throughput measure, not a decoder latency
measurement.** Latency benchmarking belongs in the separate benchmark repository and is
out of scope here.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import sinter
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from qecgen.circuits import Basis, NoiseModel, build_circuit, default_rounds
from qecgen.decoders import resolve_decoders
from qecgen.qa import (
    Interval,
    SuppressionFit,
    clopper_pearson,
    crossing_from_intervals,
    fit_suppression,
)

__all__ = [
    "SweepPoint",
    "SweepProgress",
    "build_tasks",
    "censored_points",
    "crossing_from_sweep",
    "plot_threshold",
    "run_sweep",
    "suppression_from_sweep",
    "threshold_summary",
    "write_csv",
    "write_threshold_json",
]


@dataclass(frozen=True, slots=True)
class SweepPoint:
    """One collected (decoder, distance, p) result."""

    decoder: str
    distance: int
    p: float
    rounds: int
    noise_model: str
    basis: str
    shots: int
    errors: int
    discards: int

    @property
    def rate(self) -> float:
        """Point estimate of the logical error rate."""
        return self.errors / self.shots if self.shots else 0.0


@dataclass(frozen=True, slots=True)
class SweepProgress:
    """One progress report from a collection that is still running.

    Plain types on purpose. :mod:`qecgen.run` drives the sweep and hands these to a front
    end, and neither should have to import sinter to read a progress update — the same
    boundary :mod:`qecgen.decoders` keeps by resolving decoder *names*. Translating
    ``sinter.Progress`` here is what keeps that import confined to this module.

    ``completed_tasks`` is derived rather than reported: sinter exposes no per-task "done"
    signal to a progress callback, so it is counted by mirroring the rule
    ``_ManagedTaskState.is_completed`` applies — a task stops when its accumulated shots
    reach ``max_shots`` or its accumulated errors reach ``max_errors``. That is the same
    stopping rule :func:`censored_points` already encodes, and sinter is exact-pinned so it
    cannot drift without a deliberate bump. If it ever does go stale the cost is a progress
    bar that reads slightly wrong; no collected number depends on it.
    """

    completed_tasks: int
    total_tasks: int
    shots_collected: int
    errors_seen: int
    status_message: str
    """sinter's own free-form line: tasks remaining and an ETA for each. Passed through
    verbatim rather than reworded, because it is the only estimate of time to completion
    anything here has."""


def build_tasks(
    distances: list[int],
    error_rates: list[float],
    noise_model: NoiseModel = NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL,
    rounds: int | None = None,
    basis: Basis = Basis.Z,
    rotated: bool = True,
    condition: str = "not_applicable",
) -> list[sinter.Task]:
    """Build one :class:`sinter.Task` per (distance, p), tagged with metadata.

    ``json_metadata`` carries distance, p, noise model, basis and the environment
    condition so the collected CSV is self-describing.
    """
    tasks: list[sinter.Task] = []
    for distance in distances:
        for p in error_rates:
            circuit, channels = build_circuit(distance, p, noise_model, rounds, basis, rotated)
            tasks.append(
                sinter.Task(
                    circuit=circuit,
                    json_metadata={
                        "d": distance,
                        "p": p,
                        "rounds": default_rounds(noise_model, distance, rounds),
                        "noise_model": str(noise_model),
                        "basis": str(basis),
                        "rotated": rotated,
                        "condition": condition,
                        "channels": channels.as_dict(),
                    },
                )
            )
    return tasks


def _first_line(message: Any) -> str:
    """sinter's status message, reduced to something a single cell can show.

    It is multi-line and ANSI-wrapped; both front ends render it in one cell, so the tail
    would either be clipped mid-escape or blow out the row height.
    """
    lines = str(message or "").splitlines()
    return lines[0].strip() if lines else ""


def _progress_adapter(
    callback: Callable[[SweepProgress], None],
    total_tasks: int,
    max_shots: int,
    max_errors: int,
) -> Callable[[Any], None]:
    """Wrap a :class:`SweepProgress` consumer as a ``sinter.Progress`` consumer.

    Accumulates per ``strong_id`` because sinter reports *new* statistics each time, not
    running totals: a task is sampled in batches and appears in ``new_stats`` repeatedly,
    so counting reports rather than summing them would say a 30-batch task was 30 tasks.

    Takes ``Any`` rather than ``sinter.Progress``: sinter ships no ``py.typed``, so the
    annotation would be ``Any`` in effect anyway, and naming the type here would put a
    sinter symbol in the signature of something :mod:`qecgen.run` calls.
    """
    collected: dict[Any, tuple[int, int]] = {}

    def on_progress(progress: Any) -> None:
        for stat in progress.new_stats:
            shots, errors = collected.get(stat.strong_id, (0, 0))
            collected[stat.strong_id] = (shots + int(stat.shots), errors + int(stat.errors))
        completed = sum(
            1 for shots, errors in collected.values() if shots >= max_shots or errors >= max_errors
        )
        callback(
            SweepProgress(
                # Clamped because a task can overshoot its own stopping condition by the
                # last batch, and a bar that reads 9/8 tasks looks like a bug in the tool
                # rather than the rounding it is.
                completed_tasks=min(completed, total_tasks),
                total_tasks=total_tasks,
                shots_collected=sum(shots for shots, _ in collected.values()),
                errors_seen=sum(errors for _, errors in collected.values()),
                # First line only: sinter's status is multi-line and both front ends
                # render it in a single cell.
                status_message=_first_line(progress.status_message),
            )
        )

    return on_progress


def run_sweep(
    distances: list[int],
    error_rates: list[float],
    max_errors: int = 500,
    max_shots: int = 100_000_000,
    workers: int = 4,
    decoders: tuple[str, ...] = ("pymatching",),
    noise_model: NoiseModel = NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL,
    rounds: int | None = None,
    basis: Basis = Basis.Z,
    rotated: bool = True,
    print_progress: bool = True,
    progress_callback: Callable[[SweepProgress], None] | None = None,
    max_batch_seconds: int | None = None,
) -> list[SweepPoint]:
    """Collect a sweep with sinter and return flattened results.

    ``progress_callback`` is called as collection proceeds, with tasks sinter has
    *finished* — see :func:`_progress_adapter`, which derives that by mirroring sinter's
    own stopping rule because sinter exposes no per-task done signal. Counting tasks
    *started* instead pins a bar at 100% while the sweep runs on, which is the "looks
    hung" failure a progress bar exists to prevent.

    **Raising from ``progress_callback`` cancels the sweep.** The exception propagates out
    of sinter's consumption loop, closes the ``iter_collect`` generator, and
    ``CollectionManager.__exit__`` calls ``hard_stop()``, which kills and joins every
    sinter worker. That is why a cancelled sweep leaves no orphaned processes behind —
    verified by ``tests/test_ui_worker.py`` rather than assumed.

    ``max_batch_seconds`` bounds how long that takes to be noticed. sinter's default worker
    flush period is 120 s, so without it a cancel can go unobserved for two minutes against
    a 10-second grace window — the supervisor would give up and kill the worker instead.
    """
    # Resolve before build_tasks. sinter only discovers a bad decoder name or a missing
    # backend inside a worker, after every circuit has been constructed; validating first
    # means a typo costs nothing rather than the construction of the whole task grid.
    resolved = resolve_decoders(decoders)
    tasks = build_tasks(distances, error_rates, noise_model, rounds, basis, rotated)
    # sinter expands one Task per decoder before collecting (`_collection.py` rebuilds the
    # task iterable as `for task in tasks for decoder in decoders`), so the grid a progress
    # denominator has to count is the product, not len(tasks).
    total_tasks = len(tasks) * len(resolved)
    # Passed only when set, because sinter's own default is the right one for a terminal
    # run and naming it here would pin a number this module has no opinion about.
    extra: dict[str, Any] = {}
    if max_batch_seconds is not None:
        extra["max_batch_seconds"] = max_batch_seconds
    stats: list[sinter.TaskStats] = sinter.collect(
        num_workers=workers,
        tasks=tasks,
        decoders=list(resolved),
        max_shots=max_shots,
        max_errors=max_errors,
        print_progress=print_progress,
        progress_callback=(
            None
            if progress_callback is None
            else _progress_adapter(progress_callback, total_tasks, max_shots, max_errors)
        ),
        **extra,
    )
    points: list[SweepPoint] = []
    for stat in stats:
        meta: dict[str, Any] = stat.json_metadata or {}
        points.append(
            SweepPoint(
                decoder=stat.decoder or "unknown",
                distance=int(meta.get("d", 0)),
                p=float(meta.get("p", 0.0)),
                rounds=int(meta.get("rounds", 0)),
                noise_model=str(meta.get("noise_model", "")),
                basis=str(meta.get("basis", "")),
                shots=int(stat.shots),
                errors=int(stat.errors),
                discards=int(stat.discards),
            )
        )
    return sorted(points, key=lambda s: (s.decoder, s.distance, s.p))


def write_csv(points: list[SweepPoint], path: Path) -> None:
    """Write sweep results with Clopper-Pearson bounds on every row.

    A rate is never emitted without an interval.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "decoder",
                "distance",
                "p",
                "rounds",
                "noise_model",
                "basis",
                "shots",
                "errors",
                "discards",
                "logical_error_rate",
                "ci_low",
                "ci_high",
            ]
        )
        for point in points:
            if point.shots == 0:
                # max(shots, 1) would report a fabricated one-trial interval for a
                # point that was never measured.
                continue
            interval = clopper_pearson(point.errors, point.shots)
            writer.writerow(
                [
                    point.decoder,
                    point.distance,
                    point.p,
                    point.rounds,
                    point.noise_model,
                    point.basis,
                    point.shots,
                    point.errors,
                    point.discards,
                    f"{interval.point:.8g}",
                    f"{interval.low:.8g}",
                    f"{interval.high:.8g}",
                ]
            )


def _intervals_by_decoder(
    points: Sequence[SweepPoint],
    alpha: float = 0.05,
) -> dict[str, dict[int, dict[float, Interval]]]:
    """Group measured points into ``{decoder: {distance: {p: Interval}}}``.

    Zero-shot points are dropped, matching :func:`write_csv` and :func:`plot_threshold`:
    a point that was never measured cannot bracket or constrain anything.

    ``alpha`` must reach these intervals rather than stopping at the fit: they carry the
    caller's confidence level into both the crossing rule and the suppression weights, so
    leaving them at a hardcoded default would make a requested alpha a silent no-op — and
    the sidecar records the requested value as if it applied.
    """
    grouped: dict[str, dict[int, dict[float, Interval]]] = {}
    for point in points:
        if point.shots == 0:
            continue
        interval = clopper_pearson(point.errors, point.shots, alpha=alpha)
        grouped.setdefault(point.decoder, {}).setdefault(point.distance, {})[point.p] = interval
    return grouped


def crossing_from_sweep(
    points: Sequence[SweepPoint], alpha: float = 0.05
) -> dict[str, float | None]:
    """Bracketing threshold crossing per decoder, as a reported result.

    The sweep is multi-decoder, so the crossing is not a single number: one entry per
    decoder present, mapping to None where no crossing is visible in the sampled range --
    which is an answer, not a failure.

    Delegates the rule to :func:`qecgen.qa.crossing_from_intervals` rather than
    reimplementing it, so the sweep and QA paths cannot disagree about what a crossing is.
    """
    return {
        decoder: crossing_from_intervals(by_distance)
        for decoder, by_distance in _intervals_by_decoder(points, alpha=alpha).items()
    }


def suppression_from_sweep(
    points: Sequence[SweepPoint], alpha: float = 0.05
) -> dict[tuple[str, float], SuppressionFit]:
    """Exponential-suppression fit per ``(decoder, p)``, across distances.

    ``Lambda`` is a function of ``p`` and is only interpretable below threshold. Fitting a
    single ``Lambda`` across an entire sweep would average the sub-threshold and
    super-threshold regimes into a number describing neither.
    """
    by_decoder_p: dict[tuple[str, float], dict[int, Interval]] = {}
    for decoder, by_distance in _intervals_by_decoder(points, alpha=alpha).items():
        for distance, by_p in by_distance.items():
            for p, interval in by_p.items():
                by_decoder_p.setdefault((decoder, p), {})[distance] = interval

    fits: dict[tuple[str, float], SuppressionFit] = {}
    for key, intervals in by_decoder_p.items():
        fit = fit_suppression(intervals, alpha=alpha)
        if fit is not None:
            fits[key] = fit
    return fits


def censored_points(
    points: Sequence[SweepPoint], max_errors: int, max_shots: int
) -> list[SweepPoint]:
    """Points that hit the shot ceiling before reaching the error target.

    A measured fact about what happened, not a heuristic guess about what will. These are
    the points whose intervals are widest, and they are exactly the ones a reader is most
    likely to over-trust.
    """
    return [
        point
        for point in points
        if point.shots >= max_shots and point.errors < max_errors and point.shots > 0
    ]


def threshold_summary(
    points: Sequence[SweepPoint],
    *,
    max_errors: int,
    max_shots: int,
    noise_model: str,
    basis: str,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """The crossing and suppression summary, as data.

    Split out from :func:`write_threshold_json` so the sidecar on disk and the summary a
    web UI renders are provably the same object, rather than two renderings that agree
    today. A browser assembling its own version from the CSV would be a second
    implementation of the crossing rule — the rule ``qa.crossing_from_intervals`` exists
    to keep singular.
    """
    crossings = crossing_from_sweep(points, alpha=alpha)
    fits = suppression_from_sweep(points, alpha=alpha)

    decoders: dict[str, dict[str, object]] = {}
    for decoder, crossing in sorted(crossings.items()):
        suppression = [
            {
                "p": p,
                "lambda": fit.lambda_point,
                "lambda_low": fit.lambda_low,
                "lambda_high": fit.lambda_high,
                "prefactor": fit.prefactor,
                "distances_used": list(fit.distances),
                "excluded_zero_error": list(fit.excluded_zero_error),
                "residual_dof": fit.residual_dof,
                "reduced_chi_square": fit.reduced_chi_square,
                "suppressing": fit.suppressing,
            }
            for (fit_decoder, p), fit in sorted(fits.items())
            if fit_decoder == decoder
        ]
        decoders[decoder] = {
            "crossing_p": crossing,
            "crossing_method": (
                "lowest sampled p at which the interval ordering demonstrably reverses: "
                "the largest distance's Clopper-Pearson interval sits entirely below the "
                "smallest's at a lower p and entirely above it here; overlapping "
                "intervals claim nothing. A bracketing estimate from the grid, not a fit"
            ),
            "suppression": suppression,
        }

    payload = {
        "qecgen_version": version("qecgen"),
        "alpha": alpha,
        "reported_not_asserted": (
            "Threshold crossing and Lambda are results, not constants. Both depend on the "
            "channel convention, rounds, basis and decoder. Do not compare these to a "
            "published number without matching all four."
        ),
        "dem_seen_by_decoders": (
            "decomposed (sinter derives it with decompose_errors=True, "
            "approximate_disjoint_errors=True) and hands the same DEM to every decoder"
        ),
        "noise_model": noise_model,
        "basis": basis,
        "max_errors": max_errors,
        "max_shots": max_shots,
        "censored_points": [
            {
                "decoder": point.decoder,
                "distance": point.distance,
                "p": point.p,
                "shots": point.shots,
                "errors": point.errors,
            }
            for point in censored_points(points, max_errors, max_shots)
        ],
        "decoders": decoders,
    }
    return payload


def write_threshold_json(
    points: Sequence[SweepPoint],
    path: Path,
    *,
    max_errors: int,
    max_shots: int,
    noise_model: str,
    basis: str,
    alpha: float = 0.05,
) -> None:
    """Write the crossing and suppression summary beside the sweep CSV.

    Deliberately a sidecar rather than extra CSV columns. ``sweep.csv`` is one row per
    (decoder, distance, p); ``Lambda`` spans distances and the crossing spans the whole
    file, so putting either in a per-point row means repeating a constant across a group,
    where any downstream aggregation double-counts it. JSON also lets the
    "reported, not asserted" disclaimer be a field rather than a junk column.
    """
    payload = threshold_summary(
        points,
        max_errors=max_errors,
        max_shots=max_shots,
        noise_model=noise_model,
        basis=basis,
        alpha=alpha,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False is the enforcement mechanism, not a formality: a bare NaN token is
    # not valid JSON, so a degenerate fit must be converted to null before it gets here
    # rather than passed through as a float.
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def plot_threshold(points: list[SweepPoint], path: Path, title: str | None = None) -> None:
    """Plot one curve per (decoder, distance), log y, error bars on every point.

    Built on the object-oriented Agg API rather than ``pyplot``: a module-level
    ``matplotlib.use("Agg")`` would silently switch the process-wide backend for any host
    application that merely imports something from ``qecgen.sweep``, and ``pyplot`` keeps
    global figure state alive. The OO path renders to the file and touches neither.

    A zero-error point has no place on a log axis at its point estimate of 0, but it
    still carries an informative one-sided Clopper-Pearson bound — and it is usually the
    *most suppressed* point of a quick sweep. Dropping it silently (what a log-scale
    ``errorbar`` does) hides exactly the data a reader most wants, so those points are
    drawn as downward carets at their upper bound instead.

    **This figure is the artifact of record, and it has a second renderer.** The web UI
    draws the same sweep as an interactive SVG (``frontend/src/components/ThresholdChart.tsx``)
    so it can be hovered and have series toggled. That component computes no statistics —
    it reads the rate and both bounds from the columns :func:`write_csv` wrote — but it does
    mirror the caret rule above, because that is a *drawing* decision rather than a derived
    number. Change the convention here and it must change there too; the two disagreeing
    about the same sweep is the failure this note exists to prevent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, int], list[SweepPoint]] = {}
    for point in points:
        if point.shots == 0:
            continue
        grouped.setdefault((point.decoder, point.distance), []).append(point)

    figure = Figure(figsize=(7.5, 5.5))
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    any_zero_error = False
    for (decoder, distance), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda s: s.p)
        measured = [s for s in ordered if s.errors > 0]
        zero_error = [s for s in ordered if s.errors == 0]
        label = f"{decoder}, d={distance}"
        color = None
        if measured:
            intervals = [clopper_pearson(s.errors, s.shots) for s in measured]
            ys = [i.point for i in intervals]
            lower = [max(i.point - i.low, 0.0) for i in intervals]
            upper = [max(i.high - i.point, 0.0) for i in intervals]
            container = axes.errorbar(
                [s.p for s in measured],
                ys,
                yerr=[lower, upper],
                marker="o",
                capsize=3,
                label=label,
            )
            color = container.lines[0].get_color()
        if zero_error:
            any_zero_error = True
            bounds = [clopper_pearson(s.errors, s.shots).high for s in zero_error]
            axes.plot(
                [s.p for s in zero_error],
                bounds,
                linestyle="none",
                marker="v",
                color=color,
                label=None if measured else label,
            )

    if any_zero_error:
        # Proxy artist: the caret needs its own legend entry or the bounds read as data.
        axes.plot(
            [], [], linestyle="none", marker="v", color="black", label="0 errors: upper bound"
        )

    axes.set_yscale("log")
    axes.set_xlabel("physical error rate p")
    axes.set_ylabel("logical error rate per shot")
    axes.set_title(title or "Surface code threshold (Clopper-Pearson 95% intervals)")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
