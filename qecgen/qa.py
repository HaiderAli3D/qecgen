"""Slow statistical quality assurance. Opt-in via ``--qa`` or ``pytest -m slow``.

Every comparison here is between **Clopper-Pearson intervals**, never between point
estimates. Comparing point estimates from finite samples produces a suite that fails
whenever the sampler happens to land badly, which trains everyone to ignore it.

The estimated threshold crossing is reported as a **result**, not asserted against a
hardcoded value. The commonly quoted 0.5-1% figure depends on the channel convention,
round count, basis and decoder, so it is a sanity range to print and not a test to fail
on. The same applies to the suppression factor ``Lambda``.

The threshold kernels -- :func:`crossing_from_intervals` and :func:`fit_suppression` --
are fast and deterministic, unlike the rest of this module. They live here because they
operate on :class:`Interval` and because the crossing rule must have exactly one
definition: ``qecgen.sweep`` adapts to them rather than reimplementing them.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pymatching
import stim
from scipy.stats import beta, norm

from qecgen.circuits import Basis, NoiseModel, build_circuit, default_rounds
from qecgen.dataset import DatasetMeta, EnvironmentSpec
from qecgen.sampling import DEFAULT_CHUNK_SIZE, iter_chunks, unpack_bits

__all__ = [
    "Interval",
    "LogicalErrorEstimate",
    "SuppressionFit",
    "clopper_pearson",
    "crossing_from_intervals",
    "detection_event_rate",
    "estimate_crossing",
    "estimate_environment_rates",
    "estimate_logical_error_rate",
    "fit_suppression",
    "intervals_increasing",
]


@dataclass(frozen=True, slots=True)
class Interval:
    """A Clopper-Pearson confidence interval on a binomial proportion."""

    point: float
    low: float
    high: float
    successes: int
    trials: int

    def __str__(self) -> str:
        """Render as ``point [low, high] (k/n)``."""
        return (
            f"{self.point:.5f} [{self.low:.5f}, {self.high:.5f}] ({self.successes}/{self.trials})"
        )

    def strictly_below(self, other: Interval) -> bool:
        """True when this interval lies entirely below ``other``."""
        return self.high < other.low


def clopper_pearson(successes: int, trials: int, alpha: float = 0.05) -> Interval:
    """Exact binomial confidence interval.

    Uses the Beta quantile form, which is exact rather than normal-approximate. This
    matters at the low error rates a good decoder produces, where a Wald interval can
    extend below zero.

    Args:
        successes: Observed count.
        trials: Total trials.
        alpha: 1 - confidence level. 0.05 gives a 95% interval.
    """
    if trials <= 0:
        raise ValueError(f"trials must be > 0, got {trials}")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes {successes} must lie in [0, {trials}]")

    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    high = (
        1.0
        if successes == trials
        else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    )
    return Interval(
        point=successes / trials,
        low=low,
        high=high,
        successes=successes,
        trials=trials,
    )


def intervals_increasing(intervals: list[Interval]) -> bool:
    """True when the sequence is strictly increasing in the interval sense.

    Requires each interval to lie entirely below the next. This is a strong claim and
    will be False when two adjacent points are statistically indistinguishable — which
    is the honest answer, not a failure to detect a trend.
    """
    return all(a.strictly_below(b) for a, b in itertools.pairwise(intervals))


@dataclass(frozen=True)
class LogicalErrorEstimate:
    """A logical error rate measurement with its interval and provenance."""

    distance: int
    p: float
    rounds: int
    interval: Interval
    detection_event_rate: float

    def __str__(self) -> str:
        """Render as one summary line."""
        return (
            f"d={self.distance} p={self.p:<8g} rounds={self.rounds} "
            f"logical={self.interval} det_rate={self.detection_event_rate:.5f}"
        )


def estimate_logical_error_rate(
    distance: int,
    p: float,
    seed: int,
    noise_model: NoiseModel = NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL,
    rounds: int | None = None,
    basis: Basis = Basis.Z,
    rotated: bool = True,
    max_shots: int = 2_000_000,
    target_errors: int = 200,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    alpha: float = 0.05,
) -> LogicalErrorEstimate:
    """Measure the logical error rate under PyMatching, with adaptive shot count.

    Sampling continues until ``target_errors`` logical failures have been observed or
    ``max_shots`` is reached. A fixed shot count either wastes hours at low ``p`` or
    returns a number with no statistical content.

    The decoder is built with ``pymatching.Matching.from_detector_error_model`` on the
    original decomposed DEM object — never reconstructed from our own ``H``, which would
    validate the parser against itself.
    """
    circuit, _ = build_circuit(distance, p, noise_model, rounds, basis, rotated)
    return estimate_logical_error_rate_for_circuit(
        circuit,
        distance=distance,
        p=p,
        rounds=default_rounds(noise_model, distance, rounds),
        seed=seed,
        max_shots=max_shots,
        target_errors=target_errors,
        chunk_size=chunk_size,
        alpha=alpha,
    )


def estimate_logical_error_rate_for_circuit(
    circuit: stim.Circuit,
    distance: int,
    p: float,
    rounds: int,
    seed: int,
    max_shots: int = 2_000_000,
    target_errors: int = 200,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    alpha: float = 0.05,
) -> LogicalErrorEstimate:
    """Measure the logical error rate of an **already built** circuit.

    Exists because rebuilding a circuit from ``(distance, p, noise_model)`` alone
    discards the drift axis and its value. An ``xz_bias`` environment rebuilt that way
    is measured as if it were plain uniform noise, silently reporting a number for a
    circuit that was never generated. Callers holding a real environment should pass
    its circuit here rather than reconstructing an approximation of it.
    """
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)

    n_detectors = circuit.num_detectors
    n_observables = circuit.num_observables
    resolved_rounds = rounds

    errors = 0
    shots_done = 0
    detection_events = 0

    while shots_done < max_shots and errors < target_errors:
        take = min(chunk_size, max_shots - shots_done)
        # Seed is offset per batch so successive batches are independent but the whole
        # measurement remains reproducible from `seed` alone.
        for chunk in iter_chunks(
            circuit, take, seed=seed + shots_done, chunk_size=take, emit_mechanisms=False
        ):
            predictions = matching.decode_batch(
                chunk.detectors, bit_packed_shots=True, bit_packed_predictions=True
            )
            actual = unpack_bits(chunk.observables, n_observables)
            predicted = unpack_bits(predictions, n_observables)
            errors += int(np.any(predicted != actual, axis=1).sum())
            detection_events += int(unpack_bits(chunk.detectors, n_detectors).sum())
            shots_done += chunk.n_shots

    interval = clopper_pearson(errors, shots_done, alpha)
    det_rate = detection_events / (shots_done * n_detectors) if shots_done else 0.0
    return LogicalErrorEstimate(
        distance=distance,
        p=p,
        rounds=resolved_rounds,
        interval=interval,
        detection_event_rate=det_rate,
    )


def estimate_environment_rates(
    meta: DatasetMeta,
    max_shots: int = 50_000,
    target_errors: int = 100,
) -> list[tuple[EnvironmentSpec, LogicalErrorEstimate]]:
    """Rebuild each recorded environment and measure its logical error rate.

    Each environment is rebuilt through :func:`qecgen.environments.build_environment`
    using its **recorded axis and axis value**, not just ``(distance, p)``. Rebuilding
    from the rate alone reconstructs plain uniform noise, so an ``xz_bias`` or
    ``measurement_ratio`` environment would be measured as a circuit that was never
    generated.

    Lives here rather than in the CLI because it is domain orchestration: a front end's
    job ends at argument resolution and presentation, and a second front end wanting QA
    must not have to re-derive the rebuild rule.
    """
    # Deferred import: environments drags in h5py via its streaming writer, which a
    # sweep-only consumer of this module never touches.
    from qecgen.environments import DriftAxis, build_environment

    results: list[tuple[EnvironmentSpec, LogicalErrorEstimate]] = []
    for env in meta.environments:
        build = build_environment(
            environment_id=env.environment_id,
            distance=meta.distance,
            base_p=env.p,
            axis=DriftAxis(env.axis),
            axis_value=env.axis_value,
            shots=env.shots,
            noise_model=env.noise_model,
            rounds=meta.rounds,
            basis=meta.basis,
            rotated=meta.rotated,
        )
        estimate = estimate_logical_error_rate_for_circuit(
            build.circuit,
            distance=meta.distance,
            p=env.p,
            rounds=meta.rounds,
            seed=meta.seed + env.environment_id,
            max_shots=max_shots,
            target_errors=target_errors,
        )
        results.append((env, estimate))
    return results


def crossing_from_intervals(
    intervals_by_distance: Mapping[int, Mapping[float, Interval]],
) -> float | None:
    """The crossing rule, over the only data it actually touches: ``p`` and an interval.

    Finds the lowest ``p`` at which the ordering of the largest and smallest distance
    *demonstrably* reverses: the largest distance's interval must lie entirely below the
    smallest's at some lower ``p`` (suppression was actually observed) and entirely above
    it at the reported ``p``. Overlapping intervals claim nothing. An earlier version
    compared ``point`` estimates with ``>=``, which fabricated a crossing at the lowest
    sampled ``p`` whenever the two curves tied there — including two zero-error points in
    a quick sweep — and contradicted the module rule that every comparison is between
    intervals, never point estimates.

    This is a coarse bracketing estimate from the sampled grid, not a fit, and it is
    deliberately not asserted against any published value: the threshold location
    depends on the channel convention, rounds, basis and decoder.

    Extracted so the sweep path and the QA path cannot drift into two different
    definitions of "crossing". :func:`estimate_crossing` and
    ``qecgen.sweep.crossing_from_sweep`` are both adapters over this.

    Returns:
        The bracketing ``p``, or None when the sampled range never shows both the
        separated ordering and its demonstrated reversal.
    """
    distances = sorted(intervals_by_distance)
    if len(distances) < 2:
        return None
    low_by_p = intervals_by_distance[distances[0]]
    high_by_p = intervals_by_distance[distances[-1]]

    # Match by p, never by list position. Zipping two curves positionally compares
    # unrelated physical error rates whenever the grids differ in length or ordering,
    # and reports a crossing that was never measured.
    shared = sorted(set(low_by_p) & set(high_by_p))

    outperformed = False
    for p in shared:
        if high_by_p[p].strictly_below(low_by_p[p]):
            outperformed = True
        elif outperformed and low_by_p[p].strictly_below(high_by_p[p]):
            return p
    return None


def estimate_crossing(
    estimates_by_distance: dict[int, list[LogicalErrorEstimate]],
) -> float | None:
    """Estimate where curves for different distances cross, as a **reported result**.

    Thin adapter over :func:`crossing_from_intervals`; see that function for the rule.
    """
    return crossing_from_intervals(
        {
            distance: {estimate.p: estimate.interval for estimate in estimates}
            for distance, estimates in estimates_by_distance.items()
        }
    )


@dataclass(frozen=True, slots=True)
class SuppressionFit:
    """Weighted log-linear fit of ``p_L(d) ~ A * Lambda ** (-(d + 1) / 2)``.

    ``Lambda`` ships with an interval, never as a bare number, matching the rule
    :func:`clopper_pearson` sets for every rate in this package. It cannot reuse
    :class:`Interval`, which carries ``successes``/``trials`` and documents itself as a
    binomial proportion; ``Lambda`` is a ratio of two proportions.

    ``Lambda`` is only interpretable **below threshold**. Above it the slope reverses and
    ``Lambda < 1``, meaning larger distance hurts. That is a real measurement and is
    reported rather than clamped away.
    """

    lambda_point: float
    lambda_low: float
    lambda_high: float
    prefactor: float
    distances: tuple[int, ...]
    """Distances that entered the fit."""
    excluded_zero_error: tuple[int, ...]
    """Distances dropped because they observed zero logical failures.

    ``log(0)`` is undefined, and substituting a pseudo-count (0.5/n, Jeffreys,
    ``max(k, 1)``) would fabricate a measurement -- the same thing ``sweep.write_csv``
    already refuses to do for zero-shot rows. Such a point still carries a one-sided
    bound ``p_L <= high``, and it is almost always the *largest* distance that produces
    it, so dropping it removes the most-suppressed point and biases the fitted ``Lambda``
    **downward**. A reported ``Lambda`` therefore under-states suppression whenever this
    tuple is non-empty, which is the conservative direction.
    """
    residual_dof: int
    reduced_chi_square: float | None
    alpha: float

    @property
    def suppressing(self) -> bool:
        """True when the whole interval lies above 1.

        Mirrors :meth:`Interval.strictly_below`: an interval straddling 1 means the data
        do not establish suppression, which is the honest answer rather than a failure to
        detect a trend.
        """
        return self.lambda_low > 1.0

    def __str__(self) -> str:
        """Render as one summary line."""
        excluded = (
            f" excluded(k=0)={','.join(str(d) for d in self.excluded_zero_error)}"
            if self.excluded_zero_error
            else ""
        )
        chi2 = "" if self.reduced_chi_square is None else f" chi2/dof={self.reduced_chi_square:.2f}"
        return (
            f"Lambda={self.lambda_point:.3g} "
            f"[{self.lambda_low:.3g}, {self.lambda_high:.3g}] "
            f"d={','.join(str(d) for d in self.distances)}{excluded}{chi2}"
        )


def fit_suppression(
    intervals_by_distance: Mapping[int, Interval],
    alpha: float = 0.05,
) -> SuppressionFit | None:
    """Fit the exponential suppression factor across code distances.

    Model: ``p_L(d) = A * Lambda ** (-(d + 1) / 2)``, fitted as a weighted least-squares
    line of ``log p_L`` against ``x = (d + 1) / 2``. With exactly two distances this
    degenerates to the pairwise ratio ``Lambda = p_L(d) / p_L(d + 2)``, so nothing is lost
    relative to reporting ratios; with three or more it also yields a residual, which is
    the only honest signal that the points are in the exponential regime at all.

    The choice of ``x = (d + 1) / 2`` over ``x = d / 2`` is the Google convention and
    affects only the intercept: the two differ by a constant absorbed entirely into ``A``,
    so ``Lambda`` is identical either way.

    Weights are ``1 / sigma**2`` with ``sigma`` the Clopper-Pearson interval half-width in
    log space, converted to a 1-sigma equivalent. Symmetrising an asymmetric interval in
    log space is an approximation, and it is stated rather than hidden: Clopper-Pearson is
    conservative, so the resulting ``Lambda`` interval is conservative too.

    A Wald-derived ``sigma ~ sqrt((1 - p) / k)`` was rejected. This package uses
    Clopper-Pearson precisely because Wald misbehaves at the low rates a good decoder
    produces, so importing a Wald sigma to weight a Clopper-Pearson analysis would be
    inconsistent.

    Returns:
        The fit, or None when fewer than two distances observed a non-zero error count.
    """
    z = float(norm.ppf(1 - alpha / 2))

    excluded: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    ws: list[float] = []
    used: list[int] = []
    for distance in sorted(intervals_by_distance):
        interval = intervals_by_distance[distance]
        if interval.successes == 0 or interval.point <= 0.0:
            excluded.append(distance)
            continue
        # low is 0 only when successes == 0, already excluded above, so both logs are
        # finite here.
        sigma = (math.log(interval.high) - math.log(interval.low)) / (2 * z)
        if sigma <= 0.0:
            excluded.append(distance)
            continue
        used.append(distance)
        xs.append((distance + 1) / 2)
        ys.append(math.log(interval.point))
        ws.append(1.0 / sigma**2)

    if len(set(xs)) < 2:
        return None

    s = sum(ws)
    sx = sum(w * x for w, x in zip(ws, xs, strict=True))
    sy = sum(w * y for w, y in zip(ws, ys, strict=True))
    sxx = sum(w * x * x for w, x in zip(ws, xs, strict=True))
    sxy = sum(w * x * y for w, x, y in zip(ws, xs, ys, strict=True))
    denominator = s * sxx - sx * sx
    if denominator <= 0.0:
        return None

    slope = (s * sxy - sx * sy) / denominator
    intercept = (sxx * sy - sx * sxy) / denominator
    se_slope = math.sqrt(s / denominator)

    chi_square = sum(
        w * (y - intercept - slope * x) ** 2 for w, x, y in zip(ws, xs, ys, strict=True)
    )
    dof = len(used) - 2

    # Lambda = exp(-slope) is strictly monotone in slope, so the interval on the slope
    # maps exactly onto an interval on Lambda -- no delta method needed. The interval is
    # deliberately NOT rescaled by sqrt(reduced chi-square): with 4 distances there are 2
    # residual degrees of freedom, so reduced chi-square is noise, and the weights come
    # from measured counts rather than from residuals.
    return SuppressionFit(
        lambda_point=math.exp(-slope),
        lambda_low=math.exp(-(slope + z * se_slope)),
        lambda_high=math.exp(-(slope - z * se_slope)),
        prefactor=math.exp(intercept),
        distances=tuple(used),
        excluded_zero_error=tuple(excluded),
        residual_dof=dof,
        reduced_chi_square=chi_square / dof if dof > 0 else None,
        alpha=alpha,
    )


def detection_event_rate(circuit: stim.Circuit, shots: int, seed: int) -> float:
    """Mean per-detector firing rate. Used to check the rate rises with ``p``."""
    if shots <= 0:
        # min(shots, 50_000) below would hand chunk_size=0 to iter_chunks, which raises.
        # Zero shots is a measured nothing, not an error — the same answer the trailing
        # `if done else 0.0` guard already gives.
        return 0.0
    n_detectors = circuit.num_detectors
    total = 0
    done = 0
    for chunk in iter_chunks(circuit, shots, seed=seed, chunk_size=min(shots, 50_000)):
        total += int(unpack_bits(chunk.detectors, n_detectors).sum())
        done += chunk.n_shots
    return total / (done * n_detectors) if done else 0.0
