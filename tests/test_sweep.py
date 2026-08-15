"""Sweep collection, threshold reporting and decoder plurality.

Most tests here build ``SweepPoint`` values by hand so they never invoke sinter. The one
test that actually collects is marked ``slow`` and gated on the optional ``decoders``
extra.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import requires_mwpf

from qecgen.qa import clopper_pearson, estimate_crossing, fit_suppression
from qecgen.sweep import (
    SweepPoint,
    censored_points,
    crossing_from_sweep,
    run_sweep,
    suppression_from_sweep,
    write_csv,
    write_threshold_json,
)


def _point(decoder: str, distance: int, p: float, shots: int, errors: int) -> SweepPoint:
    return SweepPoint(
        decoder=decoder,
        distance=distance,
        p=p,
        rounds=distance,
        noise_model="stim_uniform_circuit_level",
        basis="z",
        shots=shots,
        errors=errors,
        discards=0,
    )


def test_write_csv_emits_an_interval_on_every_row(tmp_path: Path) -> None:
    """House rule: a rate is never emitted without an interval."""
    points = [_point("pymatching", 3, 0.005, 10_000, 120)]
    out = tmp_path / "sweep.csv"
    write_csv(points, out)
    header, row = out.read_text().strip().splitlines()
    assert "ci_low" in header and "ci_high" in header
    fields = dict(zip(header.split(","), row.split(","), strict=True))
    assert float(fields["ci_low"]) < float(fields["logical_error_rate"]) < float(fields["ci_high"])


def test_run_sweep_rejects_an_unknown_decoder_before_building_tasks() -> None:
    """Validation must beat circuit construction; sinter only notices inside a worker."""
    with pytest.raises(ValueError, match="unknown decoder"):
        run_sweep(distances=[3], error_rates=[0.01], decoders=("nope",))


def _planted(distance: int, p: float, lam: float, prefactor: float, shots: int) -> SweepPoint:
    """A point obeying p_L(d) = prefactor * lam ** (-(d + 1) / 2) exactly."""
    rate = prefactor * lam ** (-(distance + 1) / 2)
    return _point("pymatching", distance, p, shots, round(rate * shots))


# --- crossing ---------------------------------------------------------------------


def test_crossing_from_sweep_is_per_decoder() -> None:
    """The sweep is multi-decoder, so the crossing is not a single number."""
    points = [
        # pymatching: d=5 beats d=3 at 0.005, loses at 0.02 -> crossing at 0.02
        _point("pymatching", 3, 0.005, 100_000, 2_000),
        _point("pymatching", 5, 0.005, 100_000, 500),
        _point("pymatching", 3, 0.020, 100_000, 5_000),
        _point("pymatching", 5, 0.020, 100_000, 9_000),
        # vacuous: d=5 never demonstrably beats d=3, so there is no ordering whose
        # reversal could be a crossing -- None, not a fabricated crossing at the lowest p
        _point("vacuous", 3, 0.005, 100_000, 500),
        _point("vacuous", 5, 0.005, 100_000, 2_000),
    ]
    assert crossing_from_sweep(points) == {"pymatching": 0.020, "vacuous": None}


def test_crossing_from_sweep_reports_none_when_no_crossing_is_visible() -> None:
    """No crossing in the sampled range is an answer, not a failure."""
    points = [
        _point("pymatching", 3, 0.001, 100_000, 2_000),
        _point("pymatching", 5, 0.001, 100_000, 300),
    ]
    assert crossing_from_sweep(points) == {"pymatching": None}


def test_crossing_from_sweep_ignores_zero_shot_points() -> None:
    """A point that was never measured cannot bracket anything."""
    points = [
        _point("pymatching", 3, 0.005, 0, 0),
        _point("pymatching", 5, 0.005, 0, 0),
    ]
    assert crossing_from_sweep(points) == {}


def test_estimate_crossing_and_crossing_from_sweep_agree() -> None:
    """One rule, two adapters.

    `estimate_crossing` was orphaned -- imported only by the regression suite, never by
    the CLI. It is now a thin adapter over the same kernel `crossing_from_sweep` uses,
    and this locks the two together.
    """
    from qecgen.qa import LogicalErrorEstimate

    points = [
        _point("pymatching", 3, 0.005, 100_000, 2_000),
        _point("pymatching", 5, 0.005, 100_000, 500),
        _point("pymatching", 3, 0.020, 100_000, 5_000),
        _point("pymatching", 5, 0.020, 100_000, 9_000),
    ]
    equivalent = {
        distance: [
            LogicalErrorEstimate(
                distance=point.distance,
                p=point.p,
                rounds=point.rounds,
                interval=clopper_pearson(point.errors, point.shots),
                detection_event_rate=0.0,
            )
            for point in points
            if point.distance == distance
        ]
        for distance in (3, 5)
    }
    assert estimate_crossing(equivalent) == crossing_from_sweep(points)["pymatching"]


# --- suppression fit --------------------------------------------------------------


@pytest.mark.parametrize("planted_lambda", [2.0, 3.0, 5.0])
def test_suppression_fit_recovers_a_planted_lambda(planted_lambda: float) -> None:
    """Synthetic data with a known answer. This tests the machinery, not the physics."""
    intervals = {
        d: clopper_pearson(round(0.5 * planted_lambda ** (-(d + 1) / 2) * 20_000_000), 20_000_000)
        for d in (3, 5, 7)
    }
    fit = fit_suppression(intervals)
    assert fit is not None
    assert fit.lambda_low <= planted_lambda <= fit.lambda_high
    assert fit.distances == (3, 5, 7)
    assert fit.suppressing


def test_suppression_interval_narrows_with_more_shots() -> None:
    """More shots must buy a tighter Lambda, not just a different one."""

    def fit_at(shots: int) -> float:
        intervals = {
            d: clopper_pearson(round(0.5 * 3.0 ** (-(d + 1) / 2) * shots), shots) for d in (3, 5, 7)
        }
        fit = fit_suppression(intervals)
        assert fit is not None
        return fit.lambda_high - fit.lambda_low

    assert fit_at(20_000_000) < fit_at(200_000)


def test_zero_error_points_are_excluded_not_pseudo_counted() -> None:
    """No 0.5/n, no Jeffreys, no max(k, 1). Substituting a count fabricates a measurement."""
    intervals = {
        3: clopper_pearson(900, 1_000_000),
        5: clopper_pearson(90, 1_000_000),
        7: clopper_pearson(0, 1_000_000),
    }
    fit = fit_suppression(intervals)
    assert fit is not None
    assert fit.excluded_zero_error == (7,)
    assert fit.distances == (3, 5)
    # Identical to the fit over the surviving subset -- the dropped point contributes
    # nothing at all rather than contributing a fabricated value.
    subset = fit_suppression({3: intervals[3], 5: intervals[5]})
    assert subset is not None
    assert fit.lambda_point == subset.lambda_point


def test_fewer_than_two_usable_distances_returns_none() -> None:
    assert fit_suppression({3: clopper_pearson(100, 10_000)}) is None
    assert fit_suppression({3: clopper_pearson(0, 10_000), 5: clopper_pearson(10, 10_000)}) is None
    assert fit_suppression({}) is None


def test_two_distances_reproduce_the_pairwise_ratio() -> None:
    """The fit degenerates exactly to Lambda = p_L(d) / p_L(d + 2) at two points."""
    intervals = {3: clopper_pearson(2_000, 1_000_000), 5: clopper_pearson(400, 1_000_000)}
    fit = fit_suppression(intervals)
    assert fit is not None
    ratio = intervals[3].point / intervals[5].point
    assert fit.lambda_point == pytest.approx(ratio, rel=1e-9)


def test_lambda_is_invariant_to_the_exponent_convention() -> None:
    """x = (d + 1) / 2 and x = d / 2 differ by a constant absorbed into the prefactor.

    Shifting every distance by a constant shifts x by a constant, which changes the
    intercept and leaves the slope -- and therefore Lambda -- untouched.
    """
    base = {
        d: clopper_pearson(round(0.5 * 4.0 ** (-(d + 1) / 2) * 5_000_000), 5_000_000)
        for d in (3, 5, 7)
    }
    shifted = {d + 2: interval for d, interval in base.items()}
    original = fit_suppression(base)
    moved = fit_suppression(shifted)
    assert original is not None and moved is not None
    assert original.lambda_point == pytest.approx(moved.lambda_point, rel=1e-9)


def test_above_threshold_lambda_is_reported_below_one_not_clamped() -> None:
    """Above threshold larger distance hurts. That is a real, reportable measurement."""
    intervals = {
        3: clopper_pearson(50_000, 1_000_000),
        5: clopper_pearson(120_000, 1_000_000),
        7: clopper_pearson(250_000, 1_000_000),
    }
    fit = fit_suppression(intervals)
    assert fit is not None
    assert fit.lambda_point < 1.0
    assert not fit.suppressing


def test_straddling_interval_is_not_reported_as_suppressing() -> None:
    """Overlapping evidence gives an honest 'no', mirroring intervals_increasing."""
    intervals = {3: clopper_pearson(100, 10_000), 5: clopper_pearson(95, 10_000)}
    fit = fit_suppression(intervals)
    assert fit is not None
    assert fit.lambda_low < 1.0 < fit.lambda_high
    assert not fit.suppressing


def test_suppression_from_sweep_is_keyed_by_decoder_and_p() -> None:
    """Lambda is a function of p; one fit per (decoder, p), never one per sweep."""
    points = [
        _planted(d, p, lam=3.0, prefactor=0.5, shots=5_000_000)
        for p in (0.001, 0.005)
        for d in (3, 5, 7)
    ]
    fits = suppression_from_sweep(points)
    assert set(fits) == {("pymatching", 0.001), ("pymatching", 0.005)}
    for fit in fits.values():
        assert fit.lambda_low <= 3.0 <= fit.lambda_high


# --- censoring and the sidecar ----------------------------------------------------


def test_censored_points_reports_ceiling_hits_only() -> None:
    points = [
        _point("pymatching", 7, 0.001, 100_000, 4),  # hit the ceiling, few errors
        _point("pymatching", 3, 0.001, 40_000, 500),  # reached the error target
        _point("pymatching", 5, 0.001, 0, 0),  # never measured
    ]
    censored = censored_points(points, max_errors=500, max_shots=100_000)
    assert [(c.distance, c.errors) for c in censored] == [(7, 4)]


def test_threshold_json_is_standards_compliant_and_carries_the_disclaimer(
    tmp_path: Path,
) -> None:
    """A bare NaN token is not valid JSON, and the caveat must survive into the file."""
    points = [_planted(d, 0.001, lam=3.0, prefactor=0.5, shots=5_000_000) for d in (3, 5, 7)]
    out = tmp_path / "sweep.threshold.json"
    write_threshold_json(
        points, out, max_errors=500, max_shots=10_000_000, noise_model="uniform", basis="z"
    )
    text = out.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    payload = json.loads(text)
    assert "reported_not_asserted" in payload
    assert payload["decoders"]["pymatching"]["suppression"][0]["suppressing"] is True


@pytest.mark.slow
@requires_mwpf
def test_union_find_sweep_produces_well_formed_rows(tmp_path: Path) -> None:
    """Union-Find collects alongside MWPM on identical tasks.

    Deliberately asserts only well-formedness. It does NOT assert that union-find is less
    accurate than matching: sinter's ``hypergraph_union_find`` is MWPF's hypergraph
    solver, not the classic Delfosse-Nickerson union-find, and it measurably *beats*
    pymatching on this noise model. Asserting an accuracy ordering here would encode a
    claim about the brief's expectations rather than about the code.
    """
    points = run_sweep(
        distances=[3],
        error_rates=[0.01],
        max_errors=20,
        max_shots=20_000,
        workers=1,
        decoders=("pymatching", "hypergraph_union_find"),
        print_progress=False,
    )
    decoders = {point.decoder for point in points}
    assert decoders == {"pymatching", "hypergraph_union_find"}
    for point in points:
        assert point.shots > 0
        assert 0 <= point.errors <= point.shots

    out = tmp_path / "uf.csv"
    write_csv(points, out)
    assert out.read_text().count("hypergraph_union_find") == 1
