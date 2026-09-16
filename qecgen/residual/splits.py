"""Deterministic split assignment: one int8 code per row, fixed before anything is fitted.

The split column is what makes "no fitting on validation/test rows" auditable. It is
therefore computed from the row count, the method, the fractions and the seed alone,
never from the data, and it is reproducible on any machine: a resumed build or an
independent re-run must produce the identical column or every downstream claim about
held-out rows is void.

This module deliberately imports nothing from :mod:`qecgen.residual.config`. The method
is taken as its string value (``SplitMethod`` is a ``StrEnum``, so its members pass
straight through), which keeps the split arithmetic importable and testable on its own
and rules out an import cycle between the config layer and the code it describes.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

SPLIT_NAMES: tuple[str, ...] = ("calibration", "train", "validation", "test")
"""Block order. ``calibration`` (code 0) is reserved for a source whose matching weights
are estimated from data; no v1 deliverable uses it, but it must come *first* in row order
so that a contiguous split never estimates weights from rows that follow the held-out
blocks."""

SPLIT_CODES: dict[str, int] = {name: code for code, name in enumerate(SPLIT_NAMES)}
"""The int8 code written to the CSV ``split`` column (as a name) and the raw HDF5 (as a
code). The mapping is recorded in the HDF5 attributes so a reader never infers it."""

SEEDED_PERMUTATION = "seeded_permutation"
CONTIGUOUS_BLOCKS = "contiguous_blocks"
SPLIT_METHODS: tuple[str, ...] = (SEEDED_PERMUTATION, CONTIGUOUS_BLOCKS)

_FRACTION_SUM_TOLERANCE = 1e-9


def _validated_fractions(fractions: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    """Return ``(name, fraction)`` pairs in ``SPLIT_NAMES`` order, refusing anything odd.

    Ordering by ``SPLIT_NAMES`` rather than by the mapping's insertion order matters for
    ``contiguous_blocks``: a JSON config that lists ``test`` before ``train`` must not
    quietly swap which rows are held out.
    """
    if not fractions:
        raise ValueError("splits: at least one split fraction is required")
    unknown = sorted(set(fractions) - set(SPLIT_NAMES))
    if unknown:
        raise ValueError(f"splits: unknown split name(s) {unknown}; allowed: {list(SPLIT_NAMES)}")
    ordered: list[tuple[str, float]] = []
    for name in SPLIT_NAMES:
        if name not in fractions:
            continue
        fraction = float(fractions[name])
        if not np.isfinite(fraction) or fraction <= 0.0:
            raise ValueError(f"splits: fraction for {name!r} must be positive, got {fraction!r}")
        ordered.append((name, fraction))
    total = sum(fraction for _, fraction in ordered)
    if abs(total - 1.0) > _FRACTION_SUM_TOLERANCE:
        raise ValueError(f"splits: fractions must sum to 1.0 (within 1e-9), got {total!r}")
    return tuple(ordered)


def split_boundaries(
    n_rows: int, fractions: Mapping[str, float]
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Return the block names and their exclusive end indices, in ``SPLIT_NAMES`` order.

    Boundaries are ``round(cumulative_fraction * n_rows)`` (a one-argument ``round`` on a
    float already yields an ``int``, so no cast is needed) rather than a running sum of
    per-block ``round(fraction * n_rows)``: the latter drifts by one row per block, and
    the rounded *cumulative* product is what gives exactly 70/15/15 for the deliverable
    row counts (100, 300,000, 304,000) and 60/20/20 for 50,000. The last boundary is
    forced to ``n_rows`` because a float cumulative sum of fractions that pass the
    tolerance check can still land a hair under 1.0 and strand the final row. Python's
    ``round`` uses banker's rounding on an exact .5, so a boundary may land one row either
    side of the naive expectation for small ``n_rows``; every block is still non-empty or
    the caller raises.
    """
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows < 1:
        raise ValueError(f"splits: n_rows must be a positive int, got {n_rows!r}")
    ordered = _validated_fractions(fractions)
    names = tuple(name for name, _ in ordered)
    bounds: list[int] = []
    cumulative = 0.0
    for _, fraction in ordered:
        cumulative += fraction
        bounds.append(round(cumulative * n_rows))
    bounds[-1] = n_rows
    previous = 0
    for name, bound in zip(names, bounds, strict=True):
        if bound <= previous:
            raise ValueError(
                f"splits: block {name!r} would be empty for n_rows={n_rows} with fractions "
                f"{dict(ordered)}; a dataset with an empty split is not publishable"
            )
        previous = bound
    return names, tuple(bounds)


def assign_splits(
    n_rows: int,
    method: str,
    fractions: Mapping[str, float],
    seed: int | None,
) -> np.ndarray:
    """Return an int8 array of length ``n_rows`` holding one ``SPLIT_CODES`` value per row.

    ``seeded_permutation`` shuffles row indices with
    ``np.random.default_rng(np.random.SeedSequence(seed)).permutation(n_rows)`` and cuts
    the permutation at the boundaries. The RNG construction is part of the contract (the
    tests pin it) because the resulting column is recorded in published artifacts.

    ``contiguous_blocks`` cuts ``arange(n_rows)`` at the same boundaries, preserving
    source row order for data whose acquisition order may carry drift. It takes no seed,
    and a seed is *refused* rather than ignored: a config that names a seed for a
    contiguous split is a config whose author expected shuffling.

    ``method`` is compared by string value so that a ``SplitMethod`` member and its
    literal value behave identically.
    """
    method_name = str(method)
    if method_name not in SPLIT_METHODS:
        raise ValueError(f"splits: unknown method {method_name!r}; allowed: {list(SPLIT_METHODS)}")
    names, bounds = split_boundaries(n_rows, fractions)

    if method_name == SEEDED_PERMUTATION:
        if seed is None or isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"splits: {SEEDED_PERMUTATION} requires an integer seed, got {seed!r}")
        order = np.random.default_rng(np.random.SeedSequence(seed)).permutation(n_rows)
    else:
        if seed is not None:
            raise ValueError(
                f"splits: {CONTIGUOUS_BLOCKS} takes no seed (got {seed!r}); it preserves "
                "source row order and has nothing to seed"
            )
        order = np.arange(n_rows)

    codes = np.empty(n_rows, dtype=np.int8)
    start = 0
    for name, stop in zip(names, bounds, strict=True):
        codes[order[start:stop]] = SPLIT_CODES[name]
        start = stop
    return codes


__all__ = [
    "CONTIGUOUS_BLOCKS",
    "SEEDED_PERMUTATION",
    "SPLIT_CODES",
    "SPLIT_METHODS",
    "SPLIT_NAMES",
    "assign_splits",
    "split_boundaries",
]
