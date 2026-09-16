"""Schema v1 of the residual features: names, order, exact definitions, leakage audit.

Every feature here is a cheap per-shot summary of the detection events and of
PyMatching's own output. The schema is the contract that lets the same residual model be
trained on any of the datasets, so its column names, order and arithmetic are frozen by
``SCHEMA_VERSION`` and pinned by tests on hand-computed shots; a change to any of them is
a new schema version, never an edit.

Two implementations of the same definitions live here on purpose. :func:`extract_features`
is the vectorised production path (sparse-array products over a block of packed shots);
:func:`extract_features_reference` is a pure-Python, one-shot, sets-and-loops rendering of
the brief's prose that shares no arithmetic with it. The validation spot checks recompute
published rows through the reference, so a broadcasting slip in the fast path - a ``ddof``
default, a transposed membership matrix, a big-endian unpack - is caught against an
implementation that cannot make the same mistake.

**Leakage isolation.** Feature extraction receives detector bits, PyMatching's guess and
weight, and a :class:`FeatureContext` holding *only* the detector count, the time slices
and the frozen graph summary. ``truth`` and ``pm_wrong`` are computed afterwards by the
caller and never enter this module; :func:`audit_feature_inputs` refuses a context that
carries anything else, including a subclass field, so hidden simulator state cannot be
threaded through "for convenience".
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from qecgen.residual.graph import MatchingGraphSummary, TimeSlices
from qecgen.sampling import unpack_bits

__all__ = [
    "ALL_COLUMNS",
    "FEATURE_COLUMNS",
    "INTEGER_COLUMNS",
    "LABEL_COLUMNS",
    "FeatureContext",
    "audit_feature_columns",
    "audit_feature_inputs",
    "extract_features",
    "extract_features_reference",
]

FEATURE_COLUMNS: tuple[str, ...] = (
    "n_fired_total",
    "frac_fired",
    "n_fired_first_round",
    "frac_fired_first_round",
    "n_fired_final_round",
    "frac_fired_final_round",
    "fired_per_round_mean",
    "fired_per_round_max",
    "fired_per_round_std",
    "fired_per_round_range",
    "round_of_max_fired",
    "n_active_rounds",
    "frac_active_rounds",
    "fired_round_center",
    "fired_round_spread",
    "n_fired_boundary_adjacent",
    "frac_fired_boundary_adjacent",
    "n_fired_logical_edge_adjacent",
    "frac_fired_logical_edge_adjacent",
    "n_fired_neighbor_pairs",
    "n_fired_isolated",
    "pm_weight",
    "pm_weight_per_fired",
    "pm_guess",
)
"""The residual model's input columns, in CSV order. ``pm_weight`` is a sum of matching-edge
weights, not a fault count."""

LABEL_COLUMNS: tuple[str, ...] = ("truth", "pm_wrong", "run_id", "split")
"""Labels and metadata. Never inputs: ``truth``/``pm_wrong`` are the target, ``run_id`` and
``split`` would let a model memorise row position."""

ALL_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS + LABEL_COLUMNS

INTEGER_COLUMNS: frozenset[str] = frozenset(
    {
        "n_fired_total",
        "n_fired_first_round",
        "n_fired_final_round",
        "fired_per_round_max",
        "fired_per_round_range",
        "n_active_rounds",
        "n_fired_boundary_adjacent",
        "n_fired_logical_edge_adjacent",
        "n_fired_neighbor_pairs",
        "n_fired_isolated",
        "pm_guess",
        "truth",
        "pm_wrong",
        "run_id",
    }
)
"""Columns the CSV writer formats as integers, so a binary value never reads as ``1.0``."""

_CONTEXT_FIELDS: tuple[str, ...] = ("n_detectors", "slices", "graph")


@dataclass(frozen=True)
class FeatureContext:
    """Everything feature extraction may see besides the shot itself.

    Exactly three fields, deliberately: no observables, no truth, no simulator state and
    no source labels. The class is frozen so a caller cannot attach an extra attribute
    after construction either, and :func:`audit_feature_inputs` checks both routes.

    The two denominators the schema divides by (``|boundary set|`` and ``|logical set|``)
    are asserted non-zero here rather than guarded per shot: a zero would mean the graph
    summary is not a matching graph this schema is defined on, and silently emitting
    ``0/0 -> 0.0`` would hide that in every row.
    """

    n_detectors: int
    slices: TimeSlices
    graph: MatchingGraphSummary

    def __post_init__(self) -> None:
        if self.n_detectors <= 0:
            raise ValueError("a feature context needs at least one detector")
        if self.graph.n_detectors != self.n_detectors:
            raise ValueError(
                f"graph summary covers {self.graph.n_detectors} detectors but the context "
                f"declares {self.n_detectors}"
            )
        if self.slices.slice_of_detector.shape != (self.n_detectors,):
            raise ValueError(
                f"time slices cover {self.slices.slice_of_detector.shape[0]} detectors but "
                f"the context declares {self.n_detectors}"
            )
        if tuple(self.slices.membership.shape) != (self.n_detectors, self.slices.n_slices):
            raise ValueError("slice membership matrix shape disagrees with the slice count")
        if self.slices.n_slices == 0 or self.slices.sizes.shape != (self.slices.n_slices,):
            raise ValueError("time slices must carry one size per slice")
        if int(self.slices.sizes[0]) <= 0 or int(self.slices.sizes[-1]) <= 0:
            raise ValueError("first and final time slices must be non-empty")
        if tuple(self.graph.adjacency.shape) != (self.n_detectors, self.n_detectors):
            raise ValueError("adjacency matrix is not (n_detectors, n_detectors)")
        for name, mask in (
            ("boundary", self.graph.boundary_adjacent),
            ("logical", self.graph.logical_adjacent),
        ):
            if mask.shape != (self.n_detectors,) or mask.dtype != np.bool_:
                raise ValueError(f"{name}-adjacent mask must be a bool array over the detectors")
            if int(mask.sum()) == 0:
                raise ValueError(
                    f"the {name}-adjacent node set is empty; frac_fired_{name}"
                    "-adjacent would divide by zero"
                )


def audit_feature_columns(columns: Sequence[str]) -> None:
    """Refuse any column list other than ``ALL_COLUMNS`` exactly, names and order.

    This is the programmatic leakage audit the brief asks for: a CSV whose header carries
    one extra column (a drift state, an environment id) or a reordered one is refused at
    the schema boundary rather than discovered by a model that trains on it.
    """
    observed = tuple(columns)
    if observed == ALL_COLUMNS:
        return
    expected_set = set(ALL_COLUMNS)
    observed_set = set(observed)
    extra = sorted(observed_set - expected_set)
    missing = sorted(expected_set - observed_set)
    problem = "reordered or duplicated columns"
    if extra or missing:
        problem = f"extra {extra}, missing {missing}"
    raise ValueError(
        f"feature columns differ from schema v1 ({problem}); expected exactly {list(ALL_COLUMNS)}"
    )


def audit_feature_inputs(ctx: FeatureContext) -> None:
    """Refuse a context carrying anything beyond ``n_detectors``, ``slices`` and ``graph``.

    Three routes could smuggle a hidden input past the frozen dataclass: a subclass that
    adds a field, a subclass that adds a class attribute, and ``object.__setattr__`` on
    the instance. All three are checked, and the offending names are listed so the
    failure names the leak rather than merely reporting one.
    """
    extra: set[str] = set()
    for field in dataclasses.fields(ctx):
        if field.name not in _CONTEXT_FIELDS:
            extra.add(field.name)
    instance_dict = getattr(ctx, "__dict__", {})
    for name in instance_dict:
        if name not in _CONTEXT_FIELDS:
            extra.add(name)
    for klass in type(ctx).__mro__:
        if klass is FeatureContext:
            break
        for name in vars(klass):
            if not (name.startswith("__") and name.endswith("__")) and name not in _CONTEXT_FIELDS:
                extra.add(name)
    if type(ctx) is not FeatureContext or extra:
        raise ValueError(
            f"feature context {type(ctx).__name__} exposes attributes outside "
            f"{list(_CONTEXT_FIELDS)}: {sorted(extra)}; features may not see anything else"
        )


def _check_outputs(pm_guess: np.ndarray, pm_weight: np.ndarray, n_rows: int) -> None:
    if pm_guess.shape != (n_rows,):
        raise ValueError(f"pm_guess has shape {pm_guess.shape}; expected ({n_rows},)")
    if pm_weight.shape != (n_rows,):
        raise ValueError(f"pm_weight has shape {pm_weight.shape}; expected ({n_rows},)")
    if n_rows and not np.isin(pm_guess, (0, 1)).all():
        raise ValueError("pm_guess must be binary")
    if n_rows and not np.isfinite(pm_weight).all():
        raise ValueError("pm_weight must be finite")


def extract_features(
    detectors_packed: np.ndarray,
    pm_guess: np.ndarray,
    pm_weight: np.ndarray,
    ctx: FeatureContext,
) -> np.ndarray:
    """Vectorised schema v1 features for a block of little-endian packed shots.

    Returns float64 ``(n, 24)`` in ``FEATURE_COLUMNS`` order for any ``n``, including 0.
    The caller bounds ``n`` (``feature_rows``); the per-block working set is dominated by
    the two dense int32 products, about ``2 * n * n_detectors * 4`` bytes.

    Conventions that a naive rewrite gets wrong: the unpack is little-endian through
    :func:`qecgen.sampling.unpack_bits` (NumPy's default reverses every byte);
    ``fired_per_round_std`` is the population value (``ddof=0``); ``round_of_max_fired``
    takes the *earliest* maximum (``argmax``); ``fired_round_center`` and
    ``fired_round_spread`` are ``0.0`` on an empty syndrome by definition, not NaN;
    ``pm_weight_per_fired`` divides by ``max(n_fired_total, 1)``.
    """
    fired = unpack_bits(detectors_packed, ctx.n_detectors)
    n_rows = fired.shape[0]
    guess = np.asarray(pm_guess)
    weight = np.asarray(pm_weight, dtype=np.float64)
    _check_outputs(guess, weight, n_rows)

    n_slices = ctx.slices.n_slices
    denominator = float(max(n_slices - 1, 1))
    t_norm = np.arange(n_slices, dtype=np.float64) / denominator
    fired_int = fired.astype(np.int32)
    # Sparse *array* products keep ndarray results; np.asarray pins the dtype anyway
    # because scipy is untyped and its upcasting rules are not part of this contract.
    counts = np.asarray((ctx.slices.membership.T @ fired_int.T).T, dtype=np.int32)
    neighbours = np.asarray((ctx.graph.adjacency @ fired_int.T).T, dtype=np.int32)

    n_total = counts.sum(axis=1).astype(np.float64)
    safe_total = np.maximum(n_total, 1.0)
    any_fired = n_total > 0
    sizes = ctx.slices.sizes
    n_first = counts[:, 0].astype(np.float64)
    n_final = counts[:, -1].astype(np.float64)
    per_round_mean = counts.mean(axis=1)
    per_round_max = counts.max(axis=1).astype(np.float64)
    per_round_std = counts.std(axis=1)
    per_round_range = per_round_max - counts.min(axis=1).astype(np.float64)
    round_of_max = np.argmax(counts, axis=1).astype(np.float64) / denominator
    n_active = (counts > 0).sum(axis=1).astype(np.float64)
    center = np.where(any_fired, (counts @ t_norm) / safe_total, 0.0)
    # Two-pass weighted variance, not E[t^2] - center^2: the one-pass form leaves a
    # ~1e-17 residual when every fired detector sits in one slice, and sqrt turns that
    # into a 1e-8 spread that the reference implementation (and the definition) put at 0.
    deviation = t_norm[np.newaxis, :] - center[:, np.newaxis]
    variance = (counts * deviation * deviation).sum(axis=1) / safe_total
    spread = np.where(any_fired, np.sqrt(variance), 0.0)

    boundary_mask = ctx.graph.boundary_adjacent
    logical_mask = ctx.graph.logical_adjacent
    n_boundary = fired[:, boundary_mask].sum(axis=1).astype(np.float64)
    n_logical = fired[:, logical_mask].sum(axis=1).astype(np.float64)
    # Each fired pair is counted from both endpoints; halving is exact because the
    # adjacency is symmetric, binary and loop-free.
    n_pairs = ((neighbours * fired_int).sum(axis=1) // 2).astype(np.float64)
    n_isolated = (fired & (neighbours == 0)).sum(axis=1).astype(np.float64)

    columns = [
        n_total,
        n_total / ctx.n_detectors,
        n_first,
        n_first / float(sizes[0]),
        n_final,
        n_final / float(sizes[-1]),
        per_round_mean,
        per_round_max,
        per_round_std,
        per_round_range,
        round_of_max,
        n_active,
        n_active / n_slices,
        center,
        spread,
        n_boundary,
        n_boundary / float(boundary_mask.sum()),
        n_logical,
        n_logical / float(logical_mask.sum()),
        n_pairs,
        n_isolated,
        weight,
        weight / safe_total,
        guess.astype(np.float64),
    ]
    if len(columns) != len(FEATURE_COLUMNS):
        raise AssertionError("feature column list and FEATURE_COLUMNS disagree in length")
    features = np.column_stack(columns).astype(np.float64, copy=False)
    return features.reshape(n_rows, len(FEATURE_COLUMNS))


def extract_features_reference(
    detector_bits: Sequence[int],
    pm_guess: int,
    pm_weight: float,
    ctx: FeatureContext,
) -> list[float]:
    """The brief's definitions in plain Python for one shot, used to spot-check the fast path.

    Deliberately shares nothing with :func:`extract_features` beyond the context: sets
    for the fired nodes, an explicit neighbour map, explicit loops for every count. It is
    slow, and that is fine; it runs on a handful of rows per dataset.
    """
    bits = [int(b) for b in detector_bits]
    if len(bits) != ctx.n_detectors:
        raise ValueError(f"expected {ctx.n_detectors} detector bits, got {len(bits)}")
    if any(b not in (0, 1) for b in bits):
        raise ValueError("detector bits must be 0 or 1")
    if pm_guess not in (0, 1):
        raise ValueError("pm_guess must be 0 or 1")
    if not math.isfinite(pm_weight):
        raise ValueError("pm_weight must be finite")

    fired = {d for d, b in enumerate(bits) if b == 1}
    slice_of = [int(s) for s in ctx.slices.slice_of_detector.tolist()]
    n_slices = ctx.slices.n_slices
    sizes = [int(s) for s in ctx.slices.sizes.tolist()]
    counts = [0] * n_slices
    for d in fired:
        counts[slice_of[d]] += 1

    rows, cols = ctx.graph.adjacency.nonzero()
    neighbours: dict[int, set[int]] = {d: set() for d in range(ctx.n_detectors)}
    for u, v in zip(rows.tolist(), cols.tolist(), strict=True):
        if u != v:
            neighbours[int(u)].add(int(v))
            neighbours[int(v)].add(int(u))
    boundary_nodes = {int(d) for d in np.flatnonzero(ctx.graph.boundary_adjacent)}
    logical_nodes = {int(d) for d in np.flatnonzero(ctx.graph.logical_adjacent)}

    n_total = len(fired)
    denominator = max(n_slices - 1, 1)
    t_norm = [t / denominator for t in range(n_slices)]
    mean = sum(counts) / n_slices
    std = math.sqrt(sum((c - mean) ** 2 for c in counts) / n_slices)
    max_count = max(counts)
    min_count = min(counts)
    round_of_max = counts.index(max_count) / denominator
    n_active = sum(1 for c in counts if c > 0)
    if n_total > 0:
        center = sum(c * t for c, t in zip(counts, t_norm, strict=True)) / n_total
        variance = sum(c * (t - center) ** 2 for c, t in zip(counts, t_norm, strict=True))
        spread = math.sqrt(variance / n_total)
    else:
        center = 0.0
        spread = 0.0

    n_boundary = len(fired & boundary_nodes)
    n_logical = len(fired & logical_nodes)
    n_pairs = 0
    n_isolated = 0
    for u in fired:
        fired_neighbours = neighbours[u] & fired
        n_pairs += sum(1 for v in fired_neighbours if u < v)
        if not fired_neighbours:
            n_isolated += 1

    return [
        float(n_total),
        n_total / ctx.n_detectors,
        float(counts[0]),
        counts[0] / sizes[0],
        float(counts[-1]),
        counts[-1] / sizes[-1],
        mean,
        float(max_count),
        std,
        float(max_count - min_count),
        round_of_max,
        float(n_active),
        n_active / n_slices,
        center,
        spread,
        float(n_boundary),
        n_boundary / len(boundary_nodes),
        float(n_logical),
        n_logical / len(logical_nodes),
        float(n_pairs),
        float(n_isolated),
        float(pm_weight),
        pm_weight / max(n_total, 1),
        float(pm_guess),
    ]
