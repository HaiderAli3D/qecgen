"""Baseline residual classifiers: can cheap features predict when PyMatching is wrong?

This is the validation experiment the brief asks for, not the deliverable. Two small
scikit-learn models (a logistic regression and a small gradient-boosted tree) are fitted on
the ``train`` rows of a published feature CSV to predict ``pm_wrong``; a flip threshold is
chosen on the ``validation`` rows; the corrected decoder is evaluated **once** on ``test``.

Three traps shape the protocol here.

* **The success criterion is the corrected logical error rate, not classification
  accuracy.** ``corrected_guess = pm_guess XOR predicted_pm_wrong`` fixes a PyMatching
  failure when the prediction is right and *creates* one when it is wrong, so the report
  counts both (``flips_correcting`` / ``flips_harmful``) and pairs them in an exact
  McNemar test. A classifier with 99% accuracy on a 1%-positive dataset is the always-zero
  rule, which changes nothing; ``always_zero_accuracy`` is reported so that number can be
  read against it.
* **"Flip nothing" is always a candidate threshold.** The threshold minimises the
  validation corrected error over the unique validation scores *plus* ``+inf``, and ties go
  to the largest threshold (fewest flips). An uninformative model therefore selects no
  flips and reports a neutral result; without that candidate the search would be forced to
  flip at least one row and the report would show harm the model never had to cause.
* **No held-out row enters a fit.** The scaler and both models see ``train`` only; the
  threshold sees ``validation`` only; ``test`` is touched exactly once, at the end. The
  input matrix is exactly :data:`~qecgen.residual.features.FEATURE_COLUMNS`, checked by
  :func:`~qecgen.residual.features.audit_feature_columns` on the CSV header, so a column
  added to the file (a drift state, an environment id) is refused rather than trained on.

scikit-learn is an optional extra (``residual`` in ``pyproject.toml``) and is imported
lazily inside :func:`run_sanity_models`; its absence raises
:class:`SanityModelUnavailableError` so the note can say "skipped" rather than the
pipeline silently publishing a dataset with no sanity report.
"""

from __future__ import annotations

import csv
import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binom

from qecgen.qa import Interval, clopper_pearson
from qecgen.residual import SCHEMA_VERSION
from qecgen.residual.features import FEATURE_COLUMNS, INTEGER_COLUMNS, audit_feature_columns
from qecgen.residual.splits import SPLIT_NAMES

__all__ = [
    "EVALUATION_SPLIT",
    "FIT_SPLIT",
    "MODEL_NAMES",
    "THRESHOLD_SPLIT",
    "FeatureTable",
    "SanityModelUnavailableError",
    "choose_threshold",
    "mcnemar_exact_p_value",
    "read_feature_table",
    "run_sanity_models",
]

FIT_SPLIT = "train"
THRESHOLD_SPLIT = "validation"
EVALUATION_SPLIT = "test"
_REQUIRED_SPLITS: tuple[str, ...] = (FIT_SPLIT, THRESHOLD_SPLIT, EVALUATION_SPLIT)

MODEL_NAMES: tuple[str, ...] = ("logistic_regression", "hist_gradient_boosting")
"""Report keys under ``models``; both are fitted with the same protocol."""

TARGET_COLUMN = "pm_wrong"


class SanityModelUnavailableError(RuntimeError):
    """scikit-learn is not installed, so no sanity model can be fitted.

    Raised rather than returning an empty report: the note must record the sanity model as
    *skipped* with this reason, never publish a dataset whose sanity section is silently
    absent as if the experiment had run and found nothing.
    """


@dataclass(frozen=True)
class FeatureTable:
    """A feature CSV loaded into arrays, with the input matrix isolated from the labels.

    ``features`` holds exactly :data:`FEATURE_COLUMNS` in order; ``truth``, ``pm_wrong``,
    ``run_id`` and ``split`` are kept apart so no code path can concatenate them into the
    model input by accident.
    """

    features: np.ndarray  # float64 (n, len(FEATURE_COLUMNS))
    pm_guess: np.ndarray  # uint8 (n,)
    truth: np.ndarray  # uint8 (n,)
    pm_wrong: np.ndarray  # uint8 (n,)
    run_id: np.ndarray  # int64 (n,)
    split: np.ndarray  # str (n,), values in SPLIT_NAMES

    def rows(self, split: str) -> np.ndarray:
        """Boolean mask of the rows belonging to ``split``."""
        mask: np.ndarray = self.split == split
        return mask


def read_feature_table(path: Path) -> FeatureTable:
    """Read a residual feature CSV (schema v1) with ``csv`` + numpy, refusing leaks.

    The header must equal ``ALL_COLUMNS`` exactly, integer columns must parse as integers
    (a ``1.0`` in ``pm_wrong`` means the writer's formatting contract broke), ``split``
    must be a known name and ``pm_wrong`` must equal ``pm_guess != truth`` on every row,
    because a corrected error rate computed from an inconsistent label is meaningless.
    """
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path}: empty feature CSV") from None
        audit_feature_columns(header)
        column_index = {name: i for i, name in enumerate(header)}
        feature_idx = [column_index[name] for name in FEATURE_COLUMNS]
        integer_idx = sorted(column_index[name] for name in INTEGER_COLUMNS)
        split_idx = column_index["split"]
        n_columns = len(header)

        feature_rows: list[list[float]] = []
        labels: list[tuple[int, int, int, int]] = []
        splits: list[str] = []
        for line_no, row in enumerate(reader, start=2):
            if len(row) != n_columns:
                raise ValueError(
                    f"{path}: line {line_no} has {len(row)} cells, expected {n_columns}"
                )
            for i in integer_idx:
                cell = row[i]
                if not cell.lstrip("-").isdigit():
                    raise ValueError(
                        f"{path}: line {line_no}: integer column {header[i]!r} holds {cell!r}"
                    )
            split_name = row[split_idx]
            if split_name not in SPLIT_NAMES:
                raise ValueError(
                    f"{path}: line {line_no}: unknown split {split_name!r}; "
                    f"allowed: {list(SPLIT_NAMES)}"
                )
            feature_rows.append([float(row[i]) for i in feature_idx])
            labels.append(
                (
                    int(row[column_index["pm_guess"]]),
                    int(row[column_index["truth"]]),
                    int(row[column_index["pm_wrong"]]),
                    int(row[column_index["run_id"]]),
                )
            )
            splits.append(split_name)

    if not feature_rows:
        raise ValueError(f"{path}: feature CSV has a header but no rows")
    features = np.asarray(feature_rows, dtype=np.float64).reshape(-1, len(FEATURE_COLUMNS))
    label_array = np.asarray(labels, dtype=np.int64).reshape(-1, 4)
    pm_guess = label_array[:, 0]
    truth = label_array[:, 1]
    pm_wrong = label_array[:, 2]
    for name, values in (("pm_guess", pm_guess), ("truth", truth), ("pm_wrong", pm_wrong)):
        if not np.all((values == 0) | (values == 1)):
            raise ValueError(f"{path}: column {name!r} is not binary")
    if not np.array_equal(pm_wrong, (pm_guess != truth).astype(np.int64)):
        bad = int(np.count_nonzero(pm_wrong != (pm_guess != truth)))
        raise ValueError(f"{path}: pm_wrong != (pm_guess != truth) on {bad} row(s)")
    if not np.all(np.isfinite(features)):
        raise ValueError(f"{path}: feature matrix contains NaN or inf")
    return FeatureTable(
        features=features,
        pm_guess=pm_guess.astype(np.uint8),
        truth=truth.astype(np.uint8),
        pm_wrong=pm_wrong.astype(np.uint8),
        run_id=label_array[:, 3],
        split=np.asarray(splits, dtype=str),
    )


def choose_threshold(scores: np.ndarray, pm_wrong: np.ndarray) -> tuple[float, int]:
    """Pick the flip threshold minimising the corrected error count on validation rows.

    Candidates are the unique scores plus ``+inf`` ("flip nothing"); a row is flipped when
    ``score >= threshold``. Ties resolve to the largest threshold, i.e. the fewest flips,
    so a model that cannot beat PyMatching selects ``+inf`` and the corrected decoder *is*
    PyMatching. Returns ``(threshold, corrected_error_count)``.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    wrong = np.asarray(pm_wrong, dtype=np.int64).ravel()
    if scores.shape != wrong.shape:
        raise ValueError(f"scores {scores.shape} and pm_wrong {wrong.shape} differ in length")
    if scores.size == 0:
        raise ValueError("threshold selection needs at least one validation row")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")

    # Sort by descending score; flipping the m highest-scoring rows removes the PyMatching
    # failures among them and adds the successes among them. A threshold equal to a score
    # value flips every row with that score, so only prefixes ending at a boundary between
    # distinct scores are valid candidates (plus m=0 for +inf).
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_wrong = wrong[order]
    cum_wrong = np.concatenate(([0], np.cumsum(sorted_wrong)))
    cum_right = np.concatenate(([0], np.cumsum(1 - sorted_wrong)))
    boundaries = np.flatnonzero(np.diff(sorted_scores) != 0) + 1
    candidates = np.concatenate(([0], boundaries, [scores.size])).astype(np.int64)
    total_wrong = int(cum_wrong[-1])
    errors = total_wrong - cum_wrong[candidates] + cum_right[candidates]
    best = int(candidates[int(np.argmin(errors))])  # argmin: first minimum = fewest flips
    best_errors = int(errors[int(np.argmin(errors))])
    if best == 0:
        return math.inf, best_errors
    return float(sorted_scores[best - 1]), best_errors


def mcnemar_exact_p_value(flips_correcting: int, flips_harmful: int) -> float:
    """Two-sided exact McNemar (paired binomial) p-value on the discordant flips.

    Every flipped test row either corrects a PyMatching failure or creates one, and those
    are the only rows where the corrected decoder and PyMatching disagree, so the paired
    comparison reduces to ``Binomial(b + c, 1/2)`` on the smaller count. Exact rather than
    the chi-square approximation because the counts here are often tiny (a d=25 dataset
    may have a handful of flips), where the approximation is meaningless.
    """
    if flips_correcting < 0 or flips_harmful < 0:
        raise ValueError("flip counts must be non-negative")
    n = flips_correcting + flips_harmful
    if n == 0:
        return 1.0
    smaller = min(flips_correcting, flips_harmful)
    p = 2.0 * float(binom.cdf(smaller, n, 0.5))
    return min(1.0, p)


def _interval_dict(interval: Interval) -> dict[str, Any]:
    return dataclasses.asdict(interval)


def _prevalence(pm_wrong: np.ndarray) -> float:
    return float(np.mean(pm_wrong)) if pm_wrong.size else 0.0


def _binary_classes(values: np.ndarray) -> int:
    return int(np.unique(values).size)


def _import_sklearn() -> tuple[Any, Any, Any, Any]:
    """Return ``(sklearn, StandardScaler, LogisticRegression, HistGradientBoostingClassifier)``.

    Imported here rather than at module scope so that decoding, feature extraction and
    validation never depend on an optional extra; the failure is an explicit error with the
    install line, never a silent skip.
    """
    try:
        import sklearn
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SanityModelUnavailableError(
            "scikit-learn is not installed; the sanity residual model is skipped. "
            'Install the optional extra with `pip install -e ".[residual]"`.'
        ) from exc
    return sklearn, StandardScaler, LogisticRegression, HistGradientBoostingClassifier


def _fit_and_score(
    model_name: str,
    table: FeatureTable,
    train: np.ndarray,
    seed: int,
    sklearn_classes: tuple[Any, Any, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one model on the train rows only and return a score for every row of the table.

    The scaler and the model see ``table.features[train]`` and ``table.pm_wrong[train]``
    and nothing else; scoring the whole table afterwards is a pure transform.
    """
    scaler_cls, logistic_cls, hgb_cls = sklearn_classes
    x_train = table.features[train]
    y_train = table.pm_wrong[train].astype(np.int64)
    if model_name == "logistic_regression":
        scaler = scaler_cls().fit(x_train)
        model = logistic_cls(max_iter=2000, class_weight="balanced", random_state=seed)
        model.fit(scaler.transform(x_train), y_train)
        scores = np.asarray(
            model.predict_proba(scaler.transform(table.features))[:, 1], dtype=np.float64
        )
        params: dict[str, Any] = {
            "preprocessing": "StandardScaler (fit on train)",
            "estimator": "LogisticRegression",
            "max_iter": 2000,
            "class_weight": "balanced",
            "random_state": seed,
            "n_iter": int(np.asarray(model.n_iter_).ravel()[0]),
        }
    elif model_name == "hist_gradient_boosting":
        # early_stopping must be an explicit False. sklearn's default 'auto' switches it
        # ON above 10,000 samples (every real dataset here) and then carves a *stratified*
        # validation set out of the train rows, which raises "The least populated class
        # in y has only 1 member" when train holds exactly one pm_wrong == 1 row -- a
        # realistic state for the d=25 dataset -- after the whole decode has run. With
        # it off the model runs the recorded max_iter on train rows only, so
        # `n_iter == max_iter` and the params below describe what actually ran.
        model = hgb_cls(
            max_iter=300,
            learning_rate=0.05,
            random_state=seed,
            class_weight="balanced",
            early_stopping=False,
        )
        model.fit(x_train, y_train)
        scores = np.asarray(model.predict_proba(table.features)[:, 1], dtype=np.float64)
        params = {
            "preprocessing": "none (trees are scale-invariant)",
            "estimator": "HistGradientBoostingClassifier",
            "max_iter": 300,
            "learning_rate": 0.05,
            "class_weight": "balanced",
            "early_stopping": False,
            "random_state": seed,
            "n_iter": int(model.n_iter_),
        }
    else:
        raise ValueError(f"unknown sanity model {model_name!r}; known: {list(MODEL_NAMES)}")
    if scores.shape != (table.features.shape[0],):
        raise ValueError(f"{model_name}: score shape {scores.shape} does not match the table")
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"{model_name}: non-finite scores")
    return scores, params


def _auc_metrics(y_true: np.ndarray, scores: np.ndarray) -> tuple[float | None, float | None]:
    """ROC AUC and average precision, or ``None`` when the split has one class."""
    if _binary_classes(y_true) < 2:
        return None, None
    from sklearn.metrics import average_precision_score, roc_auc_score

    return float(roc_auc_score(y_true, scores)), float(average_precision_score(y_true, scores))


def _evaluate_model(
    model_name: str,
    table: FeatureTable,
    scores: np.ndarray,
    params: dict[str, Any],
    fitted: bool,
    sklearn_version: str,
) -> dict[str, Any]:
    """Choose the threshold on validation, then evaluate exactly once on test."""
    masks = {name: table.rows(name) for name in _REQUIRED_SPLITS}
    y = {name: table.pm_wrong[mask].astype(np.int64) for name, mask in masks.items()}
    s = {name: scores[mask] for name, mask in masks.items()}
    degenerate = {name: _binary_classes(y[name]) < 2 for name in _REQUIRED_SPLITS}

    threshold, val_corrected_failures = choose_threshold(s[THRESHOLD_SPLIT], y[THRESHOLD_SPLIT])
    flip_nothing = math.isinf(threshold)
    val_flips = 0 if flip_nothing else int(np.count_nonzero(s[THRESHOLD_SPLIT] >= threshold))

    y_test = y[EVALUATION_SPLIT]
    predicted = (
        np.zeros_like(y_test)
        if flip_nothing
        else (s[EVALUATION_SPLIT] >= threshold).astype(np.int64)
    )
    corrected_wrong = y_test ^ predicted  # corrected_guess != truth  <=>  pm_wrong XOR flip
    tp = int(np.count_nonzero((predicted == 1) & (y_test == 1)))
    fp = int(np.count_nonzero((predicted == 1) & (y_test == 0)))
    fn = int(np.count_nonzero((predicted == 0) & (y_test == 1)))
    tn = int(np.count_nonzero((predicted == 0) & (y_test == 0)))
    n_test = int(y_test.size)
    if tp + fp + fn + tn != n_test:
        raise AssertionError("confusion matrix does not partition the test rows")

    recall_pos = tp / (tp + fn) if tp + fn > 0 else None
    specificity = tn / (tn + fp) if tn + fp > 0 else None
    precision_pos = tp / (tp + fp) if tp + fp > 0 else None
    present = [r for r in (recall_pos, specificity) if r is not None]
    balanced_accuracy = float(np.mean(present)) if present else None
    roc_auc, pr_auc = _auc_metrics(y_test, s[EVALUATION_SPLIT])

    pm_failures = int(y_test.sum())
    corrected_failures = int(corrected_wrong.sum())
    if corrected_failures != pm_failures - tp + fp:
        raise AssertionError("corrected failure count disagrees with the flip bookkeeping")
    pm_interval = clopper_pearson(pm_failures, n_test)
    corrected_interval = clopper_pearson(corrected_failures, n_test)
    absolute_change = corrected_interval.point - pm_interval.point
    relative_change = absolute_change / pm_interval.point if pm_interval.point > 0 else None

    return {
        "fitted": fitted,
        "degenerate": degenerate,
        "model": params,
        "fit_split": FIT_SPLIT,
        "threshold_split": THRESHOLD_SPLIT,
        "evaluation_split": EVALUATION_SPLIT,
        "feature_columns": list(FEATURE_COLUMNS),
        "target": TARGET_COLUMN,
        "sklearn_version": sklearn_version,
        "n_rows": {name: int(mask.sum()) for name, mask in masks.items()},
        "positive_prevalence": {name: _prevalence(y[name]) for name in _REQUIRED_SPLITS},
        "always_zero_accuracy": {name: 1.0 - _prevalence(y[name]) for name in _REQUIRED_SPLITS},
        "threshold": None if flip_nothing else threshold,
        "flip_nothing": flip_nothing,
        "flip_rule": "flip pm_guess when score >= threshold; threshold null means flip nothing",
        "validation": {
            "n_rows": int(masks[THRESHOLD_SPLIT].sum()),
            "pm_failures": int(y[THRESHOLD_SPLIT].sum()),
            "corrected_failures": val_corrected_failures,
            "n_flips": val_flips,
            "n_threshold_candidates": int(np.unique(s[THRESHOLD_SPLIT]).size) + 1,
        },
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "pm_test_error_rate": _interval_dict(pm_interval),
        "corrected_test_error_rate": _interval_dict(corrected_interval),
        "absolute_change": absolute_change,
        "relative_change": relative_change,
        "n_flips": tp + fp,
        "flips_correcting": tp,
        "flips_harmful": fp,
        "mcnemar_p_value": mcnemar_exact_p_value(tp, fp),
        "model_name": model_name,
    }


def run_sanity_models(features_csv: Path, *, seed: int) -> dict[str, Any]:
    """Fit both baseline models on ``train``, threshold on ``validation``, evaluate on ``test``.

    Returns the JSON-serialisable sanity report (``<name>_sanity.json``). Every number is
    finite (an infinite threshold is recorded as ``null`` + ``flip_nothing``) so the report
    survives ``json.dumps(..., allow_nan=False)``. A split with a single class marks the
    model ``degenerate`` for that split and reports ``None`` for the AUCs instead of
    failing: a d=25 dataset at p=0.005 may legitimately have no positives in a split.
    """
    sklearn, scaler_cls, logistic_cls, hgb_cls = _import_sklearn()
    sklearn_version = str(sklearn.__version__)
    table = read_feature_table(Path(features_csv))

    counts = {name: int(np.count_nonzero(table.split == name)) for name in SPLIT_NAMES}
    missing = [name for name in _REQUIRED_SPLITS if counts[name] == 0]
    if missing:
        raise ValueError(
            f"{features_csv}: no rows in split(s) {missing}; the sanity protocol needs "
            f"{list(_REQUIRED_SPLITS)}"
        )
    train = table.rows(FIT_SPLIT)
    train_classes = _binary_classes(table.pm_wrong[train])

    models: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        if train_classes < 2:
            # Nothing to fit: a constant score makes every candidate threshold equivalent
            # and the tie rule selects "flip nothing", which is the honest answer.
            scores = np.zeros(table.features.shape[0], dtype=np.float64)
            params: dict[str, Any] = {
                "estimator": None,
                "reason": "train split has a single pm_wrong class; model not fitted",
            }
            fitted = False
        else:
            scores, params = _fit_and_score(
                model_name, table, train, seed, (scaler_cls, logistic_cls, hgb_cls)
            )
            fitted = True
        models[model_name] = _evaluate_model(
            model_name, table, scores, params, fitted, sklearn_version
        )

    n_test = counts[EVALUATION_SPLIT]
    pm_test_failures = int(table.pm_wrong[table.rows(EVALUATION_SPLIT)].sum())
    return {
        "schema_version": SCHEMA_VERSION,
        # The file name only: the pipeline reads the CSV inside a run.staged() scratch
        # directory, so the absolute path would name a `.qecgen-partial-*` directory that
        # no longer exists once published and would differ between identical builds.
        "features_csv": Path(features_csv).name,
        "target": TARGET_COLUMN,
        "feature_columns": list(FEATURE_COLUMNS),
        "n_features": len(FEATURE_COLUMNS),
        "fit_split": FIT_SPLIT,
        "threshold_split": THRESHOLD_SPLIT,
        "evaluation_split": EVALUATION_SPLIT,
        "seed": seed,
        "sklearn_version": sklearn_version,
        "n_rows": counts,
        "positive_prevalence": {
            name: _prevalence(table.pm_wrong[table.rows(name)]) for name in _REQUIRED_SPLITS
        },
        "always_zero_accuracy": {
            name: 1.0 - _prevalence(table.pm_wrong[table.rows(name)]) for name in _REQUIRED_SPLITS
        },
        "pm_test_error_rate": _interval_dict(clopper_pearson(pm_test_failures, n_test)),
        "threshold_rule": (
            "minimise validation corrected error mean(pm_wrong XOR (score >= t)) over the "
            "unique validation scores plus +inf (flip nothing); ties -> largest t"
        ),
        "models": models,
    }
