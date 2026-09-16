"""Sanity residual classifiers: fit on train, threshold on validation, evaluate once on test.

The sanity model is a validation experiment, not a deliverable, so what these tests pin is
the *protocol* rather than any accuracy number: the input matrix is exactly the approved
feature columns, no validation or test row enters a fit, the threshold search includes
"flip nothing" so an uninformative model reports a neutral result instead of a harmful
one, and a missing scikit-learn is an explicit error rather than a silently empty report.
"""

from __future__ import annotations

import json
import math
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from qecgen.residual.features import ALL_COLUMNS, FEATURE_COLUMNS, INTEGER_COLUMNS
from qecgen.residual.sanity import (
    SanityModelUnavailableError,
    choose_threshold,
    mcnemar_exact_p_value,
    read_feature_table,
    run_sanity_models,
)
from qecgen.residual.splits import SPLIT_NAMES, assign_splits

pytest.importorskip("sklearn")

REPO_ROOT = Path(__file__).resolve().parent.parent
FRACTIONS = {"train": 0.7, "validation": 0.15, "test": 0.15}
MODEL_KEYS = ("logistic_regression", "hist_gradient_boosting")
PER_MODEL_KEYS = {
    "positive_prevalence",
    "always_zero_accuracy",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "precision_pos",
    "recall_pos",
    "confusion_matrix",
    "pm_test_error_rate",
    "corrected_test_error_rate",
    "absolute_change",
    "relative_change",
    "n_flips",
    "flips_correcting",
    "flips_harmful",
    "mcnemar_p_value",
    "threshold",
    "flip_nothing",
    "validation",
    "fit_split",
    "threshold_split",
    "feature_columns",
    "sklearn_version",
    "degenerate",
}


def _synthetic_rows(n_rows: int, seed: int, *, rule: str) -> dict[str, np.ndarray]:
    """Build a table with plausible feature ranges and a planted ``pm_wrong`` rule.

    ``rule="linear"`` makes ``pm_wrong`` a threshold on ``pm_weight_per_fired`` (recoverable
    by a linear model); ``rule="random"`` draws labels independently of every feature;
    ``rule="never"`` plants no positives at all so every split is single-class.
    """
    rng = np.random.default_rng(seed)
    n_detectors = 24
    n_slices = 4
    n_total = rng.integers(0, 12, size=n_rows)
    first = np.minimum(rng.integers(0, 4, size=n_rows), n_total)
    final = np.minimum(rng.integers(0, 4, size=n_rows), n_total - first)
    pm_weight = n_total * rng.uniform(1.0, 6.0, size=n_rows)
    pm_guess = rng.integers(0, 2, size=n_rows)
    weight_per_fired = pm_weight / np.maximum(n_total, 1)
    if rule == "linear":
        pm_wrong = (weight_per_fired > 4.5).astype(np.int64)
    elif rule == "random":
        pm_wrong = (rng.uniform(size=n_rows) < 0.2).astype(np.int64)
    elif rule == "never":
        pm_wrong = np.zeros(n_rows, dtype=np.int64)
    else:
        raise AssertionError(rule)
    truth = pm_guess ^ pm_wrong
    per_round_mean = n_total / n_slices
    columns: dict[str, np.ndarray] = {
        "n_fired_total": n_total,
        "frac_fired": n_total / n_detectors,
        "n_fired_first_round": first,
        "frac_fired_first_round": first / 4,
        "n_fired_final_round": final,
        "frac_fired_final_round": final / 4,
        "fired_per_round_mean": per_round_mean,
        "fired_per_round_max": np.minimum(n_total, 8),
        "fired_per_round_std": rng.uniform(0, 2, size=n_rows),
        "fired_per_round_range": np.minimum(n_total, 8),
        "round_of_max_fired": rng.uniform(size=n_rows),
        "n_active_rounds": np.minimum(n_total, n_slices),
        "frac_active_rounds": np.minimum(n_total, n_slices) / n_slices,
        "fired_round_center": rng.uniform(size=n_rows),
        "fired_round_spread": rng.uniform(0, 0.5, size=n_rows),
        "n_fired_boundary_adjacent": n_total,
        "frac_fired_boundary_adjacent": n_total / n_detectors,
        "n_fired_logical_edge_adjacent": np.minimum(n_total, 6),
        "frac_fired_logical_edge_adjacent": np.minimum(n_total, 6) / 6,
        "n_fired_neighbor_pairs": rng.integers(0, 5, size=n_rows),
        "n_fired_isolated": rng.integers(0, 5, size=n_rows),
        "pm_weight": pm_weight,
        "pm_weight_per_fired": weight_per_fired,
        "pm_guess": pm_guess,
        "truth": truth,
        "pm_wrong": pm_wrong,
        "run_id": np.arange(n_rows),
    }
    return columns


def _write_csv(
    path: Path,
    columns: dict[str, np.ndarray],
    split_codes: np.ndarray,
    *,
    header: tuple[str, ...] = ALL_COLUMNS,
) -> None:
    n_rows = len(split_codes)
    lines = [",".join(header)]
    for i in range(n_rows):
        cells: list[str] = []
        for name in header:
            if name == "split":
                cells.append(SPLIT_NAMES[int(split_codes[i])])
            elif name in INTEGER_COLUMNS:
                cells.append(str(int(columns[name][i])))
            else:
                cells.append(repr(float(columns[name][i])))
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_csv(tmp_path: Path, *, rule: str, n_rows: int = 500, seed: int = 7) -> Path:
    columns = _synthetic_rows(n_rows, seed, rule=rule)
    split_codes = assign_splits(n_rows, "seeded_permutation", FRACTIONS, seed=20260915)
    path = tmp_path / "synthetic_features.csv"
    _write_csv(path, columns, split_codes)
    return path


def _assert_json_safe(payload: dict[str, Any]) -> None:
    # The report is written next to the dataset with allow_nan=False; an inf threshold
    # or a NaN AUC would make the publish step fail after all the decoding is done.
    json.dumps(payload, allow_nan=False, sort_keys=True)


class TestThresholdChoice:
    def test_separable_scores_pick_the_lowest_positive_score(self) -> None:
        scores = np.array([0.9, 0.8, 0.1, 0.05])
        wrong = np.array([1, 1, 0, 0], dtype=np.uint8)
        threshold, errors = choose_threshold(scores, wrong)
        assert threshold == 0.8
        assert errors == 0

    def test_uninformative_scores_flip_nothing(self) -> None:
        scores = np.array([0.9, 0.5])
        wrong = np.array([0, 1], dtype=np.uint8)
        # m=0 -> 1 error; flipping row 0 -> 2 errors; flipping both -> 1 error. The tie
        # between "flip nothing" and "flip everything" resolves to the largest threshold.
        threshold, errors = choose_threshold(scores, wrong)
        assert math.isinf(threshold) and threshold > 0
        assert errors == 1

    def test_tied_scores_form_one_candidate(self) -> None:
        scores = np.array([0.9, 0.5, 0.5, 0.2])
        wrong = np.array([1, 0, 0, 1], dtype=np.uint8)
        threshold, errors = choose_threshold(scores, wrong)
        assert threshold == 0.9
        assert errors == 1

    def test_all_wrong_flips_everything(self) -> None:
        scores = np.array([0.3, 0.2, 0.1])
        wrong = np.array([1, 1, 1], dtype=np.uint8)
        threshold, errors = choose_threshold(scores, wrong)
        assert threshold == 0.1
        assert errors == 0

    def test_empty_validation_refused(self) -> None:
        with pytest.raises(ValueError, match="validation"):
            choose_threshold(np.array([]), np.array([], dtype=np.uint8))


class TestMcNemar:
    def test_no_discordant_pairs_is_one(self) -> None:
        assert mcnemar_exact_p_value(0, 0) == 1.0

    def test_five_versus_zero(self) -> None:
        assert mcnemar_exact_p_value(5, 0) == pytest.approx(2 * 0.5**5)
        assert mcnemar_exact_p_value(0, 5) == pytest.approx(2 * 0.5**5)

    def test_balanced_is_one(self) -> None:
        assert mcnemar_exact_p_value(3, 3) == pytest.approx(1.0)

    def test_negative_refused(self) -> None:
        with pytest.raises(ValueError):
            mcnemar_exact_p_value(-1, 2)


class TestFeatureTable:
    def test_extra_column_refused(self, tmp_path: Path) -> None:
        columns = _synthetic_rows(20, 1, rule="linear")
        columns["drift_state"] = np.zeros(20, dtype=np.int64)
        codes = assign_splits(20, "seeded_permutation", FRACTIONS, seed=1)
        path = tmp_path / "leaky.csv"
        _write_csv(path, columns, codes, header=(*ALL_COLUMNS, "drift_state"))
        with pytest.raises(ValueError):
            read_feature_table(path)

    def test_inconsistent_pm_wrong_refused(self, tmp_path: Path) -> None:
        columns = _synthetic_rows(20, 1, rule="linear")
        columns["pm_wrong"] = 1 - columns["pm_wrong"]
        codes = assign_splits(20, "seeded_permutation", FRACTIONS, seed=1)
        path = tmp_path / "bad.csv"
        _write_csv(path, columns, codes)
        with pytest.raises(ValueError, match="pm_wrong"):
            read_feature_table(path)

    def test_matrix_is_exactly_the_feature_columns(self, tmp_path: Path) -> None:
        path = _make_csv(tmp_path, rule="linear", n_rows=40)
        table = read_feature_table(path)
        assert table.features.shape == (40, len(FEATURE_COLUMNS))
        assert table.features.dtype == np.float64
        assert set(np.unique(table.split).tolist()) <= set(SPLIT_NAMES)


class TestRunSanityModels:
    def test_linear_rule_is_recovered_and_lowers_test_error(self, tmp_path: Path) -> None:
        path = _make_csv(tmp_path, rule="linear")
        report = run_sanity_models(path, seed=0)
        _assert_json_safe(report)
        assert report["fit_split"] == "train"
        assert report["threshold_split"] == "validation"
        assert report["feature_columns"] == list(FEATURE_COLUMNS)
        assert isinstance(report["sklearn_version"], str)
        # The published report must not embed the staging directory the pipeline read
        # the CSV from: that path no longer exists after the commit and differs between
        # otherwise identical builds.
        assert report["features_csv"] == path.name
        assert set(report["models"]) == set(MODEL_KEYS)
        hgb = report["models"]["hist_gradient_boosting"]["model"]
        assert hgb["early_stopping"] is False
        assert hgb["n_iter"] == hgb["max_iter"] == 300
        for key in MODEL_KEYS:
            model = report["models"][key]
            assert set(model) >= PER_MODEL_KEYS, PER_MODEL_KEYS - set(model)
            assert model["fit_split"] == "train"
            assert model["threshold_split"] == "validation"
            assert model["feature_columns"] == list(FEATURE_COLUMNS)
            assert model["degenerate"] == {"train": False, "validation": False, "test": False}
            pm_rate = model["pm_test_error_rate"]["point"]
            corrected = model["corrected_test_error_rate"]["point"]
            assert corrected < pm_rate
            assert model["n_flips"] > 0
            assert model["flips_correcting"] > model["flips_harmful"]
            assert model["absolute_change"] == pytest.approx(corrected - pm_rate)
            assert model["relative_change"] == pytest.approx((corrected - pm_rate) / pm_rate)
            assert model["roc_auc"] > 0.9
            assert model["pr_auc"] > 0.9
            assert model["recall_pos"] > 0.5
            assert 0.0 <= model["mcnemar_p_value"] <= 1.0
            cm = model["confusion_matrix"]
            assert cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"] == report["n_rows"]["test"]
            assert model["n_flips"] == cm["tp"] + cm["fp"]
            assert model["flips_correcting"] == cm["tp"]
            assert model["flips_harmful"] == cm["fp"]
            assert model["flip_nothing"] is False
            assert isinstance(model["threshold"], float)

    def test_random_labels_are_reported_neutrally(self, tmp_path: Path) -> None:
        path = _make_csv(tmp_path, rule="random")
        report = run_sanity_models(path, seed=0)
        _assert_json_safe(report)
        for key in MODEL_KEYS:
            model = report["models"][key]
            pm_failures = model["pm_test_error_rate"]["successes"]
            corrected_failures = model["corrected_test_error_rate"]["successes"]
            assert (
                corrected_failures
                == pm_failures - model["flips_correcting"] + model["flips_harmful"]
            )
            if model["flip_nothing"]:
                assert model["threshold"] is None
                assert model["n_flips"] == 0
                assert corrected_failures == pm_failures
                assert model["mcnemar_p_value"] == 1.0
            # Whatever the threshold, validation corrected error never exceeds PyMatching's:
            # "flip nothing" is always a candidate, so the chosen rule cannot be worse there.
            val = model["validation"]
            assert val["corrected_failures"] <= val["pm_failures"]

    def test_single_class_splits_are_degenerate_not_crashes(self, tmp_path: Path) -> None:
        path = _make_csv(tmp_path, rule="never")
        report = run_sanity_models(path, seed=0)
        _assert_json_safe(report)
        for key in MODEL_KEYS:
            model = report["models"][key]
            assert model["degenerate"] == {"train": True, "validation": True, "test": True}
            assert model["roc_auc"] is None
            assert model["pr_auc"] is None
            assert model["n_flips"] == 0
            assert model["flip_nothing"] is True
            assert model["corrected_test_error_rate"]["point"] == 0.0
            assert model["relative_change"] is None

    def test_single_train_positive_above_ten_thousand_rows_fits(self, tmp_path: Path) -> None:
        # sklearn's HistGradientBoostingClassifier default early_stopping='auto' turns
        # early stopping ON above 10,000 samples and then does a *stratified* internal
        # train/validation split, which raises "The least populated class in y has only
        # 1 member" when the train split holds exactly one pm_wrong == 1 row. That is a
        # realistic state for the d=25 dataset (pilot 0/2000 wrong) and would abort the
        # publish after the full decode. The 500-row tests never reach that code path,
        # so this one plants a single train positive in a 16,000-row table (11,200 train).
        n_rows = 16_000
        columns = _synthetic_rows(n_rows, 5, rule="never")
        split_codes = assign_splits(n_rows, "seeded_permutation", FRACTIONS, seed=20260915)
        train_rows = np.flatnonzero(split_codes == SPLIT_NAMES.index("train"))
        assert train_rows.size > 10_000
        planted = int(train_rows[0])
        columns["pm_wrong"][planted] = 1
        columns["truth"][planted] = columns["pm_guess"][planted] ^ 1
        path = tmp_path / "one_positive.csv"
        _write_csv(path, columns, split_codes)

        report = run_sanity_models(path, seed=0)
        _assert_json_safe(report)
        assert report["n_rows"]["train"] > 10_000
        for key in MODEL_KEYS:
            model = report["models"][key]
            assert model["fitted"] is True
            assert model["degenerate"] == {"train": False, "validation": True, "test": True}
            assert model["flip_nothing"] is True
            assert model["n_flips"] == 0
        hgb = report["models"]["hist_gradient_boosting"]["model"]
        assert hgb["estimator"] == "HistGradientBoostingClassifier"
        assert hgb["early_stopping"] is False
        assert hgb["n_iter"] == 300

    def test_same_seed_same_report(self, tmp_path: Path) -> None:
        path = _make_csv(tmp_path, rule="random", seed=3)
        first = run_sanity_models(path, seed=11)
        second = run_sanity_models(path, seed=11)
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    def test_missing_split_refused(self, tmp_path: Path) -> None:
        columns = _synthetic_rows(30, 1, rule="linear")
        codes = np.full(30, 1, dtype=np.int8)  # everything is train
        path = tmp_path / "train_only.csv"
        _write_csv(path, columns, codes)
        with pytest.raises(ValueError, match="validation"):
            run_sanity_models(path, seed=0)

    def test_sklearn_absent_is_explicit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _make_csv(tmp_path, rule="linear", n_rows=60)
        for name in [m for m in sys.modules if m == "sklearn" or m.startswith("sklearn.")]:
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.setitem(sys.modules, "sklearn", None)
        with pytest.raises(SanityModelUnavailableError, match="scikit-learn"):
            run_sanity_models(path, seed=0)


class TestDependencyPins:
    def test_residual_extra_and_mypy_override(self) -> None:
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        extras = pyproject["project"]["optional-dependencies"]
        assert extras["residual"] == ["scikit-learn==1.7.2"]
        assert "scikit-learn" not in " ".join(pyproject["project"]["dependencies"])
        modules = {m for o in pyproject["tool"]["mypy"]["overrides"] for m in o["module"]}
        assert {"sklearn", "sklearn.*"} <= modules
