"""Deterministic split assignment for the residual datasets.

The split column is the one thing that keeps a sanity model honest: a wrong boundary or a
non-deterministic permutation lets validation rows leak into a fit without any other check
noticing. These tests pin the exact block sizes for the deliverable row counts and the
seed contract of each method.
"""

from __future__ import annotations

import numpy as np
import pytest

from qecgen.residual.splits import (
    SPLIT_CODES,
    SPLIT_METHODS,
    SPLIT_NAMES,
    assign_splits,
    split_boundaries,
)

SIMULATED = {"train": 0.7, "validation": 0.15, "test": 0.15}
WILLOW = {"train": 0.6, "validation": 0.2, "test": 0.2}


def _sizes(codes: np.ndarray) -> dict[str, int]:
    counts = np.bincount(codes.astype(np.int64), minlength=len(SPLIT_NAMES))
    return {name: int(counts[SPLIT_CODES[name]]) for name in SPLIT_NAMES}


class TestConstants:
    def test_codes_and_names_agree(self) -> None:
        assert SPLIT_NAMES == ("calibration", "train", "validation", "test")
        assert [SPLIT_CODES[name] for name in SPLIT_NAMES] == [0, 1, 2, 3]
        assert SPLIT_CODES["calibration"] == 0, "code 0 is reserved for calibration"

    def test_methods(self) -> None:
        assert SPLIT_METHODS == ("seeded_permutation", "contiguous_blocks")


class TestBoundaries:
    @pytest.mark.parametrize("n_rows", [100, 300000, 304000])
    def test_70_15_15_exact(self, n_rows: int) -> None:
        names, bounds = split_boundaries(n_rows, SIMULATED)
        assert names == ("train", "validation", "test")
        assert bounds == (
            {100: 70, 300000: 210000, 304000: 212800}[n_rows],
            {100: 85, 300000: 255000, 304000: 258400}[n_rows],
            n_rows,
        )

    def test_60_20_20_exact(self) -> None:
        names, bounds = split_boundaries(50000, WILLOW)
        assert names == ("train", "validation", "test")
        assert bounds == (30000, 40000, 50000)

    def test_last_boundary_forced_to_n_rows(self) -> None:
        # Fractions that sum to 1 within tolerance but whose float cumsum can fall short.
        fractions = {"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3}
        _, bounds = split_boundaries(7, fractions)
        assert bounds[-1] == 7

    def test_order_follows_split_names_not_mapping_order(self) -> None:
        names, bounds = split_boundaries(100, {"test": 0.15, "train": 0.7, "validation": 0.15})
        assert names == ("train", "validation", "test")
        assert bounds == (70, 85, 100)

    def test_calibration_block_comes_first(self) -> None:
        names, bounds = split_boundaries(
            100, {"calibration": 0.1, "train": 0.6, "validation": 0.15, "test": 0.15}
        )
        assert names == SPLIT_NAMES
        assert bounds == (10, 70, 85, 100)


class TestSeededPermutation:
    def test_deterministic_for_same_seed(self) -> None:
        a = assign_splits(1000, "seeded_permutation", SIMULATED, seed=20260915)
        b = assign_splits(1000, "seeded_permutation", SIMULATED, seed=20260915)
        assert np.array_equal(a, b)

    def test_different_seed_differs(self) -> None:
        a = assign_splits(1000, "seeded_permutation", SIMULATED, seed=20260915)
        b = assign_splits(1000, "seeded_permutation", SIMULATED, seed=20260916)
        assert not np.array_equal(a, b)

    def test_exact_block_sizes_n100(self) -> None:
        codes = assign_splits(100, "seeded_permutation", SIMULATED, seed=1)
        assert codes.dtype == np.int8
        assert codes.shape == (100,)
        assert _sizes(codes) == {"calibration": 0, "train": 70, "validation": 15, "test": 15}

    @pytest.mark.parametrize("n_rows", [300000, 304000])
    def test_exact_block_sizes_deliverables(self, n_rows: int) -> None:
        codes = assign_splits(n_rows, "seeded_permutation", SIMULATED, seed=20260915)
        assert _sizes(codes) == {
            "calibration": 0,
            "train": round(0.7 * n_rows),
            "validation": round(0.15 * n_rows),
            "test": round(0.15 * n_rows),
        }

    def test_every_row_assigned_exactly_once(self) -> None:
        codes = assign_splits(257, "seeded_permutation", SIMULATED, seed=3)
        assert codes.shape == (257,)
        assert set(np.unique(codes).tolist()) <= set(SPLIT_CODES.values())
        assert sum(_sizes(codes).values()) == 257

    def test_is_a_permutation_not_a_block_layout(self) -> None:
        codes = assign_splits(100, "seeded_permutation", SIMULATED, seed=7)
        assert not np.array_equal(codes, np.sort(codes))

    def test_matches_documented_rng_contract(self) -> None:
        """The permutation is default_rng(SeedSequence(seed)).permutation(n), no more.

        A resume or a re-run on another machine must reproduce the same column, so the
        RNG construction is part of the contract rather than an implementation detail.
        """
        seed = 20260915
        perm = np.random.default_rng(np.random.SeedSequence(seed)).permutation(100)
        expected = np.empty(100, dtype=np.int8)
        expected[perm[:70]] = SPLIT_CODES["train"]
        expected[perm[70:85]] = SPLIT_CODES["validation"]
        expected[perm[85:]] = SPLIT_CODES["test"]
        codes = assign_splits(100, "seeded_permutation", SIMULATED, seed=seed)
        assert np.array_equal(codes, expected)

    @pytest.mark.parametrize("n_rows", list(range(20, 120)))
    def test_test_block_never_empty(self, n_rows: int) -> None:
        codes = assign_splits(n_rows, "seeded_permutation", SIMULATED, seed=0)
        assert _sizes(codes)["test"] >= 1

    def test_seed_required(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            assign_splits(100, "seeded_permutation", SIMULATED, seed=None)

    def test_bool_seed_refused(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            assign_splits(100, "seeded_permutation", SIMULATED, seed=True)


class TestContiguousBlocks:
    def test_non_decreasing_codes(self) -> None:
        codes = assign_splits(50000, "contiguous_blocks", WILLOW, seed=None)
        assert codes.dtype == np.int8
        assert np.all(np.diff(codes.astype(np.int16)) >= 0)

    def test_exact_willow_ranges(self) -> None:
        codes = assign_splits(50000, "contiguous_blocks", WILLOW, seed=None)
        assert np.all(codes[:30000] == SPLIT_CODES["train"])
        assert np.all(codes[30000:40000] == SPLIT_CODES["validation"])
        assert np.all(codes[40000:] == SPLIT_CODES["test"])
        assert _sizes(codes) == {
            "calibration": 0,
            "train": 30000,
            "validation": 10000,
            "test": 10000,
        }

    def test_calibration_block_first_in_row_order(self) -> None:
        fractions = {"calibration": 0.2, "train": 0.4, "validation": 0.2, "test": 0.2}
        codes = assign_splits(100, "contiguous_blocks", fractions, seed=None)
        assert np.all(codes[:20] == SPLIT_CODES["calibration"])
        assert np.all(codes[20:60] == SPLIT_CODES["train"])
        assert np.all(codes[60:80] == SPLIT_CODES["validation"])
        assert np.all(codes[80:] == SPLIT_CODES["test"])

    def test_seed_forbidden(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            assign_splits(100, "contiguous_blocks", WILLOW, seed=0)


class TestValidation:
    def test_unknown_method_refused(self) -> None:
        with pytest.raises(ValueError, match="method"):
            assign_splits(100, "random", SIMULATED, seed=0)

    def test_unknown_split_name_refused(self) -> None:
        with pytest.raises(ValueError, match="holdout"):
            assign_splits(100, "seeded_permutation", {"train": 0.5, "holdout": 0.5}, seed=0)

    def test_fractions_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum"):
            assign_splits(100, "seeded_permutation", {"train": 0.7, "test": 0.2}, seed=0)

    def test_fraction_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            assign_splits(
                100, "seeded_permutation", {"train": 1.0, "validation": 0.0, "test": 0.0}, seed=0
            )

    def test_empty_fractions_refused(self) -> None:
        with pytest.raises(ValueError, match="fraction"):
            assign_splits(100, "seeded_permutation", {}, seed=0)

    def test_n_rows_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="n_rows"):
            assign_splits(0, "seeded_permutation", SIMULATED, seed=0)

    def test_empty_block_refused(self) -> None:
        """Three rows cannot carry a 70/15/15 split; a silent empty test set is worse."""
        with pytest.raises(ValueError, match="empty"):
            assign_splits(3, "seeded_permutation", SIMULATED, seed=0)


class TestSplitMethodEnum:
    def test_enum_members_accepted(self) -> None:
        config = pytest.importorskip("qecgen.residual.config")
        by_enum = assign_splits(100, config.SplitMethod.SEEDED_PERMUTATION, SIMULATED, seed=5)
        by_str = assign_splits(100, "seeded_permutation", SIMULATED, seed=5)
        assert np.array_equal(by_enum, by_str)
        contiguous = assign_splits(100, config.SplitMethod.CONTIGUOUS_BLOCKS, WILLOW, seed=None)
        assert np.all(np.diff(contiguous.astype(np.int16)) >= 0)
