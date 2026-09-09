"""Protect test sealing and uncertainty arithmetic independently of GPU training."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from research.realism.decoder import DecoderConfig, fit_decoder, load_decoder, predict
from research.realism.evaluate import (
    block_interval,
    digest,
    interval,
    paired,
    seal,
    validate_config,
    verify_seal,
)
from research.realism.pilot import write_json


def test_binomial_boundaries_and_known_half_interval() -> None:
    assert interval(0, 10)[0] == 0
    assert interval(10, 10)[1] == 1
    assert interval(5, 10) == pytest.approx([0.187086, 0.812914], abs=1e-6)
    with pytest.raises(ValueError):
        interval(2, 1)


def test_paired_counts_and_direction() -> None:
    left = np.array([True, True, False, False])
    right = np.array([True, False, True, False])
    result = paired(left, right, 3)
    assert [
        result[k]
        for k in ("left_only_failed", "right_only_failed", "both_failed", "neither_failed")
    ] == [1, 1, 1, 1]
    assert result["left_minus_right_rate"] == 0
    assert paired(np.ones(4, dtype=bool), right, 3)["left_minus_right_rate"] == 0.5


def test_block_bootstrap_constant_remainder_and_reproducibility() -> None:
    assert block_interval(np.ones(9), 4, seed=1) == [1, 1]
    values = np.arange(200) % 2
    assert block_interval(values, 11, seed=9) == block_interval(values, 11, seed=9)
    assert block_interval(np.zeros(9), 4, seed=1) == [0, 0]


@pytest.mark.parametrize(
    "changed", ["checkpoint", "config.json", "development.json", "source_hashes.json"]
)
def test_test_seal_detects_changes(tmp_path: Path, changed: str) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"locked weights")
    for name in ("config.json", "development.json", "source_hashes.json"):
        write_json(tmp_path / name, {})
    lock = seal(tmp_path, [{"training": [{"checkpoint": str(checkpoint)}]}])
    verify_seal(tmp_path, lock)
    (tmp_path / changed).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed before test"):
        verify_seal(tmp_path, lock)


def test_saved_checkpoint_inference_is_identical(tmp_path: Path) -> None:
    x = np.random.default_rng(3).integers(0, 2, (32, 2, 3)).astype(bool)
    y = x[:, 0, 0] ^ x[:, 1, 1]
    fitted = fit_decoder(
        x, y, x, y, DecoderConfig(hidden_size=4, max_epochs=2, device="cpu"), 3, tmp_path
    )
    expected = predict(fitted.model, x)
    restored = load_decoder(fitted.checkpoint_path, device="cpu", max_seconds=5)
    assert np.array_equal(predict(restored, x), expected)


@pytest.mark.parametrize(
    "case", ["empty_cohorts", "empty_seeds", "duplicate_seeds", "missing_arm", "over_budget", "nan"]
)
def test_invalid_protocol_fails_before_training(case: str) -> None:
    config = json.loads(
        Path("research/realism/evaluation_config.json").read_text(encoding="utf-8-sig")
    )
    validate_config(config)
    if case == "empty_cohorts":
        config["cohorts"] = []
    elif case == "empty_seeds":
        config["training_seeds"] = []
    elif case == "duplicate_seeds":
        config["training_seeds"] = [17, 17]
    elif case == "missing_arm":
        config["arms"].pop()
    elif case == "over_budget":
        config["base_fit_seconds"] = 1000
    else:
        config["base_fit_seconds"] = float("nan")
    with pytest.raises(ValueError):
        validate_config(config)


def test_seal_checks_the_actual_source_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    snapshot = source / "qecgen_circuits.py"
    snapshot.write_text("original source")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"weights")
    write_json(tmp_path / "config.json", {})
    write_json(tmp_path / "development.json", {})
    write_json(tmp_path / "source_hashes.json", {"qecgen/circuits.py": digest(snapshot)})
    lock = seal(tmp_path, [{"training": [{"checkpoint": str(checkpoint)}]}])
    verify_seal(tmp_path, lock)
    snapshot.write_text("changed source")
    with pytest.raises(ValueError, match="source snapshot changed"):
        verify_seal(tmp_path, lock)
