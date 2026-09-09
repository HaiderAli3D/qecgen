"""Frozen pre-overhaul outputs, independent of the new model's implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from research.realism.legacy import LegacyConfig, fingerprint, runtime_contract

GOLDEN_PATH = Path(__file__).parents[1] / "legacy_golden.json"


def _golden() -> dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_references_cover_every_existing_noise_path_and_both_contracts() -> None:
    entries = _golden()["cases"]
    coverage = {
        (
            entry["config"]["noise_model"],
            entry["config"]["basis"],
            entry["config"]["emit_mechanisms"],
        )
        for entry in entries
    }
    assert coverage == {
        (model, basis, mechanisms)
        for model in ("code_capacity", "phenomenological", "stim_uniform_circuit_level")
        for basis in ("x", "z")
        for mechanisms in (False, True)
    }
    assert {entry["config"]["chunk_size"] for entry in entries} == {31, 64}
    assert {entry["config"]["seed"] for entry in entries} == {0, 20260909}
    assert len(entries) == 24


@pytest.mark.parametrize("case_index", range(24))
def test_legacy_matches_frozen_arrays_circuit_and_content_hash(case_index: int) -> None:
    golden = _golden()
    current = runtime_contract()
    recorded = golden["runtime"]
    # A changed library pin needs a deliberate migration, not a silent golden refresh.
    assert current["stim"] == recorded["stim"]
    assert current["numpy"] == recorded["numpy"]
    for key in ("platform", "machine", "byteorder", "stim_march"):
        if current[key] != recorded[key]:
            pytest.skip(f"Seeded Stim stream requires reference {key}={recorded[key]!r}")
    entry = golden["cases"][case_index]
    assert fingerprint(entry["config"]) == entry["fingerprint"]


def test_reference_distinguishes_chunk_call_structure() -> None:
    """A retained master seed must not disguise a changed sequence of sample calls."""
    entry = _golden()["cases"][-1]
    changed = cast(LegacyConfig, dict(entry["config"], chunk_size=32))
    assert fingerprint(changed)["content_hash"] != entry["fingerprint"]["content_hash"]
