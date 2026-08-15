"""Validation must actually catch corruption, not just pass on good data."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qecgen.circuits import NoiseModel
from qecgen.dataset import Contract, InMemoryDataset, StructureLevel
from qecgen.environments import build_multi_environment, build_single_environment
from qecgen.validate import validate_dataset, validate_dem_structure


@pytest.fixture
def good() -> InMemoryDataset:
    return build_single_environment(
        distance=3,
        p=0.008,
        shots=200,
        seed=3,
        chunk_size=64,
        structure_level=StructureLevel.DEM,
    )


def _failed(dataset: InMemoryDataset) -> set[str]:
    return {r.name for r in validate_dataset(dataset).failures}


def test_good_dataset_passes(good: InMemoryDataset) -> None:
    report = validate_dataset(good)
    assert report.ok, str(report)


def test_detects_wrong_n_detectors(good: InMemoryDataset) -> None:
    meta = dataclasses.replace(good.meta, n_detectors=999)
    broken = dataclasses.replace(good, meta=meta)
    assert "detectors.packed_width" in _failed(broken)


def test_detects_shot_count_mismatch(good: InMemoryDataset) -> None:
    meta = dataclasses.replace(good.meta, shots=12345)
    assert "manifest.shots" in _failed(dataclasses.replace(good, meta=meta))


def test_detects_big_endian_declaration(good: InMemoryDataset) -> None:
    meta = dataclasses.replace(good.meta, bit_order="big")
    assert "bit_order" in _failed(dataclasses.replace(good, meta=meta))


def test_detects_tampered_content(good: InMemoryDataset) -> None:
    detectors = good.detectors.copy()
    detectors[0, 0] ^= 0xFF
    assert "content_hash" in _failed(dataclasses.replace(good, detectors=detectors))


def test_detects_contract_mismatch(good: InMemoryDataset) -> None:
    meta = dataclasses.replace(good.meta, contract=Contract.DEM_MECHANISM)
    assert "contract_b.mechanisms_present" in _failed(dataclasses.replace(good, meta=meta))


def test_detects_environment_id_count_mismatch() -> None:
    dataset = build_multi_environment(
        distance=3, error_rates=[0.005, 0.01], shots_per_env=100, seed=1, chunk_size=64
    )
    assert dataset.environment_ids is not None
    ids = dataset.environment_ids.copy()
    ids[:] = 0
    assert "environment_ids.counts" in _failed(dataclasses.replace(dataset, environment_ids=ids))


def test_rejects_row_count_mismatch_at_construction(good: InMemoryDataset) -> None:
    """Per-shot correspondence is enforced in __post_init__, before validation."""
    with pytest.raises(ValueError, match="correspondence is broken"):
        dataclasses.replace(good, observables=good.observables[:10])


def test_zero_noise_checks_apply_only_at_p_zero() -> None:
    zero = build_single_environment(distance=3, p=0.0, shots=128, seed=1, chunk_size=64)
    names = {r.name for r in validate_dataset(zero).results}
    assert "zero_noise.detectors_all_zero" in names
    assert validate_dataset(zero).ok

    noisy = build_single_environment(distance=3, p=0.01, shots=128, seed=1, chunk_size=64)
    assert "zero_noise.detectors_all_zero" not in {r.name for r in validate_dataset(noisy).results}


def test_zero_noise_detects_impossible_events() -> None:
    zero = build_single_environment(distance=3, p=0.0, shots=64, seed=1, chunk_size=64)
    detectors = zero.detectors.copy()
    detectors[0, 0] = 1
    broken = dataclasses.replace(zero, detectors=detectors)
    assert "zero_noise.detectors_all_zero" in _failed(broken)


def test_code_capacity_zero_noise_validates() -> None:
    dataset = build_single_environment(
        distance=3,
        p=0.0,
        shots=64,
        seed=1,
        chunk_size=64,
        noise_model=NoiseModel.CODE_CAPACITY,
    )
    assert validate_dataset(dataset).ok


class TestDemValidation:
    def test_detects_component_weight_above_two(self, good: InMemoryDataset) -> None:
        """Simulates the classic parse bug: components merged into one column."""
        from qecgen.dem import DemComponent

        assert good.structure is not None
        bad_component = DemComponent(
            parent_mechanism_id=0, component_index=0, detectors=(0, 1, 2, 3), observables=()
        )
        broken = dataclasses.replace(
            good.structure, components=(bad_component, *good.structure.components[1:])
        )
        failures = {r.name for r in validate_dem_structure(broken).failures}
        assert "dem.component_weight_le_2" in failures

    def test_detects_column_count_inflation(self, good: InMemoryDataset) -> None:
        """n_mechanisms must equal error instructions, not component count."""
        import stim

        assert good.structure is not None
        dem = stim.DetectorErrorModel(good.meta.environments[0].dem)
        inflated = dataclasses.replace(good.structure, n_mechanisms=len(good.structure.components))
        failures = {r.name for r in validate_dem_structure(inflated, dem).failures}
        assert "dem.one_column_per_error_instruction" in failures

    def test_mechanism_weight_is_reported_not_asserted(self, good: InMemoryDataset) -> None:
        assert good.structure is not None
        report = validate_dem_structure(good.structure)
        entry = next(r for r in report.results if r.name == "dem.mechanism_weight_distribution")
        assert entry.passed
        assert "reported" in entry.detail


def test_report_str_shows_expectation_only_on_failure(good: InMemoryDataset) -> None:
    meta = dataclasses.replace(good.meta, bit_order="big")
    report = validate_dataset(dataclasses.replace(good, meta=meta))
    failure = next(r for r in report.results if r.name == "bit_order")
    assert "expected" in str(failure)

    passing = next(r for r in validate_dataset(good).results if r.name == "bit_order")
    assert "expected" not in str(passing)


def test_hash_check_can_be_skipped(good: InMemoryDataset) -> None:
    detectors = good.detectors.copy()
    detectors[0, 0] ^= 0xFF
    broken = dataclasses.replace(good, detectors=detectors)
    names = {r.name for r in validate_dataset(broken, check_hash=False).results}
    assert "content_hash" not in names


def test_packed_width_check_uses_ceiling(good: InMemoryDataset) -> None:
    assert good.detectors.shape[1] == int(np.ceil(good.meta.n_detectors / 8))
