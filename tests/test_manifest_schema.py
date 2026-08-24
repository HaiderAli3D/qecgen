"""The manifest must name its own target, in a form a pipeline can read.

A consumer opened a generated file and could not tell which column was the label. The
target was inferable only by reading ``contract`` and then ``DATA_CONTRACT.md`` -- a
convention, and a convention is exactly what a data pipeline cannot read. These tests
pin the declaration that replaced it.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from qecgen.dataset import (
    ENVIRONMENT_COLUMN,
    SHOT_COLUMN,
    Contract,
    DatasetMeta,
    InMemoryDataset,
    StructureLevel,
    target_columns,
)
from qecgen.environments import build_multi_environment, build_single_environment
from qecgen.exporters.csv_table import _expected_columns


@pytest.fixture
def contract_a() -> InMemoryDataset:
    return build_single_environment(distance=3, p=0.008, shots=64, seed=3, chunk_size=32)


@pytest.fixture
def contract_b() -> InMemoryDataset:
    return build_single_environment(
        distance=3, p=0.008, shots=64, seed=3, chunk_size=32, emit_mechanisms=True
    )


def test_contract_a_declares_observables_as_the_only_target(contract_a: InMemoryDataset) -> None:
    block = contract_a.meta.schema_block()
    assert block["targets"] == ["observables"]
    assert block["roles"]["observables"]["role"] == "target"
    assert block["roles"]["observables"]["target_of"] == str(Contract.LOGICAL_FRAME)
    assert block["roles"]["mechanisms"]["present"] == "absent"


def test_contract_b_declares_observables_and_mechanisms_as_targets(
    contract_b: InMemoryDataset,
) -> None:
    block = contract_b.meta.schema_block()
    assert block["targets"] == ["observables", "mechanisms"]
    assert block["roles"]["mechanisms"]["present"] == "always"
    assert block["roles"]["mechanisms"]["target_of"] == str(Contract.DEM_MECHANISM)


def test_primary_target_is_observables_under_both_contracts(
    contract_a: InMemoryDataset, contract_b: InMemoryDataset
) -> None:
    """`targets[-1]` must not be the way anyone picks the benchmark target.

    Under Contract B that expression selects the mechanism labels, which answer a
    different question and are not portable across distances or noise models.
    """
    assert contract_a.meta.schema_block()["primary_target"] == "observables"
    assert contract_b.meta.schema_block()["primary_target"] == "observables"


def test_contract_b_mechanisms_carry_the_not_physical_faults_caveat(
    contract_b: InMemoryDataset,
) -> None:
    caveat = contract_b.meta.schema_block()["roles"]["mechanisms"]["caveat"]
    assert "NOT physical Pauli faults" in caveat
    assert "not portable" in caveat


@pytest.mark.parametrize("emit", [False, True])
def test_no_declared_target_is_a_physical_fault(emit: bool) -> None:
    """Contract C is refused, not pending, so nothing may name it -- not even as absent.

    An entry marked ``"absent"`` reads as "coming soon", which is the wrong claim for a
    target that is underdetermined by quantum mechanics rather than merely unimplemented.
    """
    dataset = build_single_environment(
        distance=3, p=0.008, shots=32, seed=1, chunk_size=32, emit_mechanisms=emit
    )
    block = dataset.meta.schema_block()
    assert set(block["targets"]) <= {"observables", "mechanisms"}
    assert not any("fault" in name or "pauli" in name.lower() for name in block["roles"])
    assert "physical Pauli fault label" in block["note"]


@pytest.mark.parametrize("emit", [False, True])
def test_detectors_are_the_only_feature(emit: bool) -> None:
    dataset = build_single_environment(
        distance=3, p=0.008, shots=32, seed=1, chunk_size=32, emit_mechanisms=emit
    )
    block = dataset.meta.schema_block()
    assert block["features"] == ["detectors"]
    assert block["roles"]["detectors"]["width"] == dataset.meta.n_detectors


def test_environment_id_is_neither_a_feature_nor_a_target(contract_a: InMemoryDataset) -> None:
    """A model handed the environment id reads the noise level off the row.

    That is a shortcut past the physics the dataset exists to teach, so the id is
    declared a grouping key and appears in neither list.
    """
    block = contract_a.meta.schema_block()
    assert ENVIRONMENT_COLUMN not in block["features"]
    assert ENVIRONMENT_COLUMN not in block["targets"]
    assert block["roles"][ENVIRONMENT_COLUMN]["role"] == "grouping_key"


def test_environment_id_presence_is_not_claimed_from_the_environment_count() -> None:
    """A pooled file with one environment still carries the column.

    So presence cannot be derived from ``len(environments)``. Claiming it from the count
    would over-claim on this exact dataset, which the tree already builds elsewhere.
    """
    pooled = build_multi_environment(
        distance=3, error_rates=[0.01], shots_per_env=32, seed=1, chunk_size=32
    )
    assert pooled.environment_ids is not None
    assert len(pooled.meta.environments) == 1
    assert pooled.meta.schema_block()["roles"][ENVIRONMENT_COLUMN]["present"] == "if_pooled"


@pytest.mark.parametrize("emit", [False, True])
def test_csv_prefixes_resolve_to_the_columns_the_exporter_writes(emit: bool) -> None:
    """The declaration must resolve to real column names, or it is decoration.

    The rule a consumer applies: use ``csv_names`` where a role publishes it, else
    ``csv_prefix`` plus ``width``. If that produced anything other than the header the
    writer emits, the manifest would be naming columns that are not in the file -- the
    defect this block exists to end, restated.
    """
    dataset = build_single_environment(
        distance=3, p=0.008, shots=32, seed=1, chunk_size=32, emit_mechanisms=emit
    )
    roles = dataset.meta.schema_block()["roles"]

    def expand(name: str) -> list[str]:
        entry = roles[name]
        literal = entry.get("csv_names")
        if literal is not None:
            return list(literal)
        return [f"{entry['csv_prefix']}{i}" for i in range(entry["width"] or 0)]

    resolved = [
        SHOT_COLUMN,
        *expand("observables"),
        *expand("detectors"),
        *expand("mechanisms"),
    ]
    written = _expected_columns(
        dataset.meta, has_environment=False, has_mechanisms=dataset.mechanisms is not None
    )
    assert resolved == written


@pytest.mark.parametrize(
    ("level", "dem_present", "provenance_present"),
    [
        (StructureLevel.NONE, "absent", "absent"),
        (StructureLevel.COORDS, "always", "absent"),
        (StructureLevel.DEM, "always", "absent"),
        (StructureLevel.FULL, "always", "always"),
    ],
)
def test_side_information_and_provenance_track_the_structure_level(
    level: StructureLevel, dem_present: str, provenance_present: str
) -> None:
    dataset = build_single_environment(
        distance=3, p=0.008, shots=32, seed=1, chunk_size=32, structure_level=level
    )
    roles = dataset.meta.schema_block()["roles"]
    assert roles["dem"]["role"] == "side_information"
    assert roles["dem"]["present"] == dem_present
    assert roles["provenance"]["role"] == "never_read"
    assert roles["provenance"]["present"] == provenance_present


def test_manifest_without_a_schema_block_reads_and_regains_it(contract_a: InMemoryDataset) -> None:
    """Every manifest written before this block existed must still read.

    It is derived on the way out rather than stored on the way in, so an old file gains
    the declaration simply by being read by a current reader.
    """
    payload = contract_a.meta.to_json_dict()
    del payload["schema"]
    restored = DatasetMeta.from_json_dict(payload)
    assert restored.schema_block() == contract_a.meta.schema_block()


def test_a_disagreeing_schema_block_is_refused(contract_a: InMemoryDataset) -> None:
    """Present and contradictory is refused, never resolved.

    The stored block and the fields it derives from are two descriptions of one dataset;
    preferring either would guess which half is the corrupt one.
    """
    payload = contract_a.meta.to_json_dict()
    payload["schema"]["targets"] = ["detectors"]
    with pytest.raises(ValueError, match="disagrees with the fields it is derived from"):
        DatasetMeta.from_json_dict(payload)


def test_prose_may_change_without_being_refused(contract_a: InMemoryDataset) -> None:
    """Only the claims a consumer acts on are compared.

    A later writer improving a `meaning` sentence must not read as corruption, or the
    check becomes a reason never to improve the copy.
    """
    payload = contract_a.meta.to_json_dict()
    payload["schema"]["roles"]["detectors"]["meaning"] = "reworded, same claim"
    payload["schema"]["note"] = "reworded"
    assert DatasetMeta.from_json_dict(payload).n_detectors == contract_a.meta.n_detectors


def test_schema_is_not_a_dataclass_field() -> None:
    """Stored, it would go stale exactly where it matters.

    `jsonl` and `parquet` serialise `dataclasses.replace(meta, structure_level=...)` to
    record a downgrade. A stored block would be computed before that replace and keep
    advertising a DEM the file no longer carries.
    """
    assert "schema" not in {f.name for f in dataclasses.fields(DatasetMeta)}


def test_a_zero_width_mechanism_target_is_declared_honestly() -> None:
    """A noiseless DEM has no mechanisms, and that is correct data, not corruption."""
    dataset = build_single_environment(
        distance=3, p=0.0, shots=32, seed=1, chunk_size=32, emit_mechanisms=True
    )
    block = dataset.meta.schema_block()
    assert dataset.meta.n_mechanisms == 0
    assert block["targets"] == ["observables", "mechanisms"]
    assert block["roles"]["mechanisms"]["present"] == "always"
    assert block["roles"]["mechanisms"]["width"] == 0


def test_the_schema_block_is_not_the_source_of_the_hash_names(contract_a: InMemoryDataset) -> None:
    """The digest's alphabet must stay literal and separate from these role names.

    `content_hash` folds array *name strings* into the digest. Unifying them with the
    schema block -- which spells this one `environment_id`, singular, because that is the
    column name -- would silently rehash every dataset ever produced.
    """
    before = contract_a.compute_content_hash()
    relabelled = dataclasses.replace(
        contract_a, meta=dataclasses.replace(contract_a.meta, contract=Contract.DEM_MECHANISM)
    )
    assert relabelled.compute_content_hash() == before

    roles = contract_a.meta.schema_block()["roles"]
    assert ENVIRONMENT_COLUMN in roles
    assert "environment_ids" not in roles


def test_the_block_survives_a_json_round_trip(contract_a: InMemoryDataset) -> None:
    """It has to be plain JSON: the point is that a non-Python consumer can read it."""
    payload = json.loads(contract_a.meta.to_json())
    assert payload["schema"]["targets"] == ["observables"]
    assert payload["schema"]["roles"]["observables"]["csv_names"] == target_columns(1)
