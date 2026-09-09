"""Equal widths and detector coordinates cannot establish a logical target convention."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import stim

from qecgen.circuits import Basis, ChannelVector, build_circuit_from_channels
from research.realism.legacy_mapping import LegacyMapping, audit_legacy_mapping

SIGNATURE_DIGESTS = {
    ("x", 10): "e0fb5364fdc305c7f9e9f6e8155c6f92be50567db1d4ee71b715f437c9fce727",
    ("x", 13): "d64f0e81da12525db150768982b0973299a7049434d05c613705d342cb864047",
    ("z", 10): "24dc142719651830cf4e16ef8ef08a2c0b055ab04d6a1ef5c7ab3bd643c2c845",
    ("z", 13): "db49f5fa6a2e1e1bcd7efcb58ef9a97d7cb3b88985c1e2ecaa75d2c8ab693964",
}


def _fixture(basis: str = "z") -> stim.Circuit:
    """A synthetic relabeling exercises the audit without downloading research data."""
    original = build_circuit_from_channels(3, ChannelVector(), 2, Basis(basis)).flattened()
    result = stim.Circuit()
    sign = 1 if basis == "z" else -1
    for ins in original:
        if ins.name in ("QUBIT_COORDS", "DETECTOR"):
            coords = ins.gate_args_copy()
            x, y = coords[:2]
            remapped = [10 + sign * (-x + y) / 2, 7 + sign * (x + y - 6) / 2]
            result.append(ins.name, ins.targets_copy(), remapped + coords[2:])
        else:
            result.append(ins)
    return result


@pytest.mark.parametrize("basis", ["x", "z"])
def test_complete_forward_audit_of_relabelled_fixture(basis: str) -> None:
    mapping = audit_legacy_mapping(_fixture(basis), basis, 2)
    assert mapping.detector_permutation == tuple(range(16))
    assert mapping.evidence["exact_fault_signatures_checked"] == 79
    assert mapping.evidence["observable_mapping"] == (
        "identity; no offset or syndrome-dependent correction"
    )
    assert mapping.evidence["hardware_shots_used"] is False


def test_same_detectors_with_different_logical_representative_are_rejected() -> None:
    changed = _fixture()
    final_check = [ins for ins in changed if ins.name == "DETECTOR"][-1]
    changed.append("OBSERVABLE_INCLUDE", final_check.targets_copy(), 0)
    # This observable is still deterministic ideally, and every coordinate is unchanged.
    changed.detector_error_model()
    with pytest.raises(ValueError, match=r"Pauli basis|response mismatch"):
        audit_legacy_mapping(changed, "z", 2)


def test_geometry_noise_and_round_mismatches_fail_closed() -> None:
    changed = _fixture()
    changed.append("DEPOLARIZE1", [1], 0.01)
    with pytest.raises(ValueError, match="ideal"):
        audit_legacy_mapping(changed, "z", 2)
    with pytest.raises(ValueError, match="matching d3"):
        audit_legacy_mapping(_fixture(), "z", 3)
    with pytest.raises(ValueError, match="two extraction"):
        audit_legacy_mapping(_fixture(), "z", 1)
    original = build_circuit_from_channels(3, ChannelVector(), 2, Basis.Z)
    with pytest.raises(ValueError, match="geometry"):
        audit_legacy_mapping(original, "z", 2)


def test_column_permutation_requires_true_width_and_unpacked_boolean_bits() -> None:
    mapping = LegacyMapping((2, 0, 1), {})
    bits = np.array([[False, True, True]], dtype=np.bool_)
    np.testing.assert_array_equal(mapping.detectors(bits), [[True, False, True]])
    with pytest.raises(ValueError, match="boolean"):
        mapping.detectors(np.array([[1]], dtype=np.uint8))
    with pytest.raises(ValueError, match="width"):
        mapping.detectors(bits[:, :2])


@pytest.mark.parametrize(
    ("basis", "rounds", "file_sha"),
    [
        ("x", 10, "9cc2b0fedb742c2e7abca54a8a0946695d2b505c40f400ef56cf08d765b8af67"),
        ("x", 13, "386e4dc3106e56fd96514289e215c204a2b99f8c25e09f0ebc438df5fdc73c71"),
        ("z", 10, "fba4d5575c0afa11ce2126acbbe7d3a2546609ecac66195ea2ba696c45ef085e"),
        ("z", 13, "6cdd932ba7a44195a8d920db62d9d51cd8381c8a83940d4d1b8d5c36cbcb1f50"),
    ],
)
def test_downloaded_willow_circuits_have_complete_certified_mapping(
    basis: str, rounds: int, file_sha: str
) -> None:
    path = Path(
        f"data/realism/raw/willow-derived-{basis}-r{rounds:03d}-circuit/"
        f"d3_at_q10_7__{basis.upper()}__r{rounds:03d}.stim"
    )
    if not path.exists():
        pytest.skip("optional checksum-verified hardware ideal circuit has not been downloaded")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == file_sha
    hardware = stim.Circuit(path.read_text())
    mapping = audit_legacy_mapping(hardware, basis, rounds)
    assert sorted(mapping.detector_permutation) == list(range(8 * rounds))
    assert mapping.evidence["exact_fault_signatures_checked"] == 35 * rounds + 9
    assert mapping.evidence["signature_sha256"] == SIGNATURE_DIGESTS[basis, rounds]
    # These qubits have equal check geometry across bases but opposite logical boundaries.
    expected_first = 8 if basis == "z" else 12
    assert mapping.evidence["qubit_correspondence"]["1"] == expected_first
    # Mixed Pauli products also obey the audited map, independently of single-fault sampler calls.
    legacy = build_circuit_from_channels(3, ChannelVector(), rounds, Basis(basis)).flattened()
    hp = hardware.flattened()
    for circuit, is_hardware in ((legacy, False), (hp, True)):
        inserted = stim.Circuit()
        completed = False
        for ins in circuit:
            if (
                not completed
                and ins.name in ("CX", "CZ")
                and all(t.is_qubit_target for t in ins.targets_copy())
            ):
                for q, pauli in ((1, "X"), (10, "Y"), (19, "Z")):
                    target = mapping.evidence["qubit_correspondence"][str(q)] if is_hardware else q
                    inserted.append(pauli + "_ERROR", [target], 1)
                completed = True
            inserted.append(ins)
        det, obs = inserted.compile_detector_sampler(seed=51).sample(16, separate_observables=True)
        if is_hardware:
            hardware_det, hardware_obs = det, obs
        else:
            legacy_det, legacy_obs = det, obs
    np.testing.assert_array_equal(mapping.detectors(legacy_det), hardware_det)
    np.testing.assert_array_equal(legacy_obs, hardware_obs)
