"""Prove Contract A correspondence before comparing unchanged legacy shots.

Coordinates propose a correspondence; exact propagation of known faults verifies it.
The schedules need not have the same noise. This forward calculation does not infer
physical faults from hardware outcomes or introduce physical-fault training targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import stim
from numpy.typing import NDArray

from qecgen.circuits import Basis, ChannelVector, build_circuit_from_channels
from research.realism.import_real import detector_anchors


@dataclass(frozen=True)
class LegacyMapping:
    detector_permutation: tuple[int, ...]
    evidence: dict[str, Any]

    def detectors(self, bits: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Convert unpacked legacy detector bits to hardware column order."""
        if bits.dtype != np.bool_ or bits.ndim != 2:
            raise ValueError("mapping requires a two-dimensional unpacked boolean array")
        if bits.shape[1] != len(self.detector_permutation):
            raise ValueError("legacy detector width disagrees with the audited circuit")
        return bits[:, self.detector_permutation]


def _transform(x: float, y: float, basis: Basis) -> tuple[float, float]:
    sign = 1 if basis is Basis.Z else -1
    return 10 + sign * (-x + y) / 2, 7 + sign * (x + y - 6) / 2


def _response(circuit: stim.Circuit) -> NDArray[np.bool_]:
    det, obs = circuit.compile_detector_sampler(seed=0).sample(1, separate_observables=True)
    return np.asarray(np.concatenate((det[0], obs[0])), dtype=np.bool_)


def _fault_response(circuit: stim.Circuit, index: int, qubit: int, pauli: str) -> NDArray[np.bool_]:
    injected = circuit[:index]
    # A deterministic Clifford folds into Stim's reference and would test nothing.
    injected.append(pauli + "_ERROR", [qubit], 1)
    injected += circuit[index:]
    return _response(injected)


def _readout_response(circuit: stim.Circuit, index: int, qubit: int) -> NDArray[np.bool_]:
    injected = circuit[:index]
    ins = circuit[index]
    for target in ins.targets_copy():
        injected.append(ins.name, [target], float(target.value == qubit))
    injected += circuit[index + 1 :]
    return _response(injected)


def _round_starts(circuit: stim.Circuit) -> list[int]:
    starts: list[int] = []
    ready = True
    for i, ins in enumerate(circuit):
        if ins.name in ("M", "MR", "MX", "MRX"):
            ready = True
        if (
            ready
            and ins.name in ("CX", "CZ")
            and all(t.is_qubit_target for t in ins.targets_copy())
        ):
            starts.append(i)
            ready = False
    return starts


def audit_legacy_mapping(
    hardware_ideal: stim.Circuit, basis: str | Basis, rounds: int
) -> LegacyMapping:
    """Check geometry, all round boundaries and logical target signatures.

    Limited to the downloaded d3 q10_7 Willow layout. Checks X/Y/Z on each data
    qubit before each extraction round, every ancilla readout, and final readout.
    Clifford propagation is linear, so these generators cover their combinations.
    This does not assert equal signatures inside different gate schedules, nor
    reproduce leakage or state-dependent noise. Observable flips map identically.
    """
    basis = Basis(basis.lower())
    if isinstance(rounds, bool) or rounds < 2:
        raise ValueError("mapping requires at least two extraction rounds")
    if hardware_ideal != hardware_ideal.without_noise():
        raise ValueError("mapping requires an ideal hardware circuit, without source priors")
    source_text = str(hardware_ideal)
    hardware = hardware_ideal.flattened()
    original = build_circuit_from_channels(3, ChannelVector(), rounds, basis)
    legacy = original.flattened()
    if hardware.num_observables != 1 or hardware.num_detectors != legacy.num_detectors:
        raise ValueError("only matching d3 one-observable memory experiments are supported")
    hardware.detector_error_model()
    legacy.detector_error_model()
    hc = detector_anchors(hardware)
    lc = legacy.get_detector_coordinates()
    hkeys = {tuple(v): k for k, v in hc.items()}
    keys = [(*_transform(lc[i][0], lc[i][1], basis), lc[i][2]) for i in range(legacy.num_detectors)]
    if len(hkeys) != hardware.num_detectors or set(keys) != set(hkeys):
        raise ValueError("detector geometry or extraction-time/boundary meanings disagree")
    permutation = tuple(int(i) for i in np.argsort([hkeys[k] for k in keys]))
    indices = [*permutation, legacy.num_detectors]
    hq = {tuple(v): k for k, v in hardware.get_final_qubit_coordinates().items()}
    lq = legacy.get_final_qubit_coordinates()
    lmeasure = [i for i, ins in enumerate(legacy) if ins.name in ("M", "MX", "MR", "MRX")]
    hmeasure = [i for i, ins in enumerate(hardware) if ins.name in ("M", "MX", "MR", "MRX")]
    if len(lmeasure) != rounds + 1 or len(hmeasure) != rounds + 1:
        raise ValueError("unsupported measurement/reset grouping")
    data = [t.value for t in legacy[lmeasure[-1]].targets_copy()]
    if len(data) != 9:
        raise ValueError("the audited layout must contain nine final data qubits")
    pairs: dict[int, int] = {}
    for qubit in lq:
        transformed = _transform(lq[qubit][0], lq[qubit][1], basis)
        if transformed not in hq:
            raise ValueError("qubit geometry cannot support the proposed check correspondence")
        pairs[qubit] = hq[transformed]
    hs, ls = _round_starts(hardware), _round_starts(legacy)
    if len(hs) != rounds or len(ls) != rounds:
        raise ValueError("cannot locate one extraction start per round")
    digest = hashlib.sha256()
    checked = 0

    def compare(left: NDArray[np.bool_], right: NDArray[np.bool_], label: str) -> None:
        nonlocal checked
        aligned = left[indices]
        if not np.array_equal(aligned, right):
            raise ValueError(f"detector/logical response mismatch: {label}")
        digest.update(label.encode())
        digest.update(np.packbits(aligned, bitorder="little").tobytes())
        checked += 1

    orientations: dict[int, str] = {}
    for q in data:
        candidate = _fault_response(legacy, ls[1], q, "X")[indices]
        matches = [
            p
            for p in "XZ"
            if np.array_equal(candidate, _fault_response(hardware, hs[1], pairs[q], p))
        ]
        if len(matches) != 1:
            raise ValueError("data-qubit local Pauli basis is not uniquely established")
        orientations[q] = matches[0]
    for r, (li, hi) in enumerate(zip(ls, hs, strict=True)):
        for q in data:
            for p in "XYZ":
                hp = p if orientations[q] == "X" or p == "Y" else {"X": "Z", "Z": "X"}[p]
                compare(
                    _fault_response(legacy, li, q, p),
                    _fault_response(hardware, hi, pairs[q], hp),
                    f"round={r};legacy_qubit={q};pauli={p}",
                )
    for r, (li, hi) in enumerate(zip(lmeasure, hmeasure, strict=True)):
        targets = [t.value for t in legacy[li].targets_copy()]
        if {pairs[q] for q in targets} != {t.value for t in hardware[hi].targets_copy()}:
            raise ValueError("measurement supports or final-boundary definitions disagree")
        for q in targets:
            compare(
                _readout_response(legacy, li, q),
                _readout_response(hardware, hi, pairs[q]),
                f"measurement_layer={r};legacy_qubit={q}",
            )
    return LegacyMapping(
        permutation,
        {
            "audit_version": 1,
            "basis": str(basis),
            "rounds": rounds,
            "distance": 3,
            "stim_version": stim.__version__,
            "legacy_ideal_sha256": hashlib.sha256(str(original).encode()).hexdigest(),
            "hardware_ideal_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
            "detector_permutation": list(permutation),
            "observable_mapping": "identity; no offset or syndrome-dependent correction",
            "geometry": (
                "hardware_x=10+s*(-legacy_x+legacy_y)/2; "
                "hardware_y=7+s*(legacy_x+legacy_y-6)/2; s=+1 for Z, -1 for X"
            ),
            "qubit_correspondence": {str(q): h for q, h in pairs.items()},
            "legacy_x_maps_to_hardware_pauli": {str(q): p for q, p in orientations.items()},
            "exact_fault_signatures_checked": checked,
            "signature_sha256": digest.hexdigest(),
            "audit": "all-round data X/Y/Z; all ancilla readouts; all final data readouts",
            "scope": "Contract A detector/observable semantics; gate schedules and noise differ",
            "source_noise_weights_used": False,
            "hardware_shots_used": False,
            "logical_reference": "each circuit's own ideal reference; sweep bits zero",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("circuit", type=Path)
    parser.add_argument("--basis", choices=("x", "z"), required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_legacy_mapping(stim.Circuit(args.circuit.read_text()), args.basis, args.rounds)
    payload = json.dumps(result.evidence, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
