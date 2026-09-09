"""Read-only helpers for the device form; sampling remains in the worker."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import stim

from qecgen.configuration import ideal_circuit
from qecgen.noise import ANNOTATIONS, _layers
from qecgen.ui.schemas import ConfiguredRequest


def layout(config: dict[str, Any], data_root: Path) -> dict[str, Any]:
    """Use the simulator's layer partition instead of teaching a second convention.

    Calibration may still be incomplete when asking which qubits need values.
    Only circuit identity and source paths matter for this read-only request.
    """
    working = deepcopy(config)
    working["output"] = {"path": "layout-preview.h5", "format": "hdf5", "structure": "none"}
    working["sampling"] = {"shots": 1, "seed": 0, "chunk_size": 1}
    if working.get("mode") == "device":
        working["noise"] = {"version": 1}
        working["parameter_provenance"] = {"kind": "scenario", "description": "Layout only"}
    spec = ConfiguredRequest(config=working).to_spec(data_root)
    circuit = ideal_circuit(spec.config)
    qubits: set[int] = set()
    edges: set[tuple[int, int]] = set()
    for instruction in circuit.flattened():
        if instruction.name in ANNOTATIONS:
            continue
        targets = instruction.targets_copy()
        qubits.update(target.value for target in targets if target.is_qubit_target)
        if stim.gate_data(instruction.name).is_two_qubit_gate:
            for a, b in zip(targets[::2], targets[1::2], strict=True):
                if a.is_qubit_target and b.is_qubit_target:
                    edges.add((min(a.value, b.value), max(a.value, b.value)))
    return {
        "qubits": sorted(qubits),
        "edges": [list(edge) for edge in sorted(edges)],
        "layer_count": len(_layers(circuit)),
        "n_detectors": circuit.num_detectors,
        "n_observables": circuit.num_observables,
        "circuit_sha256": hashlib.sha256(str(circuit).encode()).hexdigest(),
        "note": (
            "Qubits include data and check qubits. Layer durations need one value per listed "
            "layer; this layout supplies no measured calibration or physical round rate."
        ),
    }
