"""Configurable superconducting noise approximations, separate from shot sampling.

This is a calibration *interface*, not a claim that these channels reproduce a device.
The T1/T2 channel is the Pauli twirl in Ghosh, Fowler and Geller, Eq. (10),
https://arxiv.org/abs/1210.5799. It loses amplitude damping's non-unital behaviour.
Leakage below means a persistent Pauli-effect proxy, never a simulated third level.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import stim

OPERATIONS = ("one_qubit_gate", "two_qubit_gate", "measurement", "reset", "idle")
ANNOTATIONS = {"QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "TICK"}
MEASUREMENTS = {"M", "MX", "MY", "MR", "MRX", "MRY"}
RESETS = {"R", "RX", "RY", "MR", "MRX", "MRY"}


def _keys(obj: dict[str, Any], allowed: set[str], where: str) -> None:
    if not isinstance(obj, dict) or any(not isinstance(key, str) for key in obj):
        raise ValueError(f"{where} must be an object with string keys")
    if extra := set(obj) - allowed:
        raise ValueError(f"Unknown {where} fields: {sorted(extra)}")


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{where} must be an object with string keys")
    return value


def _sequence(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{where} must be a list")
    return value


def _required(obj: dict[str, Any], names: set[str], where: str) -> None:
    if missing := names - obj.keys():
        raise ValueError(f"Missing {where} fields: {sorted(missing)}")


def _number(value: Any, where: str, *, minimum: float = 0, maximum: float = math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{where} must be finite and in [{minimum}, {maximum}]")
    return result


def _prob(value: Any, where: str) -> float:
    return _number(value, where, maximum=1)


def _qubits(value: Any, where: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where} requires a nonempty list of qubit indices")
    if any(type(q) is not int or q < 0 for q in value) or len(set(value)) != len(value):
        raise ValueError(f"{where} requires distinct nonnegative integer qubits")
    return value


def coherence_probabilities(duration_s: float, t1_s: float, t2_s: float) -> tuple[float, ...]:
    """T2<=2*T1 is required by this Markov model, not by every measured T2 protocol."""
    dt = _number(duration_s, "duration_s")
    t1 = _number(t1_s, "t1_s", minimum=np.finfo(float).tiny)
    t2 = _number(t2_s, "t2_s", minimum=np.finfo(float).tiny)
    if t2 > 2 * t1:
        raise ValueError("This exponential Markov approximation requires T2 <= 2*T1")
    px = -math.expm1(-dt / t1) / 4
    pz = -math.expm1(-dt / t2) / 2 - px
    return px, px, max(0.0, pz)


@dataclass(frozen=True)
class NoiseProfile:
    """Validated, copied config: covariates never silently become error probabilities."""

    _config: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_config", self._normalise(self._config))

    @classmethod
    def uniform(cls, p: float) -> NoiseProfile:
        return cls.from_dict({"version": 1, "probabilities": dict.fromkeys(OPERATIONS, p)})

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> NoiseProfile:
        return cls(raw)

    @staticmethod
    def _normalise(raw: dict[str, Any]) -> dict[str, Any]:
        obj = copy.deepcopy(raw)
        _keys(
            obj,
            {
                "version",
                "label",
                "probabilities",
                "qubit_overrides",
                "edge_overrides",
                "coherence",
                "spatial",
                "drift",
                "leakage",
                "bursts",
                "covariates",
                "classical_control_policy",
            },
            "profile",
        )
        if type(obj.get("version")) is not int or obj["version"] != 1:
            raise ValueError("Profile version must be integer 1")
        if "label" in obj and not isinstance(obj["label"], str):
            raise ValueError("label must be a string")
        if not isinstance(obj.get("classical_control_policy", "reject"), str):
            raise ValueError("classical_control_policy must be a string")
        if obj.setdefault("classical_control_policy", "reject") not in {
            "reject",
            "ideal_pauli_frame",
        }:
            raise ValueError("classical_control_policy must be reject or ideal_pauli_frame")
        probabilities = obj.setdefault("probabilities", {})
        _keys(probabilities, set(OPERATIONS), "probabilities")
        for op in OPERATIONS:
            probabilities[op] = _prob(probabilities.get(op, 0), op)
        for q, values in _object(obj.setdefault("qubit_overrides", {}), "qubit_overrides").items():
            if not q.isdigit() or str(int(q)) != q:
                raise ValueError("qubit_overrides keys must be canonical nonnegative indices")
            _keys(values, set(OPERATIONS) - {"two_qubit_gate"}, f"qubit {q}")
            for op, p in values.items():
                values[op] = _prob(p, f"qubit {q} {op}")
        for edge, p in _object(obj.setdefault("edge_overrides", {}), "edge_overrides").items():
            parts = edge.split(",")
            if len(parts) != 2 or any(not q.isdigit() for q in parts):
                raise ValueError("edge_overrides keys must be 'smaller,larger'")
            a, b = map(int, parts)
            if a >= b or edge != f"{a},{b}":
                raise ValueError("edge_overrides keys must be canonical 'smaller,larger'")
            obj["edge_overrides"][edge] = _prob(p, f"edge {edge}")
        coherence = obj.setdefault("coherence", {})
        _keys(
            coherence,
            {
                "enabled",
                "qubits",
                "layer_durations_s",
                "t2_protocol",
                "residual_gate_errors_exclude_decoherence",
                "round_end_layers",
            },
            "coherence",
        )
        if "enabled" in coherence and type(coherence["enabled"]) is not bool:
            raise ValueError("coherence.enabled must be boolean")
        if (
            "residual_gate_errors_exclude_decoherence" in coherence
            and type(coherence["residual_gate_errors_exclude_decoherence"]) is not bool
        ):
            raise ValueError("residual_gate_errors_exclude_decoherence must be boolean")
        if "t2_protocol" in coherence and coherence["t2_protocol"] != "exponential_ramsey":
            raise ValueError("coherence needs t2_protocol='exponential_ramsey'")
        for q, calibration in _object(coherence.get("qubits", {}), "coherence.qubits").items():
            if not q.isdigit() or str(int(q)) != q:
                raise ValueError("coherence qubit keys must be canonical indices")
            _keys(calibration, {"t1_s", "t2_s"}, f"coherence qubit {q}")
            _required(calibration, {"t1_s", "t2_s"}, f"coherence qubit {q}")
            coherence_probabilities(0, calibration["t1_s"], calibration["t2_s"])
        for duration in _sequence(coherence.get("layer_durations_s", []), "layer_durations_s"):
            _number(duration, "layer_durations_s")
        ends = _sequence(coherence.get("round_end_layers", []), "round_end_layers")
        if any(type(end) is not int or end <= 0 for end in ends) or ends != sorted(set(ends)):
            raise ValueError("round_end_layers must be strictly increasing positive indices")
        if coherence.get("enabled", False):
            if coherence.get("t2_protocol") != "exponential_ramsey":
                raise ValueError("coherence needs t2_protocol='exponential_ramsey'")
            has_residual = any(
                probabilities[k] for k in ("one_qubit_gate", "two_qubit_gate", "idle")
            )
            has_residual |= any(obj["edge_overrides"].values())
            has_residual |= any(
                values.get(k, 0)
                for values in obj["qubit_overrides"].values()
                for k in ("one_qubit_gate", "idle")
            )
            if (
                has_residual
                and coherence.get("residual_gate_errors_exclude_decoherence") is not True
            ):
                raise ValueError("Explicitly acknowledge residual gate errors exclude decoherence")
            if not coherence.get("qubits") or not coherence.get("layer_durations_s"):
                raise ValueError("coherence requires qubit calibration and layer_durations_s")
        spatial = obj.setdefault("spatial", [])
        for effect in _sequence(spatial, "spatial"):
            _keys(effect, {"enabled", "qubits", "paulis", "probability", "basis"}, "spatial")
            _required(effect, {"qubits", "paulis", "probability", "basis"}, "spatial")
            if "enabled" in effect and type(effect["enabled"]) is not bool:
                raise ValueError("spatial.enabled must be boolean")
            qs = _qubits(effect["qubits"], "spatial")
            if (
                not isinstance(effect["paulis"], str)
                or len(effect["paulis"]) != len(qs)
                or any(p not in "XYZ" for p in effect["paulis"])
            ):
                raise ValueError("spatial needs one X/Y/Z per qubit")
            _prob(effect["probability"], "spatial probability")
            if effect.get("basis") not in ("measured", "scenario_assumption"):
                raise ValueError("spatial basis must say measured or scenario_assumption")
        dynamic_fields = {
            "drift": {"enabled", "qubits", "pauli", "baseline_probability", "rho", "sigma_logit"},
            "leakage": {
                "enabled",
                "qubits",
                "entry_probability",
                "recovery_probability",
                "reset_removal_probability",
                "effect_probability",
                "neighbor_edges",
                "neighbor_effect_probability",
            },
            "bursts": {
                "enabled",
                "qubits",
                "pauli",
                "onset_probability",
                "recovery_probability",
                "effect_probability",
            },
        }
        for mechanism, fields in dynamic_fields.items():
            effect = obj.setdefault(mechanism, {})
            _keys(effect, fields, mechanism)
            if "enabled" in effect and type(effect["enabled"]) is not bool:
                raise ValueError(f"{mechanism}.enabled must be boolean")
            if effect.get("enabled", False):
                _required(effect, fields - {"neighbor_edges"}, mechanism)
            if "qubits" in effect:
                _qubits(effect["qubits"], mechanism)
            if "pauli" in effect and effect["pauli"] not in ("X", "Y", "Z"):
                raise ValueError(f"{mechanism}.pauli must be X/Y/Z")
            for field in fields - {"enabled", "qubits", "pauli", "neighbor_edges"}:
                if field not in effect:
                    continue
                if field == "sigma_logit":
                    _number(effect[field], field)
                elif field == "rho":
                    _number(effect[field], field, minimum=-1, maximum=1)
                else:
                    _prob(effect[field], f"{mechanism}.{field}")
            if (
                mechanism == "drift"
                and "baseline_probability" in effect
                and not 0 < effect["baseline_probability"] < 1
            ):
                raise ValueError("drift baseline_probability must be strictly between 0 and 1")
            if mechanism == "leakage":
                for edge in _sequence(effect.get("neighbor_edges", []), "neighbor_edges"):
                    if len(_qubits(edge, "neighbor_edges")) != 2:
                        raise ValueError(
                            "neighbor_edges must contain directed [source,target] pairs"
                        )
                    if edge[0] not in effect.get("qubits", []):
                        raise ValueError("Every leakage neighbor source must be a leakage qubit")
        covariates = obj.setdefault("covariates", {})
        _keys(
            covariates,
            {
                "transition_frequency_hz",
                "cryostat_temperature_k",
                "effective_qubit_temperature_k",
                "humidity_relative_fraction",
            },
            "covariates",
        )
        for key, value in covariates.items():
            if key == "transition_frequency_hz":
                for q, frequency in _object(value, key).items():
                    if not q.isdigit() or str(int(q)) != q:
                        raise ValueError("frequency map requires qubit indices")
                    _number(frequency, key, minimum=np.finfo(float).tiny)
            elif key == "humidity_relative_fraction":
                _prob(value, key)
            else:
                _number(value, key, minimum=np.finfo(float).tiny)
        # Refuse NaN/objects and retain an exact JSON-serializable reproduction configuration.
        json.dumps(obj, allow_nan=False)
        return obj

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._config)

    @property
    def dynamic(self) -> bool:
        return any(
            self._config[name].get("enabled", False) for name in ("drift", "leakage", "bursts")
        )


def _probability(config: dict[str, Any], op: str, q: int) -> float:
    return float(config["qubit_overrides"].get(str(q), {}).get(op, config["probabilities"][op]))


def _layers(ideal: stim.Circuit) -> list[list[stim.CircuitInstruction]]:
    if ideal.without_noise() != ideal:
        raise ValueError("Input must be ideal; existing noise would be counted twice")
    result: list[list[stim.CircuitInstruction]] = [[]]
    for instruction in ideal.flattened():
        result[-1].append(instruction)
        if instruction.name == "TICK":
            result.append([])
    if not result[-1]:
        result.pop()
    return result


def _validate_circuit(ideal: stim.Circuit, config: dict[str, Any]) -> set[int]:
    if ideal.without_noise() != ideal:
        raise ValueError("Input must be ideal; existing noise would be counted twice")
    used: set[int] = set()
    used_edges: set[str] = set()
    for instruction in ideal.flattened():
        if instruction.name in ANNOTATIONS:
            continue
        gate = stim.gate_data(instruction.name)
        if instruction.name not in RESETS | MEASUREMENTS and not gate.is_unitary:
            raise ValueError(f"Unsupported instruction: {instruction.name}")
        targets = instruction.targets_copy()
        if gate.is_two_qubit_gate:
            physical_pairs = [
                (a.value, b.value)
                for a, b in zip(targets[::2], targets[1::2], strict=True)
                if a.is_qubit_target and b.is_qubit_target
            ]
            pair_qubits = [q for pair in physical_pairs for q in pair]
            used_edges.update(f"{min(a, b)},{max(a, b)}" for a, b in physical_pairs)
            if len(set(pair_qubits)) != len(pair_qubits):
                raise ValueError("Overlapping two-qubit pairs require separate TICK layers")
        if any(not target.is_qubit_target for target in targets):
            allowed = config["classical_control_policy"] == "ideal_pauli_frame"
            allowed &= instruction.name in {"CX", "CY", "CZ"}
            allowed &= all(
                (a.is_qubit_target or a.is_sweep_bit_target) and b.is_qubit_target
                for a, b in zip(targets[::2], targets[1::2], strict=True)
            )
            if not allowed:
                raise ValueError(f"Unsupported record/sweep/Pauli targets in {instruction.name}")
        used.update(target.value for target in targets if target.is_qubit_target)
    specified = set(map(int, config["qubit_overrides"]))
    for edge in config["edge_overrides"]:
        specified.update(map(int, edge.split(",")))
    for effect in config["spatial"]:
        specified.update(effect["qubits"])
    for mechanism in ("drift", "leakage", "bursts"):
        if config[mechanism].get("enabled", False):
            specified.update(config[mechanism]["qubits"])
            for edge in config[mechanism].get("neighbor_edges", []):
                specified.update(edge)
    if specified - used:
        raise ValueError(f"Profile references unused qubits: {sorted(specified - used)}")
    if unused_edges := set(config["edge_overrides"]) - used_edges:
        raise ValueError(f"Profile references unused gate edges: {sorted(unused_edges)}")
    coherence = config["coherence"]
    if coherence.get("enabled", False):
        if set(map(int, coherence["qubits"])) != used:
            raise ValueError("Enabled coherence needs exactly every used qubit's calibration")
        if len(coherence["layer_durations_s"]) != len(_layers(ideal)):
            raise ValueError("layer_durations_s must have one duration per TICK-separated layer")
        if any(end > len(_layers(ideal)) for end in coherence.get("round_end_layers", [])):
            raise ValueError("round_end_layers exceeds circuit layer count")
    return used


def _static_segments(ideal: stim.Circuit, profile: NoiseProfile) -> list[stim.Circuit]:
    config = profile.to_dict()
    used = _validate_circuit(ideal, config)
    segments = []
    for index, layer in enumerate(_layers(ideal)):
        segment = stim.Circuit()
        active: set[int] = set()
        measured: set[int] = set()
        coherence = config["coherence"]
        pxyz_by_qubit = {}
        if coherence.get("enabled", False):
            for q in sorted(used):
                calibration = coherence["qubits"][str(q)]
                pxyz_by_qubit[q] = coherence_probabilities(
                    coherence["layer_durations_s"][index],
                    calibration["t1_s"],
                    calibration["t2_s"],
                )
        trailing_tick = False
        for instruction in layer:
            name = instruction.name
            if name == "TICK":
                trailing_tick = True
                continue
            if name in ANNOTATIONS:
                segment.append(instruction)
                continue
            targets = [
                target.value for target in instruction.targets_copy() if target.is_qubit_target
            ]
            active.update(targets)
            if name in MEASUREMENTS:
                # Exposure assigned to a measurement layer must precede the measurement.
                # Applying it after the final readout would silently erase its entire effect.
                for q in targets:
                    if q in measured:
                        raise ValueError(
                            "Repeated measurement of one qubit within a layer needs TICK"
                        )
                    if any(pxyz_by_qubit.get(q, ())):
                        segment.append("PAULI_CHANNEL_1", [q], pxyz_by_qubit[q])
                    measured.add(q)
                # Native measurement noise flips the report, not the postmeasurement state.
                # A Pauli before a reused, non-resetting measurement does something different.
                for target in instruction.targets_copy():
                    p = _probability(config, "measurement", target.value)
                    segment.append(name, [target], p if p else [], tag=instruction.tag)
            else:
                segment.append(instruction)
            if name in RESETS:
                error = "Z_ERROR" if name in {"RX", "MRX"} else "X_ERROR"
                for q in targets:
                    if p := _probability(config, "reset", q):
                        segment.append(error, [q], p)
            if stim.gate_data(name).is_unitary:
                if stim.gate_data(name).is_two_qubit_gate:
                    original_targets = instruction.targets_copy()
                    for first, second in zip(
                        original_targets[::2], original_targets[1::2], strict=True
                    ):
                        if first.is_sweep_bit_target:
                            # Explicit ideal frame convention, not an inferred calibrated X gate.
                            continue
                        a, b = first.value, second.value
                        edge = f"{min(a, b)},{max(a, b)}"
                        p = config["edge_overrides"].get(
                            edge, config["probabilities"]["two_qubit_gate"]
                        )
                        if p:
                            segment.append("DEPOLARIZE2", [a, b], p)
                else:
                    for q in targets:
                        if p := _probability(config, "one_qubit_gate", q):
                            segment.append("DEPOLARIZE1", [q], p)
        # Annotations-only tails have no physical exposure and must not add an idle layer.
        if active or trailing_tick:
            for q in sorted(used - active):
                if p := _probability(config, "idle", q):
                    segment.append("DEPOLARIZE1", [q], p)
            for q in sorted(used - measured):
                if any(pxyz_by_qubit.get(q, ())):
                    segment.append("PAULI_CHANNEL_1", [q], pxyz_by_qubit[q])
            for effect in config["spatial"]:
                if not effect.get("enabled", True):
                    continue
                paulis = [
                    getattr(stim, f"target_{p.lower()}")(q)
                    for q, p in zip(effect["qubits"], effect["paulis"], strict=True)
                ]
                segment.append("CORRELATED_ERROR", paulis, effect["probability"])
        if trailing_tick:
            segment.append("TICK")
        segments.append(segment)
    return segments


def build_noisy_circuit(ideal: stim.Circuit, profile: NoiseProfile) -> stim.Circuit:
    """The static model cannot encode hidden-state trajectories; reject their omission."""
    if profile.dynamic:
        raise ValueError(
            "Dynamic profile requires iter_profile_chunks; no exact static DEM is available"
        )
    result = stim.Circuit()
    for segment in _static_segments(ideal, profile):
        result += segment
    return result


RNG_STREAM_NAMES = ("stim", "drift_state", "drift_effect", "burst_state", "burst_effect", "leakage")


def profile_audit(
    ideal: stim.Circuit, profile: NoiseProfile, seed: int, shots: int, chunk_size: int
) -> dict[str, Any]:
    """Describe reproducible assumptions without sampling or exposing hidden trajectories."""
    if type(shots) is not int or shots < 0 or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("shots must be nonnegative and chunk_size positive integers")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    config = profile.to_dict()
    _validate_circuit(ideal, config)
    names = RNG_STREAM_NAMES
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    coherence = config["coherence"]
    duration = sum(coherence["layer_durations_s"]) if coherence.get("enabled", False) else None
    ends = coherence.get("round_end_layers", []) if coherence.get("enabled", False) else []
    round_durations = [
        sum(coherence["layer_durations_s"][a:b])
        for a, b in zip([0, *ends[:-1]][: len(ends)], ends, strict=True)
    ]
    return {
        "noise_schema_version": 1,
        "contract": "A",
        "profile": config,
        "profile_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "ideal_circuit_sha256": hashlib.sha256(str(ideal).encode()).hexdigest(),
        "stim_version": stim.__version__,
        "numpy_version": np.__version__,
        "seed": seed,
        "rng_streams": list(names),
        "rng_derivation": "SeedSequence(seed, sha256(name)[:16] little-endian uint32 spawn_key)",
        "shots": shots,
        "chunk_size": chunk_size,
        "bit_order": "little",
        "n_detectors": ideal.num_detectors,
        "n_observables": ideal.num_observables,
        "backend": "batched_flip_simulator" if profile.dynamic else "compiled_detector_sampler",
        "physical_duration_s": duration,
        "round_durations_s": round_durations,
        "round_rates_hz": [1 / t if t else None for t in round_durations],
        "state_scope": {
            "drift": "acquisition_shots",
            "bursts": "acquisition_shots",
            "leakage": "circuit_layers_reset_each_shot",
        },
        "covariate_response": "none; frequency, temperature and humidity are recorded only",
        "approximations": [
            "stochastic Pauli channels",
            "lumped decoherence before measured targets and after other layer operations",
            "classical leakage-effect proxy, no higher-level state",
            "burst/drift state constant inside each acquisition shot",
        ],
        "contains_mechanism_targets": False,
        "classical_control_policy": config["classical_control_policy"],
        "synthetic_sweep_bits": "all zero; conditional control pulse infidelity not modeled",
    }
