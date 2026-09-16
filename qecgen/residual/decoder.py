"""One frozen PyMatching model per residual dataset, and the chunk decoding it performs.

A residual dataset's labels (``pm_guess``, ``pm_weight``, ``pm_wrong``) are only as
reproducible as the decoder that produced them, so this module makes the decoder an
*artifact*: a :class:`DecoderModel` carries the DEM text that will be published, its
hashes, the ``pymatching.Matching`` built from it, the circuit whose coordinates define
the time slices, and a provenance record naming exactly how the DEM was obtained. Four
traps shape it:

* **The text is the model, not the in-memory object.** Stim prints shortest round-trip
  floats, so a DEM parsed back from its own text is *not* equal to the object it was
  printed from (measured: probabilities differ at the 1e-18 level, ``approx_equals`` with
  ``atol=0`` is false). A matcher built from the in-memory object and a matcher rebuilt
  from the published ``.dem`` would then carry different edge weights and could resolve a
  tie differently. Every builder therefore serialises first and builds the matcher from
  the parsed *text*, which is a fixed point of ``str``; validation's "rebuild from the
  published file" then reproduces the labels exactly rather than approximately.
* **Exactly one logical observable.** A second observable makes ``pm_guess``/``truth``
  arrays rather than bits, and reducing them to one bit (XOR, first column, ...) would be
  a silent redefinition of the target. :class:`MultiObservableError` stops the source and
  spells out the schema change that would be needed instead.
* **Little-endian everywhere.** Predictions come back bit-packed and are unpacked with
  :func:`qecgen.sampling.unpack_bits`; a NumPy default here would flip every byte's bits
  and still return well-formed 0/1 values.
* **A dynamic profile has no exact DEM.** :func:`frozen_reference_profile` builds a
  *reference* model from the stationary point of the drift process and drops bursts and
  leakage outright, and says so in a transformation text that travels with the model.
  Nothing here claims the reference DEM describes the sampled process.

Correlated matching is refused in this schema version: ``enable_correlations=True`` is a
different decoder configuration whose results must never be mixed with standard matching
(the brief's separate-configuration rule).
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pymatching
import stim

from qecgen.dataset import dem_digest
from qecgen.noise import NoiseProfile, build_noisy_circuit
from qecgen.residual.config import DecoderKind
from qecgen.sampling import packed_width, unpack_bits

__all__ = [
    "DecoderModel",
    "MultiObservableError",
    "decode_chunk",
    "dem_stats",
    "from_circuit",
    "from_frozen_reference",
    "from_official_dem",
    "from_static_profile",
    "frozen_reference_profile",
    "pm_wrong",
    "truth_from_packed",
]

MULTI_OBSERVABLE_SCHEMA = (
    "The residual feature schema (version 1) is defined for exactly one logical observable: "
    "`truth`, `pm_guess` and `pm_wrong` are single bits. Reducing an observable array to one "
    "bit (XOR, first column, any-flip) would silently redefine the target, so this source is "
    "stopped instead. A multi-observable dataset needs an explicit schema revision: one column "
    "per observable for the truth (`truth_0` .. `truth_k`) and for PyMatching's guess "
    "(`pm_guess_0` .. `pm_guess_k`), with `pm_wrong` defined as any mismatch between the two "
    "vectors (`pm_wrong = any(pm_guess_i != truth_i)`), a bumped `SCHEMA_VERSION`, and the "
    "same columns in every dataset that shares the schema."
)


class MultiObservableError(ValueError):
    """More than one logical observable: the source is stopped, never reduced to one bit."""

    def __init__(self, n_observables: int, where: str) -> None:
        self.n_observables = n_observables
        super().__init__(
            f"{where} carries {n_observables} logical observables. {MULTI_OBSERVABLE_SCHEMA}"
        )


@dataclass(frozen=True, eq=False)
class DecoderModel:
    """A frozen decoder: the DEM text of record, its matcher, and how it was obtained.

    ``eq=False`` because two models are the same model iff their ``dem_sha256`` agree;
    dataclass equality would compare ``pymatching.Matching`` objects by identity and
    ``stim.DetectorErrorModel`` objects by float bits, neither of which is the question.
    """

    kind: DecoderKind
    dem: stim.DetectorErrorModel
    """Parsed from ``dem_text``; the object the matcher was built from."""

    dem_text: str
    """The exact text that will be published as ``<name>_decoder.dem``."""

    dem_sha256: str
    """sha256 of ``dem_text.encode("utf-8")``; for an official DEM, of the member bytes."""

    dem_blake2b128: str
    """:func:`qecgen.dataset.dem_digest` of ``dem_text``, the digest qecgen manifests use."""

    matching: pymatching.Matching
    n_detectors: int
    n_observables: int
    circuit: stim.Circuit
    """The circuit whose detector coordinates define the time slices.

    For circuit-derived models this is the noisy circuit the DEM came from; for an
    official DEM it is the ideal circuit whose coordinates were verified equal to the DEM's.
    """

    provenance: dict[str, Any] = field(default_factory=dict)
    enable_correlations: bool = False

    def __post_init__(self) -> None:
        if self.enable_correlations:
            raise ValueError(
                "enable_correlations=True is a separate decoder configuration; schema "
                "version 1 records standard matching only and never mixes the two"
            )
        if self.n_observables != 1:
            raise MultiObservableError(self.n_observables, "DecoderModel")
        if self.dem.num_observables != 1:
            raise MultiObservableError(self.dem.num_observables, "DEM")
        if self.circuit.num_observables != 1:
            raise MultiObservableError(self.circuit.num_observables, "circuit")
        counts = {
            "dem": self.dem.num_detectors,
            "matching": self.matching.num_detectors,
            "circuit": self.circuit.num_detectors,
            "n_detectors": self.n_detectors,
        }
        if len(set(counts.values())) != 1:
            raise ValueError(f"detector counts disagree: {counts}")
        if self.matching.num_fault_ids != 1:
            raise ValueError(
                f"matching carries {self.matching.num_fault_ids} fault ids; expected exactly 1"
            )
        if self.dem_sha256 != hashlib.sha256(self.dem_text.encode("utf-8")).hexdigest():
            raise ValueError("dem_sha256 does not hash dem_text")
        if self.dem_blake2b128 != dem_digest(self.dem_text):
            raise ValueError("dem_blake2b128 does not digest dem_text")
        json.dumps(self.provenance, sort_keys=True, allow_nan=False)


def dem_stats(dem: stim.DetectorErrorModel) -> dict[str, int]:
    """Count what PyMatching will and will not decode in ``dem``.

    PyMatching splits each ``error`` at ``^`` separators and ignores any component whose
    *raw* detector-target count exceeds two (it does not XOR-reduce repeated targets
    first, unlike :func:`qecgen.dem.parse_dem`). ``components_gt2_detectors`` is
    therefore the count of ignored hyperedges; ``errors_gt2_detectors_without_separator``
    is the older, coarser view of the same thing (an undecomposed correlated error) and is
    kept because the note reports it.
    """
    num_errors = 0
    with_separator = 0
    gt2 = 0
    gt2_without_separator = 0
    hyper_components = 0
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        num_errors += 1
        components: list[list[int]] = [[]]
        for target in instruction.targets_copy():
            if target.is_separator():
                components.append([])
            elif target.is_relative_detector_id():
                components[-1].append(int(target.val))
        has_separator = len(components) > 1
        with_separator += has_separator
        n_targets = sum(len(component) for component in components)
        if n_targets > 2:
            gt2 += 1
            gt2_without_separator += not has_separator
        hyper_components += sum(len(component) > 2 for component in components)
    return {
        "num_errors": num_errors,
        "errors_with_separator": with_separator,
        "errors_gt2_detectors": gt2,
        "errors_gt2_detectors_without_separator": gt2_without_separator,
        "components_gt2_detectors": hyper_components,
    }


def _merge_provenance(caller: dict[str, Any], computed: dict[str, Any]) -> dict[str, Any]:
    """Refuse a caller-supplied key that would shadow a computed fact.

    A caller passing ``circuit_sha256`` would otherwise silently replace the value this
    module measured with one it merely asserts, and the metadata file would look
    verified while recording an unverified claim.
    """
    collisions = sorted(set(caller) & set(computed))
    if collisions:
        raise ValueError(f"provenance keys are computed here and cannot be supplied: {collisions}")
    return {**copy.deepcopy(caller), **computed}


def _assemble(
    *,
    kind: DecoderKind,
    dem_text: str,
    dem_sha256: str,
    circuit: stim.Circuit,
    provenance: dict[str, Any],
) -> DecoderModel:
    """Parse the text of record, build the matcher from it, and freeze the model."""
    dem = stim.DetectorErrorModel(dem_text)
    if dem.num_observables != 1:
        raise MultiObservableError(dem.num_observables, f"{kind.value} DEM")
    if circuit.num_observables != 1:
        raise MultiObservableError(circuit.num_observables, "circuit")
    if dem.num_detectors != circuit.num_detectors:
        raise ValueError(
            f"DEM declares {dem.num_detectors} detectors but the circuit has "
            f"{circuit.num_detectors}"
        )
    matching = pymatching.Matching.from_detector_error_model(dem)
    stats = dem_stats(dem)
    computed = {
        "dem_stats": stats,
        "num_detectors": dem.num_detectors,
        "num_observables": dem.num_observables,
        "num_errors": dem.num_errors,
        "matching_num_edges": int(matching.num_edges),
        "matching_construction": "pymatching.Matching.from_detector_error_model(dem)",
        "enable_correlations": False,
        "matching_mode": "standard matching (enable_correlations=False)",
        "stim_version": stim.__version__,
        "pymatching_version": pymatching.__version__,
    }
    return DecoderModel(
        kind=kind,
        dem=dem,
        dem_text=dem_text,
        dem_sha256=dem_sha256,
        dem_blake2b128=dem_digest(dem_text),
        matching=matching,
        n_detectors=dem.num_detectors,
        n_observables=dem.num_observables,
        circuit=circuit,
        provenance=_merge_provenance(provenance, computed),
    )


def _circuit_sha256(circuit: stim.Circuit) -> str:
    """The convention ``configuration.write_configured`` records as ``circuit_sha256``."""
    return hashlib.sha256(str(circuit).encode()).hexdigest()


def _profile_sha256(profile: NoiseProfile) -> str:
    """The convention ``noise.profile_audit`` records as ``profile_sha256``."""
    payload = json.dumps(profile.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _from_noisy_circuit(
    circuit: stim.Circuit, kind: DecoderKind, provenance: dict[str, Any]
) -> DecoderModel:
    if circuit.num_observables != 1:
        raise MultiObservableError(circuit.num_observables, "circuit")
    dem_text = str(circuit.detector_error_model(decompose_errors=True))
    computed = {
        "method": "circuit.detector_error_model(decompose_errors=True)",
        "decompose_errors": True,
        "circuit_sha256": _circuit_sha256(circuit),
        "fitted_in_this_pipeline": False,
        "third_party_fitting": None,
    }
    return _assemble(
        kind=kind,
        dem_text=dem_text,
        dem_sha256=hashlib.sha256(dem_text.encode("utf-8")).hexdigest(),
        circuit=circuit,
        provenance=_merge_provenance(provenance, computed),
    )


def from_circuit(
    circuit: stim.Circuit, kind: DecoderKind, provenance: dict[str, Any]
) -> DecoderModel:
    """Exact DEM of a noisy Stim circuit (the legacy independent-noise path)."""
    if kind is DecoderKind.OFFICIAL_DEM:
        raise ValueError("an official DEM is not derived from a circuit; use from_official_dem")
    return _from_noisy_circuit(circuit, kind, provenance)


def from_static_profile(
    ideal: stim.Circuit, profile: NoiseProfile, provenance: dict[str, Any]
) -> DecoderModel:
    """Exact DEM of the one fixed noisy circuit a static device profile produces.

    ``circuit_sha256`` in the provenance is computed the way ``configuration`` records
    ``generation_audit.circuit_sha256``, so a source's manifest can be checked against it.
    """
    if profile.dynamic:
        raise ValueError(
            "Dynamic profile has no exact static DEM; use from_frozen_reference, which "
            "records the transformation it applies"
        )
    noisy = build_noisy_circuit(ideal, profile)
    computed = {
        "ideal_circuit_sha256": _circuit_sha256(ideal),
        "profile_sha256": _profile_sha256(profile),
        "profile": profile.to_dict(),
        "exact_dem": True,
    }
    return _from_noisy_circuit(
        noisy, DecoderKind.STATIC_PROFILE_DEM, _merge_provenance(provenance, computed)
    )


def frozen_reference_profile(profile: NoiseProfile) -> tuple[NoiseProfile, str]:
    """Freeze a dynamic profile at its stationary point, dropping what has no static form.

    The sampler injects the drift Pauli once per circuit layer on every drift qubit with
    a per-shot probability that is the logistic of ``logit(baseline) + state``; at the
    process's stationary point (state 0) that probability is exactly
    ``baseline_probability``, and a static single-qubit ``spatial`` term with the same
    Pauli and probability is appended at the same position in every layer by
    ``noise._static_segments``. That is a representable static effect and is kept.

    Bursts (a hidden on/off state shared by a whole shot) and leakage (a persistent
    per-qubit state with reset removal and neighbour effects) have no per-layer static
    Bernoulli representation: their marginal would be a *mixture* over hidden state, not
    an independent channel, and any single number written here would be an assumption
    the reference model then presents as a fact. They are disabled outright, and the
    returned text says so, so the omission is stated rather than smoothed over.
    """
    if not profile.dynamic:
        raise ValueError("frozen_reference_profile needs a dynamic profile; this one is static")
    config = profile.to_dict()
    drift = config["drift"]
    lines = [
        "Frozen reference profile derived from the dynamic generation profile.",
        "Kept unchanged: probabilities, qubit_overrides, edge_overrides, spatial, coherence, "
        "covariates, classical_control_policy.",
    ]
    spatial = list(config["spatial"])
    if drift.get("enabled", False):
        for q in drift["qubits"]:
            spatial.append(
                {
                    "qubits": [q],
                    "paulis": drift["pauli"],
                    "probability": drift["baseline_probability"],
                    "basis": "scenario_assumption",
                    "enabled": True,
                }
            )
        lines.append(
            "drift: disabled; represented at its stationary point (drift state 0, where the "
            f"per-layer probability equals baseline_probability={drift['baseline_probability']}) "
            f"as one static single-qubit spatial term per drift qubit {list(drift['qubits'])} "
            f"with Pauli {drift['pauli']} and probability {drift['baseline_probability']} "
            "(basis scenario_assumption). The AR(1) logit fluctuation (rho, sigma_logit) is "
            "not represented."
        )
    else:
        lines.append("drift: not enabled in the dynamic profile; nothing to represent.")
    for mechanism in ("bursts", "leakage"):
        if config[mechanism].get("enabled", False):
            lines.append(
                f"{mechanism}: disabled and omitted; a hidden-state mechanism has no static "
                "per-layer representation without a hidden-state assumption, so no static "
                "term is substituted for it."
            )
        else:
            lines.append(f"{mechanism}: not enabled in the dynamic profile; nothing to omit.")
    lines.append(
        "The resulting DEM is a reference model for decoding, not an exact DEM of the "
        "dynamic sampling process."
    )
    config["spatial"] = spatial
    for mechanism in ("drift", "bursts", "leakage"):
        config[mechanism] = {"enabled": False}
    frozen = NoiseProfile.from_dict(config)
    if frozen.dynamic:  # pragma: no cover - NoiseProfile would have to change its own rule
        raise RuntimeError("frozen profile still reports dynamic")
    return frozen, "\n".join(lines)


def from_frozen_reference(
    ideal: stim.Circuit, dynamic_profile: NoiseProfile, provenance: dict[str, Any]
) -> DecoderModel:
    """Reference DEM for a dynamic profile, with the transformation recorded beside it."""
    if not dynamic_profile.dynamic:
        raise ValueError(
            "from_frozen_reference needs a dynamic profile; a static profile has an exact "
            "DEM and belongs to from_static_profile"
        )
    frozen, transformation = frozen_reference_profile(dynamic_profile)
    noisy = build_noisy_circuit(ideal, frozen)
    computed = {
        "ideal_circuit_sha256": _circuit_sha256(ideal),
        "dynamic_profile_sha256": _profile_sha256(dynamic_profile),
        "dynamic_profile": dynamic_profile.to_dict(),
        "frozen_profile_sha256": _profile_sha256(frozen),
        "frozen_profile": frozen.to_dict(),
        "transformation": transformation,
        "exact_dem": False,
        "claim": (
            "frozen reference model built from the drift stationary point; bursts and "
            "leakage omitted; not an exact DEM of the dynamic process"
        ),
    }
    return _from_noisy_circuit(
        noisy, DecoderKind.FROZEN_REFERENCE_DEM, _merge_provenance(provenance, computed)
    )


def from_official_dem(
    dem_bytes: bytes, circuit: stim.Circuit, provenance: dict[str, Any]
) -> DecoderModel:
    """A shipped DEM used verbatim: the member bytes are the text of record.

    The bytes are kept exactly (line endings included) so the published ``.dem`` is the
    member itself and hashes to the recorded member sha256. The DEM's detector
    coordinates must equal the circuit's for every detector: the circuit is what defines
    the time slices and the b8 bit order, and a DEM whose detectors are numbered
    differently would decode a permuted syndrome without any other symptom.
    """
    dem_text = dem_bytes.decode("utf-8")
    if dem_text.encode("utf-8") != dem_bytes:  # pragma: no cover - decode is lossless
        raise ValueError("official DEM bytes do not round-trip through UTF-8")
    dem = stim.DetectorErrorModel(dem_text)
    if dem.num_observables != 1:
        raise MultiObservableError(dem.num_observables, "official DEM")
    if dem.num_detectors != circuit.num_detectors:
        raise ValueError(
            f"official DEM declares {dem.num_detectors} detectors but the circuit has "
            f"{circuit.num_detectors}"
        )
    dem_coordinates = dem.get_detector_coordinates()
    circuit_coordinates = circuit.get_detector_coordinates()
    mismatched = [
        d
        for d in range(circuit.num_detectors)
        if list(dem_coordinates.get(d, [])) != list(circuit_coordinates.get(d, []))
    ]
    if mismatched:
        raise ValueError(
            f"official DEM detector coordinates differ from the circuit's for "
            f"{len(mismatched)} detector(s) (first: D{mismatched[0]}: "
            f"{dem_coordinates.get(mismatched[0])} vs {circuit_coordinates.get(mismatched[0])})"
        )
    computed = {
        "method": "official DEM used verbatim; matching graph and weights are the publisher's",
        "coordinates_equal_circuit": True,
        "circuit_sha256": _circuit_sha256(circuit),
        "fitted_in_this_pipeline": False,
    }
    return _assemble(
        kind=DecoderKind.OFFICIAL_DEM,
        dem_text=dem_text,
        dem_sha256=hashlib.sha256(dem_bytes).hexdigest(),
        circuit=circuit,
        provenance=_merge_provenance(provenance, computed),
    )


def decode_chunk(
    model: DecoderModel, detectors_packed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Decode one packed chunk: ``(pm_guess uint8 (n,), pm_weight float64 (n,))``.

    The packed width is checked against the model's true detector count before
    PyMatching sees it: PyMatching only checks the byte count against its own node
    count, so a chunk packed for a different circuit of the same byte width would decode
    as a scrambled syndrome. Predictions are unpacked with the little-endian helper; the
    weights are PyMatching's summed edge weights, not a fault count.
    """
    if detectors_packed.dtype != np.uint8:
        raise ValueError(f"packed detectors must be uint8, got {detectors_packed.dtype}")
    expected = packed_width(model.n_detectors)
    if detectors_packed.ndim != 2 or detectors_packed.shape[1] != expected:
        raise ValueError(
            f"packed detectors have shape {detectors_packed.shape}; expected "
            f"(rows, {expected}) for {model.n_detectors} detectors"
        )
    n = detectors_packed.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.float64)
    predictions, weights = model.matching.decode_batch(
        detectors_packed,
        return_weights=True,
        bit_packed_shots=True,
        bit_packed_predictions=True,
        enable_correlations=False,
    )
    packed_predictions = np.asarray(predictions, dtype=np.uint8)
    if packed_predictions.shape != (n, packed_width(model.n_observables)):
        raise ValueError(
            f"PyMatching returned predictions of shape {packed_predictions.shape} for {n} shots"
        )
    guess = unpack_bits(packed_predictions, model.n_observables)[:, 0].astype(np.uint8)
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    if guess.shape != (n,) or weight.shape != (n,):
        raise ValueError(
            f"decoded {guess.shape[0]} guesses and {weight.shape[0]} weights for {n} shots"
        )
    if not np.all(np.isfinite(weight)) or np.any(weight < 0):
        raise ValueError("PyMatching returned a non-finite or negative solution weight")
    return guess, weight


def _binary_1d(values: np.ndarray, name: str) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {values.shape}")
    out = np.asarray(values)
    if out.dtype == np.bool_:
        out = out.astype(np.uint8)
    if not np.issubdtype(out.dtype, np.integer) or not np.isin(out, (0, 1)).all():
        raise ValueError(f"{name} must be binary (0/1) integers")
    return out.astype(np.uint8)


def pm_wrong(pm_guess: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """``uint8 (pm_guess != truth)``, computed only after features and decoding exist.

    Shapes are compared before values: a broadcast between ``(n,)`` and ``(n, 1)`` would
    give an ``(n, n)`` matrix that ``.astype(uint8)`` accepts without complaint.
    """
    if pm_guess.shape != truth.shape:
        raise ValueError(f"pm_guess shape {pm_guess.shape} != truth shape {truth.shape}")
    guess = _binary_1d(pm_guess, "pm_guess")
    actual = _binary_1d(truth, "truth")
    return np.asarray(guess != actual, dtype=np.uint8)


def truth_from_packed(observables_packed: np.ndarray, n_observables: int) -> np.ndarray:
    """The recorded logical flip as ``uint8 (n,)``; refuses anything but one observable."""
    if n_observables != 1:
        raise MultiObservableError(n_observables, "observables array")
    if observables_packed.dtype != np.uint8:
        raise ValueError(f"packed observables must be uint8, got {observables_packed.dtype}")
    return unpack_bits(observables_packed, 1)[:, 0].astype(np.uint8)
