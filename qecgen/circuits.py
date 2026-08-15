"""Surface code circuit construction and noise models.

Circuits come from :meth:`stim.Circuit.generated`. This module does **not** contain a
surface code builder; it selects a code task string, chooses a channel vector, and
optionally rewrites noise instructions that ``generated`` cannot express (see
:func:`apply_xz_bias`).
"""

from __future__ import annotations

import enum
import math
from dataclasses import asdict, dataclass
from typing import assert_never

import stim

XZ_BIAS_SCOPE = "all_single_qubit_depolarize1_data_and_gate; depolarize2_unbiased"
"""Exactly which channels an :func:`apply_xz_bias` rewrite touches.

Recorded in the manifest so a results table cannot misattribute the bias to data noise
alone, which is what the documentation previously implied.
"""

__all__ = [
    "XZ_BIAS_SCOPE",
    "Basis",
    "ChannelVector",
    "NoiseModel",
    "apply_xz_bias",
    "build_circuit",
    "build_circuit_from_channels",
    "channels_for",
    "code_task",
]


class Basis(enum.StrEnum):
    """Memory experiment basis."""

    Z = "z"
    X = "x"


class NoiseModel(enum.StrEnum):
    """Named noise conventions.

    Each member documents exactly which of Stim's four channels it sets and which it
    leaves at zero. Swapping conventions moves the threshold, so the resolved
    :class:`ChannelVector` is stored in every manifest rather than only the model name.
    """

    CODE_CAPACITY = "code_capacity"
    """Data qubit depolarisation only.

    Sets ``before_round_data_depolarization = p``. Leaves
    ``after_clifford_depolarization``, ``after_reset_flip_probability`` and
    ``before_measure_flip_probability`` at zero. Forces ``rounds = 1``: with perfect
    measurements, extra rounds add no information.
    """

    PHENOMENOLOGICAL = "phenomenological"
    """Data qubit depolarisation plus measurement flips.

    Sets ``before_round_data_depolarization = p`` and
    ``before_measure_flip_probability = p``. Leaves ``after_clifford_depolarization``
    and ``after_reset_flip_probability`` at zero, i.e. no Clifford gate noise.
    """

    STIM_UNIFORM_CIRCUIT_LEVEL = "stim_uniform_circuit_level"
    """All four Stim channels set to a single ``p``.

    Sets ``after_clifford_depolarization``, ``after_reset_flip_probability``,
    ``before_measure_flip_probability`` and ``before_round_data_depolarization`` all
    to ``p``.

    The name is deliberate. Setting all four channels to one value is **one valid
    synthetic convention**, not a universal definition of "circuit-level noise".
    Hardware-motivated models weight these channels differently, and the threshold
    location depends on the choice.
    """


@dataclass(frozen=True, slots=True)
class ChannelVector:
    """The four noise probabilities :meth:`stim.Circuit.generated` accepts.

    Stored in every manifest so a results table can state exactly which channels were
    active, rather than only a model name whose meaning may drift.
    """

    after_clifford_depolarization: float = 0.0
    after_reset_flip_probability: float = 0.0
    before_measure_flip_probability: float = 0.0
    before_round_data_depolarization: float = 0.0

    def __post_init__(self) -> None:
        """Reject any channel that is not a finite probability.

        This is the single choke point every circuit-building path flows through, and
        it exists because the failure it prevents is silent: Stim accepts NaN channel
        probabilities, produces an all-zero dataset from them, and the resulting file
        passes every structural check while containing no information at all. A NaN
        also serialises to the bare token ``NaN``, which is not valid JSON and which
        strict parsers downstream will reject.
        """
        for name, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(
                    f"channel {name} must be a finite probability, got {value!r}. "
                    "NaN and infinity silently produce all-zero datasets that pass "
                    "validation."
                )
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"channel {name} must lie in [0, 1], got {value!r}")

    def as_dict(self) -> dict[str, float]:
        """Return the channels as a plain dict, suitable for JSON and manifests."""
        return asdict(self)

    def scaled(self, factor: float) -> ChannelVector:
        """Return a copy with every channel multiplied by ``factor``."""
        return ChannelVector(
            after_clifford_depolarization=self.after_clifford_depolarization * factor,
            after_reset_flip_probability=self.after_reset_flip_probability * factor,
            before_measure_flip_probability=self.before_measure_flip_probability * factor,
            before_round_data_depolarization=self.before_round_data_depolarization * factor,
        )

    @property
    def is_noiseless(self) -> bool:
        """True when every channel is exactly zero."""
        return all(v == 0.0 for v in self.as_dict().values())


def code_task(basis: Basis, rotated: bool) -> str:
    """Build the ``surface_code:...`` task string for :meth:`stim.Circuit.generated`."""
    layout = "rotated" if rotated else "unrotated"
    return f"surface_code:{layout}_memory_{basis.value}"


def channels_for(noise_model: NoiseModel, p: float) -> ChannelVector:
    """Resolve a named noise model and a single rate ``p`` into explicit channels."""
    match noise_model:
        case NoiseModel.CODE_CAPACITY:
            return ChannelVector(before_round_data_depolarization=p)
        case NoiseModel.PHENOMENOLOGICAL:
            return ChannelVector(
                before_round_data_depolarization=p,
                before_measure_flip_probability=p,
            )
        case NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL:
            return ChannelVector(
                after_clifford_depolarization=p,
                after_reset_flip_probability=p,
                before_measure_flip_probability=p,
                before_round_data_depolarization=p,
            )
        case _:
            assert_never(noise_model)


def default_rounds(noise_model: NoiseModel, distance: int, rounds: int | None) -> int:
    """Resolve the round count, forcing ``rounds=1`` for code capacity noise.

    Raises:
        ValueError: if ``CODE_CAPACITY`` is combined with an explicit ``rounds != 1``.
            Silently overriding would produce a file whose manifest disagrees with its
            contents.
    """
    if noise_model is NoiseModel.CODE_CAPACITY:
        if rounds is not None and rounds != 1:
            raise ValueError(
                f"CODE_CAPACITY is defined with rounds=1 (measurements are perfect), "
                f"but rounds={rounds} was requested. Pass rounds=1 or omit it."
            )
        return 1
    return distance if rounds is None else rounds


def build_circuit_from_channels(
    distance: int,
    channels: ChannelVector,
    rounds: int,
    basis: Basis = Basis.Z,
    rotated: bool = True,
) -> stim.Circuit:
    """Generate a surface code circuit from an explicit channel vector."""
    if distance < 2:
        raise ValueError(f"distance must be >= 2, got {distance}")
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    return stim.Circuit.generated(
        code_task(basis, rotated),
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=channels.after_clifford_depolarization,
        after_reset_flip_probability=channels.after_reset_flip_probability,
        before_measure_flip_probability=channels.before_measure_flip_probability,
        before_round_data_depolarization=channels.before_round_data_depolarization,
    )


def build_circuit(
    distance: int,
    p: float,
    noise_model: NoiseModel = NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL,
    rounds: int | None = None,
    basis: Basis = Basis.Z,
    rotated: bool = True,
) -> tuple[stim.Circuit, ChannelVector]:
    """Build a surface code memory circuit and return it with its channel vector.

    Args:
        distance: Code distance.
        p: The single rate the noise model is parameterised by.
        noise_model: Which channels ``p`` is applied to. See :class:`NoiseModel`.
        rounds: Number of measurement rounds. Defaults to ``distance``. Must be 1 (or
            omitted) for :attr:`NoiseModel.CODE_CAPACITY`.
        basis: Memory basis, Z or X.
        rotated: Rotated layout when True, unrotated when False.

    Returns:
        The circuit and the resolved :class:`ChannelVector` that produced it. The
        channel vector is returned rather than recomputed downstream so the manifest
        records precisely what was simulated.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    resolved_rounds = default_rounds(noise_model, distance, rounds)
    channels = channels_for(noise_model, p)
    circuit = build_circuit_from_channels(distance, channels, resolved_rounds, basis, rotated)
    return circuit, channels


def apply_xz_bias(circuit: stim.Circuit, eta: float) -> stim.Circuit:
    """Rewrite single-qubit depolarising noise into a Z-biased Pauli channel.

    ``stim.Circuit.generated`` exposes only four *symmetric* scalar probabilities and
    cannot express unequal X/Z rates at all. To sweep a bias axis we therefore rewrite
    ``DEPOLARIZE1(p)`` instructions in the generated circuit into
    ``PAULI_CHANNEL_1(px, py, pz)``.

    The bias is defined as ``eta = pz / (px + py)`` with the total single-qubit error
    probability preserved::

        pz = p * eta / (eta + 1)
        px = py = p / (2 * (eta + 1))

    ``eta = 0.5`` gives ``px = py = pz = p/3``, which is the depolarising point, so it is
    the unbiased reference. Note that the resulting *detector error model* is not
    bit-identical to the one built from ``DEPOLARIZE1``: the two agree on mechanism
    support exactly and on probabilities to first order, but Stim composes the channels
    through slightly different arithmetic, leaving an O(p^2) residual (measured at
    ``diff/p^2 ~= 0.356``, constant across p = 1e-2, 1e-3, 1e-4). An isolated
    single-channel comparison agrees to 1e-16, confirming the formula rather than the
    composition is exact.

    **Scope, stated precisely.** This rewrites **every** ``DEPOLARIZE1`` instruction in
    the circuit. In a Stim-generated surface code that means both
    ``before_round_data_depolarization`` (data qubit noise) **and**
    ``after_clifford_depolarization`` applied after single-qubit Cliffords. An earlier
    version of this docstring claimed only data noise was affected; that was wrong, and
    a gate-noise-only circuit demonstrates the difference.

    Two-qubit ``DEPOLARIZE2`` noise **is** left symmetric. Biasing it correctly requires
    the 15-parameter ``PAULI_CHANNEL_2`` and a convention for correlated two-qubit bias
    that nobody has specified. So a dataset on this axis carries biased single-qubit
    noise (data and gate) and unbiased two-qubit gate noise. The manifest records this
    as ``bias_scope``.

    Args:
        circuit: Any circuit, including one containing ``REPEAT`` blocks.
        eta: Bias ratio, must be positive.

    Returns:
        A new circuit; the input is not modified.
    """
    # `eta <= 0.0` alone admits NaN (every comparison with NaN is False) and +inf
    # (pz becomes inf/inf = NaN), which then fail deep inside stim with an opaque
    # "wasn't a probability" — or succeed silently on a circuit with no DEPOLARIZE1.
    if not (math.isfinite(eta) and eta > 0.0):
        raise ValueError(f"eta must be a finite positive number, got {eta}")

    out = stim.Circuit()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            out.append(
                stim.CircuitRepeatBlock(
                    instruction.repeat_count,
                    apply_xz_bias(instruction.body_copy(), eta),
                )
            )
            continue

        if instruction.name == "DEPOLARIZE1":
            args = instruction.gate_args_copy()
            if len(args) == 1:
                p = args[0]
                pz = p * eta / (eta + 1.0)
                pxy = p / (2.0 * (eta + 1.0))
                out.append("PAULI_CHANNEL_1", instruction.targets_copy(), [pxy, pxy, pz])
                continue

        out.append(instruction)
    return out
