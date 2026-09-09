"""Train-only marginal calibration, not identification of physical error causes.

An independent DEM gives E[detector] = (1-prod_j(1-2*p_j))/2 over mechanisms
touching that detector. Fitting these marginals cannot identify all correlations.
The same training statistics calibrate both the uniform and heterogeneous baselines.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import stim
from numpy.typing import NDArray
from scipy.optimize import least_squares, minimize_scalar

from research.realism.model import NoiseProfile, build_noisy_circuit


def detector_marginals(circuit: stim.Circuit) -> NDArray[np.float64]:
    dem = circuit.detector_error_model().flattened()
    products = np.ones(circuit.num_detectors, dtype=np.float64)
    for instruction in dem:
        if instruction.type != "error":
            continue
        support: set[int] = set()
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                support.symmetric_difference_update([target.val])
        probability = float(instruction.args_copy()[0])
        for detector in support:
            products[detector] *= 1 - 2 * probability
    return (1 - products) / 2


def uniform_profile(probability: float) -> NoiseProfile:
    config = NoiseProfile.uniform(probability).to_dict()
    config["classical_control_policy"] = "ideal_pauli_frame"
    config["label"] = "Uniform effective operation probabilities fitted on real TRAIN only"
    return NoiseProfile.from_dict(config)


def heterogeneous_profile(
    probability: float, qubits: list[int], log_factors: NDArray[np.float64]
) -> NoiseProfile:
    config = uniform_profile(probability).to_dict()
    config["label"] = "Regularized readout heterogeneity fitted on real TRAIN marginals only"
    config["qubit_overrides"] = {
        str(qubit): {"measurement": float(probability * np.exp(factor))}
        for qubit, factor in zip(qubits, log_factors, strict=True)
    }
    return NoiseProfile.from_dict(config)


def fit_profiles(
    ideal: stim.Circuit, training_detectors: NDArray[np.bool_], *, regularization: float = 0.01
) -> tuple[NoiseProfile, NoiseProfile, dict[str, Any]]:
    if training_detectors.ndim != 2 or training_detectors.shape[1] != ideal.num_detectors:
        raise ValueError("training detector dimensions disagree with the actual circuit")
    if len(training_detectors) < 100 or regularization <= 0:
        raise ValueError("calibration requires >=100 training shots and positive regularization")
    target = training_detectors.mean(axis=0)

    def objective(log_p: float) -> float:
        model = build_noisy_circuit(ideal, uniform_profile(float(np.exp(log_p))))
        return float(np.mean((detector_marginals(model) - target) ** 2))

    fitted = minimize_scalar(
        objective,
        bounds=(np.log(1e-5), np.log(0.05)),
        method="bounded",
        options={"xatol": 0.005, "maxiter": 30},
    )
    if not fitted.success:
        raise RuntimeError(f"uniform fitting did not converge: {fitted.message}")
    probability = float(np.exp(fitted.x))
    measured = sorted(
        {
            t.value
            for op in ideal.flattened()
            if op.name in {"M", "MX", "MY", "MR", "MRX", "MRY"}
            for t in op.targets_copy()
            if t.is_qubit_target
        }
    )

    def residual(factors: NDArray[np.float64]) -> NDArray[np.float64]:
        profile = heterogeneous_profile(probability, measured, factors)
        errors = detector_marginals(build_noisy_circuit(ideal, profile)) - target
        return np.concatenate([errors / np.sqrt(ideal.num_detectors), regularization * factors])

    result = least_squares(
        residual,
        np.zeros(len(measured)),
        bounds=(-2.0, 2.0),
        max_nfev=30,
        ftol=1e-5,
        xtol=1e-5,
        gtol=1e-5,
    )
    if not result.success:
        raise RuntimeError(f"heterogeneous fitting did not converge: {result.message}")
    uniform = uniform_profile(probability)
    heterogeneous = heterogeneous_profile(probability, measured, result.x)
    return (
        uniform,
        heterogeneous,
        {
            "fit_partition": "train",
            "train_shots": len(training_detectors),
            "uniform_p": probability,
            "uniform_probability_bounds": [1e-5, 0.05],
            "regularization": regularization,
            "measurement_qubits": measured,
            "log_measurement_factors": result.x.tolist(),
            "factor_bounds": [float(np.exp(-2)), float(np.exp(2))],
            "uniform_train_marginal_rmse": float(np.sqrt(objective(float(fitted.x)))),
            "heterogeneous_train_marginal_rmse": float(
                np.sqrt(
                    np.mean(
                        (detector_marginals(build_noisy_circuit(ideal, heterogeneous)) - target)
                        ** 2
                    )
                )
            ),
            "uniform_converged": bool(fitted.success),
            "heterogeneous_converged": bool(result.success),
            "physical_interpretation": (
                "effective readout heterogeneity; not unique physical-cause estimates"
            ),
            "source_dem_weights_used": False,
            "unmodelled": [
                "calibrated leakage",
                "thermal state dependence",
                "coherent errors",
                "hardware-time drift",
            ],
        },
    )
