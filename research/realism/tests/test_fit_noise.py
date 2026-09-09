"""Analytic detector means must combine faults by parity rather than addition."""

from __future__ import annotations

import numpy as np
import stim

from research.realism.fit_noise import detector_marginals


def test_two_faults_cancel_in_detector_mean() -> None:
    circuit = stim.Circuit("R 0\nX_ERROR(0.2) 0\nX_ERROR(0.3) 0\nM 0\nDETECTOR rec[-1]")
    np.testing.assert_allclose(detector_marginals(circuit), [0.38], atol=1e-12)


def test_correlated_fault_changes_both_means_without_becoming_two_draws() -> None:
    circuit = stim.Circuit(
        "R 0 1\nCORRELATED_ERROR(0.2) X0 X1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]"
    )
    np.testing.assert_allclose(detector_marginals(circuit), [0.2, 0.2], atol=1e-12)
