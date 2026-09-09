"""Read-only capture of legacy output semantics before the realism pilot.

Run ``python -m research.realism.legacy`` from the checkout root to inspect the
current fingerprints. This command prints a candidate; it deliberately cannot
overwrite the checked-in reference, because regenerating expectations during a
test would turn a regression guard into a comparison of a function with itself.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import stim

from qecgen.circuits import Basis, NoiseModel
from qecgen.dataset import StructureLevel, git_commit, library_versions
from qecgen.environments import build_single_environment


class LegacyConfig(TypedDict):
    distance: int
    p: float
    shots: int
    seed: int
    noise_model: str
    rounds: int
    basis: str
    rotated: bool
    chunk_size: int
    emit_mechanisms: bool


def runtime_contract() -> dict[str, str]:
    """Record SIMD selection as well as the usual machine and version details.

    Stim explicitly restricts seeded reproducibility by SIMD width. AMD64 alone
    does not identify that width. The private detector is confined to this pilot
    evidence tool and is safe only under the exact Stim pin asserted by the test.
    """
    detector = stim._detect_machine_architecture
    return {
        "python": platform.python_version(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "stim_march": str(detector._UNSTABLE_detect_march()),
        "stim": stim.__version__,
        "numpy": np.__version__,
    }


def _array_fingerprint(array: npt.NDArray[np.generic] | None) -> dict[str, object] | None:
    if array is None:
        return None
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
    }


def fingerprint(config: LegacyConfig) -> dict[str, object]:
    dataset = build_single_environment(
        distance=config["distance"],
        p=config["p"],
        shots=config["shots"],
        seed=config["seed"],
        noise_model=NoiseModel(config["noise_model"]),
        rounds=config["rounds"],
        basis=Basis(config["basis"]),
        rotated=config["rotated"],
        chunk_size=config["chunk_size"],
        emit_mechanisms=config["emit_mechanisms"],
        structure_level=StructureLevel.DEM,
    )
    environment = dataset.meta.environments[0]
    assert environment.channels is not None
    return {
        "circuit_sha256": hashlib.sha256(environment.circuit.encode("utf-8")).hexdigest(),
        "dem_sha256": hashlib.sha256(environment.dem.encode("utf-8")).hexdigest(),
        "channels": environment.channels.as_dict(),
        "n_detectors": dataset.meta.n_detectors,
        "n_observables": dataset.meta.n_observables,
        "n_mechanisms": dataset.meta.n_mechanisms,
        "contract": str(dataset.meta.contract),
        "bit_order": dataset.meta.bit_order,
        "content_hash": dataset.meta.content_hash,
        "detectors": _array_fingerprint(dataset.detectors),
        "observables": _array_fingerprint(dataset.observables),
        "environment_ids": _array_fingerprint(dataset.environment_ids),
        "mechanisms": _array_fingerprint(dataset.mechanisms),
    }


def capture() -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for model in NoiseModel:
        for basis in Basis:
            for emit_mechanisms in (False, True):
                for seed, chunk_size in ((0, 31), (20260909, 64)):
                    config: LegacyConfig = {
                        "distance": 3,
                        "p": 0.013,
                        "shots": 257,
                        "seed": seed,
                        "noise_model": str(model),
                        "rounds": 1 if model is NoiseModel.CODE_CAPACITY else 3,
                        "basis": str(basis),
                        "rotated": True,
                        "chunk_size": chunk_size,
                        "emit_mechanisms": emit_mechanisms,
                    }
                    cases.append({"config": config, "fingerprint": fingerprint(config)})
    root = Path(__file__).resolve().parents[2]
    return {
        "reference_version": 1,
        "captured_utc_date": "2026-09-09",
        "runtime": runtime_contract(),
        "versions": library_versions(),
        "source_commit": git_commit(root),
        "source_files_sha256": {
            relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in (
                "qecgen/circuits.py",
                "qecgen/sampling.py",
                "qecgen/environments.py",
                "qecgen/dataset.py",
                "qecgen/dem.py",
            )
        },
        "call_contract": {
            "entrypoint": "qecgen.environments.build_single_environment",
            "seed": "Passed unchanged to one compiled sampler; no child-seed derivation.",
            "contract_a": "Circuit detector sampler, separate_observables=True, bit_packed=True.",
            "contract_b": "DEM sampler, bit_packed=True, return_errors=True, all arrays one draw.",
            "chunks": "Repeated chunk_size then remainder; sampler compiled once per dataset.",
            "hashes": (
                "SHA256 of UTF-8 circuit/DEM; SHA256 of C-order array bytes with dtype/shape."
            ),
            "content_hash": (
                "Existing qecgen BLAKE2b-256 array-content hash, including absent arrays."
            ),
            "exclusions": "No timestamps, output-format bytes, or metadata git commit in digests.",
            "portability": (
                "Seeded array references require matching Stim, NumPy, platform and SIMD."
            ),
        },
        "cases": cases,
    }


if __name__ == "__main__":
    print(json.dumps(capture(), indent=2, allow_nan=False))
