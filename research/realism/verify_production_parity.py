"""Tie the frozen evaluation sampler to production using its actual fitted profiles.

No data are fitted and no held-out hardware outcomes are read. For every configured
cohort, compare exact noisy circuit text and sampled detector/observable bytes under
both implementations, with matching Stim version, machine, seed and chunk size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import stim

from qecgen.dataset import content_hash
from qecgen.noise import NoiseProfile, build_noisy_circuit
from qecgen.sampling import sample_profile
from research.realism import model as frozen


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(evaluation: Path, *, shots: int, seed: int, chunk_size: int) -> dict[str, Any]:
    """Fail closed if the development records omit even one configured cohort or arm."""
    if shots <= 0 or chunk_size <= 0 or seed < 0:
        raise ValueError("shots and chunk_size must be positive; seed must be nonnegative")
    config = json.loads((evaluation / "config.json").read_text(encoding="utf-8"))
    development = json.loads((evaluation / "development.json").read_text(encoding="utf-8"))
    source_hashes = json.loads((evaluation / "source_hashes.json").read_text(encoding="utf-8"))
    if _digest(Path(frozen.__file__)) != source_hashes["model.py"]:
        raise ValueError("The current research model differs from the frozen evaluation source")
    if _digest(evaluation / "source" / "model.py") != source_hashes["model.py"]:
        raise ValueError("The evaluation's archived model failed its recorded checksum")
    configured = {row["name"]: row for row in config["cohorts"]}
    records = {row["name"]: row for row in development}
    if not configured or len(configured) != len(config["cohorts"]):
        raise ValueError("Evaluation must contain distinct configured cohorts")
    if len(records) != len(development) or records.keys() != configured.keys():
        raise ValueError("Development profiles do not yet cover every configured cohort")
    result: dict[str, Any] = {
        "status": "checking",
        "evaluation": str(evaluation.resolve()),
        "evaluation_config_sha256": _digest(evaluation / "config.json"),
        "frozen_model_sha256": source_hashes["model.py"],
        "production_source_sha256": {
            name: _digest(Path(name))
            for name in ("qecgen/noise.py", "qecgen/sampling.py", "qecgen/dataset.py")
        },
        "verifier_sha256": _digest(Path(__file__)),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "stim": stim.__version__,
        },
        "shots_per_case": shots,
        "seed": seed,
        "chunk_size": chunk_size,
        "classical_control_policy": "ideal_pauli_frame",
        "scope": "Static fitted profiles only; does not validate dynamic proxies against hardware",
        "held_out_hardware_outcomes_read": False,
        "content_hash_scope": "Contract A, one environment: environment_ids=None, mechanisms=None",
        "cases": [],
    }
    for name, cohort in configured.items():
        record = records[name]
        profiles = record["profiles"]
        if set(profiles) != {"uniform_matched", "heterogeneous_readout"}:
            raise ValueError(f"Unexpected or missing fitted profile arms for {name}")
        circuit_path = Path(cohort["circuit"])
        if _digest(circuit_path) != cohort["expected"]["circuit_sha256"]:
            raise ValueError(f"Source circuit checksum failed for {name}")
        ideal = stim.Circuit(circuit_path.read_text(encoding="utf-8"))
        for arm, raw in profiles.items():
            if raw.get("classical_control_policy") != "ideal_pauli_frame":
                raise ValueError("Fitted profile must explicitly use the evaluated frame policy")
            production = NoiseProfile.from_dict(raw)
            prototype = frozen.NoiseProfile.from_dict(raw)
            production_circuit = build_noisy_circuit(ideal, production)
            prototype_circuit = frozen.build_noisy_circuit(ideal, prototype)
            exact_circuit = str(production_circuit) == str(prototype_circuit)
            new = sample_profile(ideal, production, shots, seed, chunk_size)
            old = frozen.sample_profile(ideal, prototype, shots, seed, chunk_size)
            detectors_equal = np.array_equal(new.detectors, old.detectors)
            observables_equal = np.array_equal(new.observables, old.observables)
            hashes = [content_hash(sample.detectors, sample.observables) for sample in (new, old)]
            case = {
                "cohort": name,
                "arm": arm,
                "profile": raw,
                "profile_sha256": new.audit["profile_sha256"],
                "ideal_circuit_sha256": cohort["expected"]["circuit_sha256"],
                "noisy_circuit_sha256": [
                    hashlib.sha256(str(c).encode()).hexdigest()
                    for c in (production_circuit, prototype_circuit)
                ],
                "exact_circuit_text_equal": exact_circuit,
                "detector_arrays_equal": detectors_equal,
                "observable_arrays_equal": observables_equal,
                "content_hashes": hashes,
                "n_detectors": ideal.num_detectors,
                "n_observables": ideal.num_observables,
                "equal": exact_circuit
                and detectors_equal
                and observables_equal
                and hashes[0] == hashes[1],
            }
            result["cases"].append(case)
    result["status"] = "passed" if all(case["equal"] for case in result["cases"]) else "failed"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation", type=Path, default=Path("data/realism/results/final-evaluation")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/realism/results/production-parity.json")
    )
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    args = parser.parse_args()
    result = verify(args.evaluation, shots=args.shots, seed=args.seed, chunk_size=args.chunk_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Production parity: {result['status']}; {len(result['cases'])} cases", flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
