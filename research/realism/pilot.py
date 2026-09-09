"""Config-driven development comparison; the reserved real test set is never scored.

Validation selects decoder checkpoints, so these measurements are design diagnostics,
not the final independent transfer result. The whole source/fit/training lineage is saved.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pymatching
import stim
from numpy.typing import NDArray

from qecgen.sampling import unpack_bits
from research.realism.decoder import DecoderConfig, fit_decoder, predict
from research.realism.fit_noise import fit_profiles
from research.realism.import_real import (
    blocked_split,
    detector_anchors,
    detector_sequence,
    load_willow_derived,
)
from research.realism.model import build_noisy_circuit, sample_profile


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _array_sha(values: NDArray[Any]) -> str:
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode())
    digest.update(str(values.dtype).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def compare_syndromes(
    synthetic: NDArray[np.bool_], validation: NDArray[np.bool_], ideal: stim.Circuit
) -> dict[str, Any]:
    left = synthetic.astype(np.float64)
    right = validation.astype(np.float64)
    mean_left, mean_right = left.mean(axis=0), right.mean(axis=0)
    covariance_left = left.T @ left / len(left) - np.outer(mean_left, mean_left)
    covariance_right = right.T @ right / len(right) - np.outer(mean_right, mean_right)
    coordinates = detector_anchors(ideal)
    errors: dict[str, list[float]] = {"spatial": [], "temporal": [], "spacetime": []}
    for i in range(ideal.num_detectors):
        for j in range(i):
            a, b = coordinates[i], coordinates[j]
            group = "spatial" if a[2] == b[2] else "temporal" if a[:2] == b[:2] else "spacetime"
            errors[group].append(float(covariance_left[i, j] - covariance_right[i, j]))
    return {
        "detector_marginal_rmse": float(np.sqrt(np.mean((mean_left - mean_right) ** 2))),
        "detector_means_synthetic": mean_left.tolist(),
        "detector_means_validation": mean_right.tolist(),
        "pair_covariance_rmse": {
            name: float(np.sqrt(np.mean(np.square(values)))) if values else None
            for name, values in errors.items()
        },
        "pair_counts": {name: len(values) for name, values in errors.items()},
        "syndrome_weight_quantiles": {
            "quantiles": [0.5, 0.9, 0.99],
            "synthetic": np.quantile(left.sum(axis=1), [0.5, 0.9, 0.99]).tolist(),
            "validation": np.quantile(right.sum(axis=1), [0.5, 0.9, 0.99]).tolist(),
        },
        "caveat": "descriptive development statistics; no causal or chronology claim",
    }


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["version"] != 1 or config["phase"] != "design_validation_only":
        raise ValueError("only version1 design-validation runs are implemented before Gate2")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    source_snapshot = output / "source"
    source_snapshot.mkdir()
    source_hashes = {}
    for name in ("pilot.py", "import_real.py", "fit_noise.py", "model.py", "decoder.py"):
        payload = Path(__file__).with_name(name).read_bytes()
        (source_snapshot / name).write_bytes(payload)
        source_hashes[name] = hashlib.sha256(payload).hexdigest()
    write_json(output / "source_hashes.json", source_hashes)
    write_json(output / "status.json", {"status": "running", "phase": "intake"})
    started = time.monotonic()
    cohort = load_willow_derived(Path(config["table"]), Path(config["circuit"]), config["expected"])
    split = blocked_split(len(cohort.detectors), guard=config["guard_shots"])
    split_record = {
        "kind": "guarded source-row blocks; chronology unverified",
        "guard": config["guard_shots"],
        "final_test_scored": False,
        "cohort_source": cohort.source,
        "partitions": {
            name: {
                "rows": len(rows),
                "first": int(rows[0]),
                "last": int(rows[-1]),
                "sha256": _array_sha(rows),
            }
            for name, rows in (
                ("train", split.train),
                ("validation", split.validation),
                ("test", split.test),
            )
        },
    }
    write_json(output / "split.json", split_record)
    # Only these partitions are unpacked. Test targets never enter fit, metrics or decoder calls.
    train_bits = unpack_bits(cohort.detectors[split.train], cohort.ideal.num_detectors)
    validation_bits = unpack_bits(cohort.detectors[split.validation], cohort.ideal.num_detectors)
    train_y = unpack_bits(cohort.observables[split.train], 1).ravel()
    validation_y = unpack_bits(cohort.observables[split.validation], 1).ravel()
    train_x, layout = detector_sequence(train_bits, cohort.ideal)
    validation_x, validation_layout = detector_sequence(validation_bits, cohort.ideal)
    if layout != validation_layout:
        raise RuntimeError("training and validation detector identities differ")
    write_json(output / "layout.json", layout)
    print(
        f"intake: train={len(train_y)}, validation={len(validation_y)}, "
        f"reserved test={len(split.test)}",
        flush=True,
    )
    write_json(output / "status.json", {"status": "running", "phase": "train-only noise fitting"})
    uniform, heterogeneous, fit = fit_profiles(
        cohort.ideal, train_bits, regularization=config["readout_regularization"]
    )
    write_json(output / "noise_fit.json", fit)
    print(f"noise fitted: p={fit['uniform_p']:.6g}", flush=True)
    arms = {"uniform_matched": uniform, "heterogeneous_readout": heterogeneous}
    result: dict[str, Any] = {
        "phase": "design_validation_only",
        "final_test_scored": False,
        "source": cohort.source,
        "split": split_record,
        "fit": fit,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("stim", "numpy", "torch", "pymatching", "scipy", "pyarrow")
        },
        "constant_zero_validation_failures": int(validation_y.sum()),
        "validation_shots": len(validation_y),
        "profiles": {name: profile.to_dict() for name, profile in arms.items()},
        "statistics": {},
        "matching": {},
        "training": [],
        "limitations": [
            "Third-party hardware-derived source; original Zenodo archive unavailable.",
            "Guarded row splits do not establish chronological or independent-run transfer.",
            "Validation selects checkpoints; final test remains reserved for the approved study.",
            "Rich comparison fits readout heterogeneity; physical processes are separate pilots.",
            "Initial sweep preparation uses Pauli-frame equivalence, not state-dependent physics.",
        ],
    }
    for name, profile in arms.items():
        circuit = build_noisy_circuit(cohort.ideal, profile)
        matcher = pymatching.Matching.from_detector_error_model(
            circuit.detector_error_model(decompose_errors=True)
        )
        predicted = (
            matcher.decode_batch(cohort.detectors[split.validation], bit_packed_shots=True)
            .ravel()
            .astype(bool)
        )
        failures = predicted != validation_y
        np.save(output / f"{name}-matching-validation-failures.npy", failures)
        result["matching"][name] = {
            "failures": int(failures.sum()),
            "rate": float(failures.mean()),
            "prior_source": "real training marginals only",
        }
    deadline = time.monotonic() + min(float(config["training_budget_seconds"]), 3600)
    for seed in config["training_seeds"]:
        train_sets = {"real": (train_x, train_y)}
        for name, profile in arms.items():
            sample_start = time.monotonic()
            sample = sample_profile(
                cohort.ideal,
                profile,
                shots=len(train_y),
                seed=config["seed"] + seed,
                chunk_size=config["chunk_size"],
            )
            duration = time.monotonic() - sample_start
            bits = unpack_bits(sample.detectors, cohort.ideal.num_detectors)
            xs, synthetic_layout = detector_sequence(bits, cohort.ideal)
            if synthetic_layout != layout:
                raise RuntimeError("synthetic detector identity differs from hardware")
            ys = unpack_bits(sample.observables, 1).ravel()
            train_sets[name] = (xs, ys)
            write_json(
                output / f"{name}-seed{seed}-generation.json",
                {
                    "audit": sample.audit,
                    "elapsed_seconds": duration,
                    "shots_per_second": len(ys) / duration,
                    "detector_sha256": _array_sha(sample.detectors),
                    "observable_sha256": _array_sha(sample.observables),
                },
            )
            if seed == config["training_seeds"][0]:
                result["statistics"][name] = compare_syndromes(bits, validation_bits, cohort.ideal)
        for name, (xs, ys) in train_sets.items():
            print(f"training {name}, seed={seed}", flush=True)
            write_json(
                output / "status.json",
                {"status": "running", "phase": "training", "arm": name, "seed": seed},
            )
            decoder_config = DecoderConfig(**config["decoder"], deadline_monotonic=deadline)
            trained = fit_decoder(
                xs,
                ys,
                validation_x,
                validation_y,
                decoder_config,
                seed,
                output / f"decoder-{name}-seed{seed}",
            )
            predictions = predict(trained.model, validation_x, deadline_monotonic=deadline)
            failures = predictions != validation_y
            np.save(output / f"{name}-seed{seed}-validation-failures.npy", failures)
            result["training"].append(
                {
                    "arm": name,
                    "seed": seed,
                    "failures": int(failures.sum()),
                    "rate": float(failures.mean()),
                    "summary": trained.summary,
                    "config": asdict(decoder_config),
                    "training_inputs_sha256": _array_sha(xs),
                    "training_targets_sha256": _array_sha(ys),
                }
            )
            print(
                f"validation {name}, seed={seed}: {int(failures.sum())}/{len(failures)}", flush=True
            )
            write_json(output / "results.json", result)
    result["elapsed_seconds"] = time.monotonic() - started
    result["training_budget_seconds"] = config["training_budget_seconds"]
    write_json(output / "results.json", result)
    write_json(output / "status.json", {"status": "complete", "final_test_scored": False})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("research/realism/pilot_config.json"))
    args = parser.parse_args()
    output = Path(json.loads(args.config.read_text(encoding="utf-8"))["output"])
    if output.exists():
        raise ValueError(f"refusing to overwrite an existing research run: {output}")
    try:
        run(args.config)
    except Exception as error:
        if output.exists():
            write_json(
                output / "failure.json", {"type": type(error).__name__, "message": str(error)}
            )
            write_json(
                output / "status.json",
                {
                    "status": "failed",
                    "final_test_scored": False,
                    "partial_results_available": (output / "results.json").exists(),
                },
            )
        raise


if __name__ == "__main__":
    main()
