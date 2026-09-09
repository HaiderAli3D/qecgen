"""Locked, bounded decoder-transfer experiment with a separately sealed test phase.

Source row order is not verified acquisition chronology. Intervals below describe
sampling and block sensitivity; they cannot establish independent hardware runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pymatching
import stim
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.stats import beta

from qecgen.circuits import Basis, NoiseModel, build_circuit
from qecgen.sampling import iter_chunks, unpack_bits
from research.realism.decoder import DecoderConfig, fit_decoder, load_decoder, predict
from research.realism.fit_noise import detector_marginals, fit_profiles
from research.realism.import_real import blocked_split, detector_sequence, load_willow_derived
from research.realism.legacy_mapping import LegacyMapping, audit_legacy_mapping
from research.realism.model import NoiseProfile, build_noisy_circuit, sample_profile
from research.realism.pilot import _array_sha, compare_syndromes, write_json


def validate_config(config: dict[str, Any]) -> None:
    """A missing arm or empty seed list must not publish a vacuous complete study."""
    if config["version"] != 1 or config["phase"] != "locked_final_evaluation":
        raise ValueError("requires version1 locked_final_evaluation")
    arms = [
        "real",
        "uniform_matched",
        "heterogeneous_readout",
        "heterogeneous_finetuned",
        "legacy_uniform",
    ]
    if config["arms"] != arms:
        raise ValueError("all five registered arms must appear in their dependency order")
    if not config["cohorts"] or not config["training_seeds"]:
        raise ValueError("at least one cohort and training seed are required")
    if len({item["name"] for item in config["cohorts"]}) != len(config["cohorts"]):
        raise ValueError("cohort names must be unique")
    if len(set(config["training_seeds"])) != len(config["training_seeds"]):
        raise ValueError("training seeds must be unique")
    for name in ("training_budget_seconds", "base_fit_seconds", "fine_tune_seconds"):
        if isinstance(config[name], bool) or not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError("training budgets must be finite and positive")
    nominal = (
        len(config["cohorts"])
        * len(config["training_seeds"])
        * (4 * config["base_fit_seconds"] + config["fine_tune_seconds"])
    )
    if nominal > config["training_budget_seconds"] or config["training_budget_seconds"] > 3000:
        raise ValueError("nominal training caps exceed the registered or approved budget")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_legacy(
    ideal: stim.Circuit, training: NDArray[np.bool_], basis: str, rounds: int
) -> tuple[stim.Circuit, LegacyMapping, dict[str, Any]]:
    mapping = audit_legacy_mapping(ideal, Basis(basis.lower()), rounds)
    target = training.mean(axis=0)

    def circuit_at(log_probability: float) -> stim.Circuit:
        return build_circuit(
            3,
            float(np.exp(log_probability)),
            NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL,
            rounds,
            Basis(basis.lower()),
        )[0]

    def objective(log_probability: float) -> float:
        predicted = detector_marginals(circuit_at(log_probability))[
            list(mapping.detector_permutation)
        ]
        return float(np.mean((predicted - target) ** 2))

    fitted = minimize_scalar(
        objective,
        bounds=(np.log(1e-5), np.log(0.05)),
        method="bounded",
        options={"xatol": 0.005, "maxiter": 30},
    )
    if not fitted.success:
        raise RuntimeError("train-only legacy p fit did not converge")
    circuit = circuit_at(float(fitted.x))
    return (
        circuit,
        mapping,
        {
            "p": float(np.exp(fitted.x)),
            "train_marginal_rmse": float(np.sqrt(fitted.fun)),
            "fit_partition": "train",
            "mapping": mapping.evidence,
            "circuit": str(circuit),
            "probability_bounds": [1e-5, 0.05],
            "detector_permutation": list(mapping.detector_permutation),
        },
    )


def legacy_sample(
    circuit: stim.Circuit, mapping: LegacyMapping, shots: int, seed: int, chunk_size: int
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    chunks = list(iter_chunks(circuit, shots, seed, chunk_size=chunk_size))
    bits = unpack_bits(np.concatenate([chunk.detectors for chunk in chunks]), circuit.num_detectors)
    ys = unpack_bits(np.concatenate([chunk.observables for chunk in chunks]), 1).ravel()
    return mapping.detectors(bits), ys


def interval(failures: int, shots: int) -> list[float]:
    if shots < 1 or not 0 <= failures <= shots:
        raise ValueError("invalid binomial counts")
    return [
        0.0 if failures == 0 else float(beta.ppf(0.025, failures, shots - failures + 1)),
        1.0 if failures == shots else float(beta.ppf(0.975, failures + 1, shots - failures)),
    ]


def block_interval(
    values: NDArray[Any], block: int, *, seed: int, repetitions: int = 2000
) -> list[float]:
    """Resample contiguous source-row blocks, including the shorter last block.

    Equal selection probability for each block plus a ratio of total sums/lengths
    handles the remainder without discarding shots. This is sensitivity analysis,
    not a claim that adjacent blocks are physically independent.
    """
    if values.ndim != 1 or not len(values) or block < 1 or repetitions < 2:
        raise ValueError("nonempty vector, positive block and >=2 repetitions required")
    starts = np.arange(0, len(values), block)
    sums = np.add.reduceat(values.astype(float), starts)
    counts = np.minimum(block, len(values) - starts)
    rng = np.random.default_rng(seed)
    choices = rng.integers(len(starts), size=(repetitions, len(starts)))
    ratios = sums[choices].sum(axis=1) / counts[choices].sum(axis=1)
    return np.quantile(ratios, [0.025, 0.975]).tolist()  # type: ignore[no-any-return]


def summarize(failures: NDArray[np.bool_], seed: int) -> dict[str, Any]:
    return {
        "shots": len(failures),
        "failures": int(failures.sum()),
        "rate": float(failures.mean()),
        "iid_descriptive_cp95": interval(int(failures.sum()), len(failures)),
        "block_sensitivity95": {
            str(block): block_interval(failures, block, seed=seed) for block in (128, 512)
        },
    }


def paired(left: NDArray[np.bool_], right: NDArray[np.bool_], seed: int) -> dict[str, Any]:
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError("paired failures require matching nonempty vectors")
    difference = left.astype(int) - right.astype(int)
    return {
        "left_only_failed": int(np.count_nonzero(left & ~right)),
        "right_only_failed": int(np.count_nonzero(right & ~left)),
        "both_failed": int(np.count_nonzero(left & right)),
        "neither_failed": int(np.count_nonzero(~left & ~right)),
        "left_minus_right_rate": float(difference.mean()),
        "block_sensitivity95": {
            str(block): block_interval(difference, block, seed=seed) for block in (128, 512)
        },
    }


def _split_record(split: Any) -> dict[str, Any]:
    return {
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
    }


def seal(output: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Bind every chosen weight file before any test features or labels are used."""
    checkpoints = {
        str(Path(record["checkpoint"]).relative_to(output)): digest(Path(record["checkpoint"]))
        for cohort in records
        for record in cohort["training"]
    }
    payload = {
        "config_sha256": digest(output / "config.json"),
        "checkpoints": checkpoints,
        "development_records_sha256": digest(output / "development.json"),
        "source_hashes_sha256": digest(output / "source_hashes.json"),
        "final_test_scored": False,
        "selection_complete": True,
    }
    write_json(output / "test_lock.json", payload)
    return payload


def verify_seal(output: Path, payload: dict[str, Any]) -> None:
    for relative, expected in payload["checkpoints"].items():
        if digest(output / relative) != expected:
            raise ValueError("locked checkpoint changed before test evaluation")
    for name, key in (
        ("config.json", "config_sha256"),
        ("development.json", "development_records_sha256"),
        ("source_hashes.json", "source_hashes_sha256"),
    ):
        if digest(output / name) != payload[key]:
            raise ValueError(f"locked {name} changed before test evaluation")
    hashes = json.loads((output / "source_hashes.json").read_text())
    for name, expected in hashes.items():
        snapshot = name.replace("qecgen/", "qecgen_")
        if digest(output / "source" / snapshot) != expected:
            raise ValueError("locked source snapshot changed before test evaluation")


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    validate_config(config)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    source = output / "source"
    source.mkdir()
    hashes = {}
    for name in (
        "evaluate.py",
        "decoder.py",
        "model.py",
        "fit_noise.py",
        "import_real.py",
        "pilot.py",
        "legacy_mapping.py",
    ):
        payload = Path(__file__).with_name(name).read_bytes()
        (source / name).write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    for name in ("circuits.py", "sampling.py", "noise.py"):
        payload = Path("qecgen", name).read_bytes()
        (source / ("qecgen_" + name)).write_bytes(payload)
        hashes["qecgen/" + name] = hashlib.sha256(payload).hexdigest()
    write_json(output / "source_hashes.json", hashes)
    records: list[dict[str, Any]] = []
    training_elapsed = 0.0
    started = time.monotonic()
    for cohort_config in config["cohorts"]:
        name = cohort_config["name"]
        destination = output / name
        destination.mkdir()
        cohort = load_willow_derived(
            Path(cohort_config["table"]), Path(cohort_config["circuit"]), cohort_config["expected"]
        )
        split = blocked_split(len(cohort.detectors), config["guard_shots"])
        train_bits = unpack_bits(cohort.detectors[split.train], cohort.ideal.num_detectors)
        valid_bits = unpack_bits(cohort.detectors[split.validation], cohort.ideal.num_detectors)
        train_x, layout = detector_sequence(train_bits, cohort.ideal)
        valid_x, _ = detector_sequence(valid_bits, cohort.ideal)
        train_y = unpack_bits(cohort.observables[split.train], 1).ravel()
        valid_y = unpack_bits(cohort.observables[split.validation], 1).ravel()
        uniform, heterogeneous, fitted = fit_profiles(
            cohort.ideal, train_bits, regularization=config["readout_regularization"]
        )
        profiles = {"uniform_matched": uniform, "heterogeneous_readout": heterogeneous}
        legacy_circuit, mapping, legacy_fit = fit_legacy(
            cohort.ideal,
            train_bits,
            cohort_config["expected"]["basis"],
            cohort_config["expected"]["rounds"],
        )
        record: dict[str, Any] = {
            "name": name,
            "source": cohort.source,
            "split": _split_record(split),
            "layout": layout,
            "noise_fit": fitted,
            "legacy_fit": legacy_fit,
            "profiles": {k: v.to_dict() for k, v in profiles.items()},
            "statistics_validation": {},
            "training": [],
        }
        records.append(record)
        for seed in config["training_seeds"]:
            sets = {"real": (train_x, train_y)}
            legacy_bits, legacy_y = legacy_sample(
                legacy_circuit, mapping, len(train_y), config["seed"] + seed, config["chunk_size"]
            )
            legacy_x, _ = detector_sequence(legacy_bits, cohort.ideal)
            sets["legacy_uniform"] = (legacy_x, legacy_y)
            if seed == config["training_seeds"][0]:
                record["statistics_validation"]["legacy_uniform"] = compare_syndromes(
                    legacy_bits, valid_bits, cohort.ideal
                )
            for arm, profile in profiles.items():
                sample = sample_profile(
                    cohort.ideal,
                    profile,
                    shots=len(train_y),
                    seed=config["seed"] + seed,
                    chunk_size=config["chunk_size"],
                )
                bits = unpack_bits(sample.detectors, cohort.ideal.num_detectors)
                xs, _ = detector_sequence(bits, cohort.ideal)
                sets[arm] = (xs, unpack_bits(sample.observables, 1).ravel())
                if seed == config["training_seeds"][0]:
                    record["statistics_validation"][arm] = compare_syndromes(
                        bits, valid_bits, cohort.ideal
                    )
            pretrain = None
            for arm in config["arms"]:
                fine_tune = arm == "heterogeneous_finetuned"
                xs, ys = (train_x, train_y) if fine_tune else sets[arm]
                options = dict(config["decoder"])
                options["max_seconds"] = (
                    config["fine_tune_seconds"] if fine_tune else config["base_fit_seconds"]
                )
                if fine_tune:
                    if pretrain is None:
                        raise ValueError("fine-tune must follow heterogeneous pretraining")
                    options["init_checkpoint"] = str(pretrain)
                options["deadline_monotonic"] = (
                    time.monotonic() + config["training_budget_seconds"] - training_elapsed
                )
                print(f"train {name} {arm} seed={seed}", flush=True)
                write_json(
                    output / "status.json",
                    {"status": "training", "cohort": name, "arm": arm, "seed": seed},
                )
                fit = fit_decoder(
                    xs,
                    ys,
                    valid_x,
                    valid_y,
                    DecoderConfig(**options),
                    seed,
                    destination / f"{arm}-{seed}",
                )
                training_elapsed += fit.summary["elapsed_seconds"]
                if arm == "heterogeneous_readout":
                    pretrain = fit.checkpoint_path
                entry = {
                    "arm": arm,
                    "seed": seed,
                    "checkpoint": str(fit.checkpoint_path),
                    "summary": fit.summary,
                    "train_x_sha256": _array_sha(xs),
                    "train_y_sha256": _array_sha(ys),
                }
                record["training"].append(entry)
                print(
                    f"validation {fit.summary['validation_errors']}/{len(valid_y)}, "
                    f"epoch={fit.summary['best_epoch']}",
                    flush=True,
                )
                write_json(output / "development.json", records)
    lock = seal(output, records)
    verify_seal(output, lock)
    # Exclusive marker prevents a second scoring call against the same frozen experiment.
    with (output / "test_started.json").open("x") as stream:
        json.dump(
            {"lock_sha256": digest(output / "test_lock.json"), "selection_complete": True}, stream
        )
    results: dict[str, Any] = {
        "phase": "held_out_final",
        "cohorts": [],
        "training_seconds": training_elapsed,
        "lock_sha256": digest(output / "test_lock.json"),
        "limitations": config["limitations"],
    }
    for cohort_config, record in zip(config["cohorts"], records, strict=True):
        cohort = load_willow_derived(
            Path(cohort_config["table"]), Path(cohort_config["circuit"]), cohort_config["expected"]
        )
        split = blocked_split(len(cohort.detectors), config["guard_shots"])
        bits = unpack_bits(cohort.detectors[split.test], cohort.ideal.num_detectors)
        xs, _ = detector_sequence(bits, cohort.ideal)
        ys = unpack_bits(cohort.observables[split.test], 1).ravel()
        destination = output / record["name"]
        evaluation: dict[str, Any] = {
            "name": record["name"],
            "test_shots": len(ys),
            "decoders": [],
            "matching": {},
            "paired": [],
            "statistics_test": {},
        }
        vectors = {}
        for entry in record["training"]:
            model = load_decoder(
                entry["checkpoint"], device=config["decoder"]["device"], max_seconds=30
            )
            failures = predict(model, xs) != ys
            arm, seed = entry["arm"], entry["seed"]
            vectors[(arm, seed)] = failures
            np.save(destination / f"test-failures-{arm}-{seed}.npy", failures)
            evaluation["decoders"].append(
                {"arm": arm, "seed": seed, **summarize(failures, config["seed"])}
            )
        for arm, profile in record["profiles"].items():
            circuit = build_noisy_circuit(cohort.ideal, NoiseProfile.from_dict(profile))
            matcher = pymatching.Matching.from_detector_error_model(
                circuit.detector_error_model(decompose_errors=True)
            )
            failures = (
                matcher.decode_batch(cohort.detectors[split.test], bit_packed_shots=True)
                .ravel()
                .astype(bool)
                != ys
            )
            np.save(destination / f"test-failures-matching-{arm}.npy", failures)
            evaluation["matching"][arm] = summarize(failures, config["seed"])
            sample = sample_profile(
                cohort.ideal,
                NoiseProfile.from_dict(profile),
                shots=record["split"]["train"]["rows"],
                seed=config["seed"] + config["training_seeds"][0],
                chunk_size=config["chunk_size"],
            )
            synthetic = unpack_bits(sample.detectors, cohort.ideal.num_detectors)
            evaluation["statistics_test"][arm] = compare_syndromes(synthetic, bits, cohort.ideal)
        legacy_circuit = stim.Circuit(record["legacy_fit"]["circuit"])
        mapping = LegacyMapping(
            tuple(record["legacy_fit"]["detector_permutation"]), record["legacy_fit"]["mapping"]
        )
        legacy_bits, _ = legacy_sample(
            legacy_circuit,
            mapping,
            record["split"]["train"]["rows"],
            config["seed"] + config["training_seeds"][0],
            config["chunk_size"],
        )
        evaluation["statistics_test"]["legacy_uniform"] = compare_syndromes(
            legacy_bits, bits, cohort.ideal
        )
        legacy_test = bits[:, np.argsort(mapping.detector_permutation)]
        matcher = pymatching.Matching.from_detector_error_model(
            legacy_circuit.detector_error_model(decompose_errors=True)
        )
        failures = matcher.decode_batch(legacy_test).ravel().astype(bool) != ys
        np.save(destination / "test-failures-matching-legacy_uniform.npy", failures)
        evaluation["matching"]["legacy_uniform"] = summarize(failures, config["seed"])
        for seed in config["training_seeds"]:
            for left, right in (
                ("heterogeneous_readout", "uniform_matched"),
                ("uniform_matched", "real"),
                ("heterogeneous_readout", "real"),
                ("heterogeneous_finetuned", "real"),
                ("legacy_uniform", "real"),
                ("heterogeneous_readout", "legacy_uniform"),
            ):
                evaluation["paired"].append(
                    {
                        "left": left,
                        "right": right,
                        "seed": seed,
                        **paired(vectors[(left, seed)], vectors[(right, seed)], config["seed"]),
                    }
                )
        results["cohorts"].append(evaluation)
        write_json(output / "results.json", results)
    results["elapsed_seconds"] = time.monotonic() - started
    write_json(output / "results.json", results)
    write_json(output / "status.json", {"status": "complete", "final_test_scored": True})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    output = Path(config["output"])
    if output.exists():
        raise FileExistsError("A recorded experiment cannot be overwritten or rescored")
    try:
        run(args.config)
    except Exception as error:
        if output.exists():
            write_json(
                output / "failure.json", {"type": type(error).__name__, "message": str(error)}
            )
            write_json(output / "status.json", {"status": "failed"})
        raise


if __name__ == "__main__":
    main()
