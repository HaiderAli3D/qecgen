"""Regenerate public transfer figures from aggregate evidence, without raw shot data.

Ingestion requires a complete sealed experiment. Plot-only regeneration needs only
the committed summary; it never fits parameters or reads held-out detector rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/evidence/realism-summary.json"
ARMS = {
    "legacy_uniform": "Old qecgen",
    "uniform_matched": "Uniform matched",
    "heterogeneous_readout": "Heterogeneous readout",
    "real": "Real training",
    "heterogeneous_finetuned": "Synthetic + fine-tuning",
}
COLORS = ["#778899", "#348abd", "#e09f3e", "#2a9d8f", "#8f5daf"]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ingest(directory: Path) -> dict[str, Any]:
    status = read_json(directory / "status.json")
    if status != {"status": "complete", "final_test_scored": True}:
        raise ValueError("Only a complete final experiment can become public evidence")
    result = read_json(directory / "results.json")
    config = read_json(directory / "config.json")
    lock = read_json(directory / "test_lock.json")
    if result["phase"] != "held_out_final" or result["lock_sha256"] != digest(
        directory / "test_lock.json"
    ):
        raise ValueError("Missing or changed final-evaluation lock")
    if not lock["selection_complete"] or len(lock["checkpoints"]) != 60:
        raise ValueError("Expected 60 checkpoints sealed before test scoring")
    for filename, key in (
        ("config.json", "config_sha256"),
        ("development.json", "development_records_sha256"),
        ("source_hashes.json", "source_hashes_sha256"),
    ):
        if digest(directory / filename) != lock[key]:
            raise ValueError(f"Sealed {filename} changed")
    for filename, expected in lock["checkpoints"].items():
        if digest(directory / filename) != expected:
            raise ValueError(f"Sealed checkpoint changed: {filename}")
    records = read_json(directory / "development.json")
    if len(result["cohorts"]) != 4 or len(records) != 4:
        raise ValueError("Expected all four cohorts")
    names = {"X-r010", "X-r013", "Z-r010", "Z-r013"}
    if {c["name"] for c in result["cohorts"]} != names or {r["name"] for r in records} != names:
        raise ValueError("Expected the four documented X/Z, 10/13-round cohorts")
    expected_runs = {(arm, seed) for arm in ARMS for seed in config["training_seeds"]}
    if len(expected_runs) != 15:
        raise ValueError("Expected five arms and three distinct seeds")
    for cohort in result["cohorts"]:
        if cohort["test_shots"] != 9872:
            raise ValueError("This report's documented split requires 9,872 test rows per cohort")
        runs = cohort["decoders"]
        if len(runs) != 15 or {(r["arm"], r["seed"]) for r in runs} != expected_runs:
            raise ValueError("Incomplete or duplicate decoder outcomes")
        for run in runs:
            if run["shots"] != cohort["test_shots"] or not 0 <= run["failures"] <= run["shots"]:
                raise ValueError("Inconsistent failure counts")
            if abs(run["rate"] - run["failures"] / run["shots"]) > 1e-12:
                raise ValueError("Rate disagrees with recorded counts")
        for metrics in cohort["statistics_test"].values():
            # The shared pilot routine retains a historical 'validation' key.
            # Rename it only in this public aggregate; never rewrite source evidence.
            metrics["detector_means_test"] = metrics.pop("detector_means_validation")
            quantiles = metrics["syndrome_weight_quantiles"]
            quantiles["test"] = quantiles.pop("validation")
            metrics["caveat"] = "first synthetic seed versus held-out test; descriptive only"
    training = [
        {
            "name": record["name"],
            "split": record["split"],
            "source": record["source"],
            "noise_fit": record["noise_fit"],
            "legacy_fit_and_mapping": {
                key: value for key, value in record["legacy_fit"].items() if key != "circuit"
            },
            "runs": [
                {"arm": run["arm"], "seed": run["seed"], "summary": run["summary"]}
                for run in record["training"]
            ],
        }
        for record in records
    ]
    evidence_paths = {
        "production_performance": ROOT / "data/realism/results/production-performance/report.json",
        "calibration_forecast": ROOT / "data/realism/results/calibration_drift_summary.json",
        "density_reference": ROOT / "data/realism/results/model/reference.json",
        "production_parity": ROOT / "data/realism/results/production-parity.json",
    }
    supporting_evidence = {name: read_json(path) for name, path in evidence_paths.items()}
    if supporting_evidence["production_performance"]["status"] != "complete":
        raise ValueError("Production performance evidence is incomplete")
    parity = supporting_evidence["production_parity"]
    if parity["status"] != "passed" or len(parity["cases"]) != 8:
        raise ValueError("Expected eight passing production/prototype profile comparisons")
    if not all(case["equal"] for case in parity["cases"]):
        raise ValueError("A production/prototype profile comparison differs")
    for row in supporting_evidence["production_performance"]["measurements"]:
        if row["config"]["sampling"]["chunk_size"] != 10000 or (
            row["config"]["circuit"]["distance"],
            row["config"]["circuit"]["rounds"],
        ) != (3, 10):
            raise ValueError("Performance caption requires d=3, 10 rounds, 10,000-shot chunks")
    return {
        "summary_version": 1,
        "kind": "aggregate evidence; no per-shot measurements",
        "input_sha256": {
            filename: digest(directory / filename)
            for filename in (
                "results.json",
                "development.json",
                "config.json",
                "test_lock.json",
                "source_hashes.json",
                "runtime-environment.json",
                "independent-verification.json",
            )
        },
        "training_seeds": config["training_seeds"],
        "decoder_config": config["decoder"],
        "runtime_environment": read_json(directory / "runtime-environment.json"),
        "supporting_evidence_sha256": {name: digest(path) for name, path in evidence_paths.items()},
        "supporting_evidence": supporting_evidence,
        "training": training,
        "results": result,
    }


def save_figure(figure: Any, name: str) -> None:
    directory = ROOT / "docs/images"
    directory.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(directory / f"realism-{name}.{extension}", dpi=170, bbox_inches="tight")
    plt.close(figure)


def figures(summary: dict[str, Any]) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    cohorts = summary["results"]["cohorts"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True, layout="constrained")
    for axis, cohort in zip(axes.ravel(), cohorts, strict=True):
        for index, arm in enumerate(ARMS):
            rates = [100 * r["rate"] for r in cohort["decoders"] if r["arm"] == arm]
            axis.scatter(np.full(len(rates), index), rates, color=COLORS[index], s=28, zorder=3)
            axis.plot([index - 0.2, index + 0.2], [mean(rates)] * 2, color=COLORS[index], lw=3)
        matching = 100 * cohort["matching"]["uniform_matched"]["rate"]
        axis.axhline(matching, color="#444444", linestyle="--", lw=1, label="Matched MWPM")
        axis.set_title(cohort["name"])
        axis.set_xticks(range(len(ARMS)), [label.replace(" ", "\n") for label in ARMS.values()])
        axis.set_ylabel("Held-out logical failure (%)")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8, loc="upper right")
    figure.suptitle(
        "Decoder transfer: three training seeds, same held-out shots\n"
        "Dots: seeds; bars: means; no confidence intervals on seed means"
    )
    save_figure(figure, "transfer")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for index, arm in enumerate(list(ARMS)[:3]):
        values = [c["statistics_test"][arm] for c in cohorts]
        axes[0].plot(
            range(4),
            [v["detector_marginal_rmse"] for v in values],
            "o-",
            color=COLORS[index],
            label=ARMS[arm],
        )
        axes[1].plot(
            range(4),
            [v["pair_covariance_rmse"]["temporal"] for v in values],
            "o-",
            color=COLORS[index],
            label=ARMS[arm],
        )
    for axis in axes:
        axis.set_xticks(range(4), [c["name"] for c in cohorts], rotation=15)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
        axis.set_ylabel("RMSE (dimensionless; smaller is closer)")
    axes[0].set_title("Detector firing rates")
    axes[1].set_title("Same-check temporal covariance")
    figure.suptitle("First synthetic seed versus held-out hardware; no uncertainty estimate")
    save_figure(figure, "distribution")

    performance = summary["supporting_evidence"]["production_performance"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for dynamic, label, color in (
        (False, "Static device", COLORS[1]),
        (True, "Dynamic", COLORS[2]),
    ):
        selected = [
            row
            for row in performance["measurements"]
            if row["config"]["mode"] == "device"
            and row["config"]["output"]["format"] == "hdf5"
            and bool(row["config"]["noise"]["drift"].get("enabled", False)) == dynamic
        ]
        sizes = [row["config"]["sampling"]["shots"] for row in selected]
        axes[0].plot(
            sizes, [row["shots_per_second"] for row in selected], "o-", color=color, label=label
        )
        axes[1].plot(
            sizes,
            [row["peak_working_set_bytes"] / 2**20 for row in selected],
            "o-",
            color=color,
            label=label,
        )
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Total shots; chunk size fixed at 10,000")
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Generation and HDF5 export throughput (shots/s)")
    axes[1].set_ylabel("Fresh-process peak working set (MiB)")
    peak_mib = max(row["peak_working_set_bytes"] for row in performance["measurements"]) / 2**20
    axes[1].set_ylim(0, peak_mib * 1.15)
    figure.suptitle(
        "Production streaming: illustrative d=3, 10-round scenarios; one trial per point"
    )
    save_figure(figure, "performance")


def result_text(summary: dict[str, Any]) -> str:
    result = summary["results"]
    cohorts = result["cohorts"]
    stops = Counter(
        run["summary"]["stop_reason"] for record in summary["training"] for run in record["runs"]
    )
    lines = [
        "The sealed final experiment completed 60 training/fine-tuning runs across four",
        "cohorts. The table shows each arm's mean over three seeds and its individual",
        "failure counts (seeds 17, 29, 43), all on the same 9,872 test shots within that cohort.",
        "",
        "| Cohort | Training arm | Failures by seed | Mean failure |",
        "|---|---|---|---|",
    ]
    for cohort in cohorts:
        for arm, label in ARMS.items():
            runs = sorted(
                (r for r in cohort["decoders"] if r["arm"] == arm), key=lambda r: r["seed"]
            )
            counts = ", ".join(str(r["failures"]) for r in runs)
            rate = 100 * mean(r["rate"] for r in runs)
            lines.append(f"| {cohort['name']} | {label} | {counts} | {rate:.3f}% |")
    lines += [
        "",
        "![Decoder seed outcomes](images/realism-transfer.png)",
        "",
        "Equal-weight averages across cohorts and seeds are descriptive, not pooled",
        "independent observations:",
        "",
        "| Training arm | Mean across four cohorts and three seeds |",
        "|---|---|",
    ]
    averages = {}
    for arm, label in ARMS.items():
        rate = mean(r["rate"] for c in cohorts for r in c["decoders"] if r["arm"] == arm)
        averages[arm] = rate
        lines.append(f"| {label} | {100 * rate:.3f}% |")
    delta = 100 * (averages["heterogeneous_readout"] - averages["uniform_matched"])
    direction = "worse" if delta > 0 else "better" if delta < 0 else "unchanged"
    differences = [
        pair["left_minus_right_rate"]
        for cohort in cohorts
        for pair in cohort["paired"]
        if pair["left"] == "heterogeneous_readout" and pair["right"] == "uniform_matched"
    ]
    better = sum(value < 0 for value in differences)
    worse = sum(value > 0 for value in differences)
    fine_delta = 100 * (averages["heterogeneous_finetuned"] - averages["real"])
    fine_direction = "worse" if fine_delta > 0 else "better" if fine_delta < 0 else "unchanged"
    real_below_matching = sum(
        run["rate"] > cohort["matching"]["uniform_matched"]["rate"]
        for cohort in cohorts
        for run in cohort["decoders"]
        if run["arm"] == "real"
    )
    cohort_comparisons = []
    for baseline in ("legacy_uniform", "uniform_matched"):
        gains, losses = [], []
        for cohort in cohorts:
            cohort_means = {
                arm: mean(r["rate"] for r in cohort["decoders"] if r["arm"] == arm)
                for arm in (baseline, "heterogeneous_readout")
            }
            if cohort_means["heterogeneous_readout"] < cohort_means[baseline]:
                gains.append(cohort["name"])
            elif cohort_means["heterogeneous_readout"] > cohort_means[baseline]:
                losses.append(cohort["name"])
        cohort_comparisons.append(
            f"By cohort mean, heterogeneous readout improves over {ARMS[baseline]} in "
            f"{', '.join(gains) or 'none'} and worsens in {', '.join(losses) or 'none'}."
        )
    lines += [
        "",
        f"The heterogeneous-readout arm is **{abs(delta):.4f} percentage points {direction}**",
        "than uniform noise on the same experimental circuit on this descriptive average.",
        "This does not identify a physical cause or establish performance on other devices.",
        *cohort_comparisons,
        f"Across the 12 cohort/seed pairs, {better} improve and {worse} worsen;",
        f"{12 - better - worse} tie. Fine-tuning is {abs(fine_delta):.4f} percentage points",
        f"{fine_direction} than real-only training on the same descriptive average.",
        f"The real-trained GRU is worse than matched uniform MWPM in {real_below_matching}/12",
        "runs. A validation plateau therefore does not establish decoder competence.",
        "",
        "| Cohort | Matched uniform MWPM | Matched heterogeneous MWPM | Old-qecgen MWPM |",
        "|---|---|---|---|",
    ]
    for cohort in cohorts:
        cells = [
            f"{100 * cohort['matching'][arm]['rate']:.3f}%"
            for arm in ("uniform_matched", "heterogeneous_readout", "legacy_uniform")
        ]
        lines.append(f"| {cohort['name']} | {' | '.join(cells)} |")
    lines += [
        "",
        "The paired differences below compare the same shots. Negative values favor",
        "the first arm. Intervals resample source-row blocks of 512; the committed",
        "summary also retains 128-row intervals and per-arm descriptive binomial",
        "intervals. Block resampling is a sensitivity analysis, not proof that blocks",
        "are independent hardware acquisitions.",
        "",
        "| Cohort | Seed | Comparison | Difference (percentage points) | 512-row interval |",
        "|---|---|---|---|---|",
    ]
    for cohort in cohorts:
        for pair in cohort["paired"]:
            if (pair["left"], pair["right"]) not in {
                ("heterogeneous_readout", "uniform_matched"),
                ("heterogeneous_finetuned", "real"),
            }:
                continue
            low, high = (100 * v for v in pair["block_sensitivity95"]["512"])
            label = (
                "Heterogeneous - uniform"
                if pair["right"] == "uniform_matched"
                else "Fine-tuned - real"
            )
            lines.append(
                f"| {cohort['name']} | {pair['seed']} | {label} | "
                f"{100 * pair['left_minus_right_rate']:+.3f} | [{low:+.3f}, {high:+.3f}] |"
            )
    lines += [
        "",
        "![Distribution diagnostics](images/realism-distribution.png)",
        "",
        "Distribution metrics use only the first synthetic seed. The committed summary",
        "also retains spatial and spacetime covariance errors and syndrome-weight tails.",
        "These comparisons have no estimated sampling uncertainty.",
        "",
        f"Recorded training/fine-tuning time: **{result['training_seconds']:.1f} seconds**;",
        f"complete experiment wall time: **{result['elapsed_seconds']:.1f} seconds**.",
        "Recorded stopping reasons: "
        + ", ".join(f"{name}: {count}" for name, count in sorted(stops.items()))
        + ".",
        "The checkpoint lock and source-artifact SHA-256 values are retained in",
        "[the committed aggregate summary](evidence/realism-summary.json).",
        "Full checkpoints and per-shot failure vectors remain in ignored local results.",
        "",
        "For all eight fitted static profiles (two models by four cohorts), a separate",
        "production/prototype check found exact noisy-circuit text, detector arrays,",
        "observable arrays and content hashes at 10,000 shots, seed 20260927 and",
        "10,000-shot chunks. No held-out hardware outcomes entered that check. This",
        "links these evaluated static profiles to production; it is not a validation",
        "of every dynamic process or a guarantee across runtime versions.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, help="Ingest a completed sealed experiment directory"
    )
    args = parser.parse_args()
    if args.results is not None:
        summary = ingest(args.results)
        SUMMARY.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    else:
        summary = read_json(SUMMARY)
    if summary["summary_version"] != 1:
        raise ValueError("Unsupported public summary version")
    figures(summary)
    path = ROOT / "docs/REALISM.md"
    text = path.read_text(encoding="utf-8")
    start, end = "<!-- REALISM_RESULTS_START -->", "<!-- REALISM_RESULTS_END -->"
    before, remainder = text.split(start)
    _, after = remainder.split(end)
    text = before + start + "\n" + result_text(summary) + "\n" + end + after
    start, end = "<!-- REALISM_PERFORMANCE_START -->", "<!-- REALISM_PERFORMANCE_END -->"
    before, remainder = text.split(start)
    _, after = remainder.split(end)
    performance_lines = [
        "| Model | Format | Shots | Shots/s | Peak working set (MiB) |",
        "|---|---|---|---|---|",
    ]
    for row in summary["supporting_evidence"]["production_performance"]["measurements"]:
        config = row["config"]
        kind = (
            "legacy"
            if config["mode"] == "legacy"
            else ("dynamic" if config["noise"]["drift"].get("enabled", False) else "static device")
        )
        performance_lines.append(
            f"| {kind} | {config['output']['format']} | {config['sampling']['shots']:,} | "
            f"{row['shots_per_second']:,.0f} | {row['peak_working_set_bytes'] / 2**20:.2f} |"
        )
    performance_lines += [
        "",
        "![Production throughput and memory](images/realism-performance.png)",
        "",
        "Measurements include generation and export, with 10,000-shot chunks, d=3",
        "and 10 rounds. Peak working set includes interpreter/imports. The static",
        "and dynamic 100,000-shot HDF5/NPZ pairs reproduced identical arrays and",
        "content hashes. Single trials under concurrent machine activity do not",
        "establish a latency guarantee or predict larger-distance cost. Static and",
        "legacy profiles also contain different operation probabilities, so their",
        "throughput ratio is not a controlled implementation-only comparison.",
    ]
    text = before + start + "\n" + "\n".join(performance_lines) + "\n" + end + after
    path.write_text(text, encoding="utf-8")
    print("Regenerated docs/REALISM.md and PNG/PDF figures from aggregate evidence")


if __name__ == "__main__":
    main()
