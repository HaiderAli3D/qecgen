"""Compare scientific outputs while ignoring timestamps and elapsed-time metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def compare(first: Path, second: Path) -> dict[str, Any]:
    def read(directory: Path, name: str) -> Any:
        return json.loads((directory / name).read_text(encoding="utf-8"))

    for directory in (first, second):
        if read(directory, "status.json")["status"] != "complete":
            raise ValueError(f"run is not complete: {directory}")
    a, b = read(first, "results.json"), read(second, "results.json")
    checks = {
        "split_exact": a["split"] == b["split"],
        "noise_fit_exact": a["fit"] == b["fit"],
        "profiles_exact": a["profiles"] == b["profiles"],
        "syndrome_statistics_exact": a["statistics"] == b["statistics"],
        "matching_counts_exact": a["matching"] == b["matching"],
        "reserved_test_unscored": not a["final_test_scored"] and not b["final_test_scored"],
    }
    left = {(row["arm"], row["seed"]): row for row in a["training"]}
    right = {(row["arm"], row["seed"]): row for row in b["training"]}
    expected = {
        (arm, seed)
        for arm in ("real", "uniform_matched", "heterogeneous_readout")
        for seed in read(first, "config.json")["training_seeds"]
    }
    checks["training_arms_complete_and_exact"] = (
        bool(expected)
        and set(left) == set(right) == expected
        and len(a["training"]) == len(b["training"]) == len(expected)
    )
    checks["no_timing_stop"] = all(
        row["summary"]["stop_reason"] in {"max_epochs", "validation_patience"}
        for run in (a, b)
        for row in run["training"]
    )
    rows = []
    for key in sorted(set(left) & set(right)):
        x, y = left[key], right[key]
        filename = f"{key[0]}-seed{key[1]}-validation-failures.npy"
        bits_equal = np.array_equal(np.load(first / filename), np.load(second / filename))
        inputs_equal = all(
            x[field] == y[field] for field in ("training_inputs_sha256", "training_targets_sha256")
        )
        best_equal = all(
            x["summary"][field] == y["summary"][field]
            for field in ("best_epoch", "validation_loss", "validation_errors", "epochs_completed")
        )
        rows.append(
            {
                "arm": key[0],
                "seed": key[1],
                "inputs_exact": inputs_equal,
                "best_checkpoint_metrics_exact": best_equal,
                "prediction_outcomes_exact": bits_equal,
            }
        )
    checks["all_training_inputs_exact"] = all(row["inputs_exact"] for row in rows)
    checks["all_checkpoint_metrics_exact"] = all(
        row["best_checkpoint_metrics_exact"] for row in rows
    )
    checks["all_prediction_outcomes_exact"] = all(row["prediction_outcomes_exact"] for row in rows)
    seconds = sum(row["summary"]["elapsed_seconds"] for run in (a, b) for row in run["training"])
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "arms": rows,
        "first_run": str(first),
        "second_run": str(second),
        "recorded_training_wall_seconds_both_runs": seconds,
        "scope": (
            "same machine, pinned libraries, source-reviewed metadata/deadline fix; no timing stop"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.first, args.second)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
