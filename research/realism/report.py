"""Render the model-design checkpoint from completed pilot result files.

The reserved hardware test partition is never read here. Seed ranges are shown
as descriptive training variation, not as confidence intervals over devices or
experiments. A running pilot produces an explicit partial note instead of a
report that mistakes an incrementally written results file for a finished run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

ARMS = ("real", "uniform_matched", "heterogeneous_readout")
LABELS = ("Real training", "Uniform synthetic", "Heterogeneous readout")
COLORS = ("#246A73", "#B2692A", "#6759A6")
SEEDS = (17, 29, 43)


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _save(fig: Figure, directory: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _style(ax: Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)


def arm_summary(pilot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Refuse missing or duplicated runs rather than averaging whatever finished."""
    runs = pilot["training"]
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    observed = {(run["arm"], run["seed"]) for run in runs}
    if observed != expected or len(runs) != len(expected):
        raise ValueError("A completed report requires exactly three seeds for each of three arms")
    result = {}
    for arm in ARMS:
        selected = sorted((run for run in runs if run["arm"] == arm), key=lambda r: r["seed"])
        rates = [run["failures"] / pilot["validation_shots"] for run in selected]
        for run, rate in zip(selected, rates, strict=True):
            if not np.isclose(rate, run["rate"], rtol=0, atol=1e-14):
                raise ValueError("Reported decoder rate disagrees with integer failure counts")
        result[arm] = {
            "runs": selected,
            "rates": rates,
            "mean": mean(rates),
            "minimum": min(rates),
            "maximum": max(rates),
            "failures": [run["failures"] for run in selected],
        }
    return result


def _decoder_plot(pilot: dict[str, Any], summary: dict[str, dict[str, Any]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.5))
    for index, (arm, color) in enumerate(zip(ARMS, COLORS, strict=True)):
        rates = summary[arm]["rates"]
        for offset, rate, marker in zip((-0.14, 0, 0.14), rates, ("o", "^", "s"), strict=True):
            ax.scatter(index + offset, rate, s=64, color=color, marker=marker, zorder=3)
        ax.plot([index - 0.24, index + 0.24], [summary[arm]["mean"]] * 2, color=color, lw=2.2)
        ax.text(index, 0.12, f"Mean {summary[arm]['mean']:.3%}", ha="center", color=color)
    matching = pilot["matching"]["uniform_matched"]["rate"]
    ax.axhline(matching, color="#57616C", ls="--", lw=1.2)
    ax.text(2.48, matching - 0.002, f"Matching: {matching:.3%}", ha="right", va="top", size=10)
    ax.set_xticks(range(3), LABELS)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 0.13)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Logical prediction failures / validation shots")
    ax.set_title("Validation transfer: three training sources, three seeds each", pad=22)
    _style(ax)
    fig.text(
        0.08,
        0.02,
        "One d=3, Z, 10-round cohort; 9,744 shared validation shots. Lower is better.\n"
        "Dots: seeds 17, 29, 43 from left to right. Lines: means, not confidence intervals.\n"
        "Validation selected checkpoints; not final-test or independent-run evidence.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    _save(fig, output, "decoder_validation")


def _detector_plot(pilot: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    stats = pilot["statistics"]
    maximum = 0.0
    for arm, color, label in zip(ARMS[1:], COLORS[1:], LABELS[1:], strict=True):
        item = stats[arm]
        hardware = item["detector_means_validation"]
        synthetic = item["detector_means_synthetic"]
        maximum = max(maximum, max(hardware), max(synthetic))
        axes[0].scatter(hardware, synthetic, s=28, alpha=0.65, color=color, label=label)
    limit = maximum * 1.08
    axes[0].plot([0, limit], [0, limit], color="#777777", ls="--", lw=1)
    axes[0].set(xlim=(0, limit), ylim=(0, limit))
    axes[0].set_xlabel("Hardware validation detection probability")
    axes[0].set_ylabel("Synthetic detection probability")
    axes[0].set_title("All 80 detector marginals in the selected cohort")
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(frameon=False, fontsize=9)
    categories = ("spatial", "temporal", "spacetime")
    positions = np.arange(3)
    for offset, arm, color in zip((-0.18, 0.18), ARMS[1:], COLORS[1:], strict=True):
        values = [stats[arm]["pair_covariance_rmse"][key] for key in categories]
        axes[1].bar(positions + offset, values, width=0.34, color=color)
    axes[1].set_xticks(positions, ("Spatial", "Temporal", "Space + time"))
    axes[1].set_ylabel("RMSE of pairwise covariance (dimensionless)")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
    axes[1].set_title("Remaining pairwise correlation discrepancies")
    for ax in axes:
        _style(ax)
    fig.text(
        0.055,
        0.02,
        "Train-fitted models versus real validation; first synthetic seed only, no intervals.\n"
        "Pair groups: 264 spatial, 364 temporal, 2,532 spacetime. Lower discrepancy is better.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    _save(fig, output, "detector_statistics")


def _reference_plot(reference: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    states = ("00", "11")
    positions = np.arange(2)
    motif = reference["repeated_parity"]
    for offset, key, label, color in (
        (-0.18, "exact_observable_flip_probability", "Exact two-level channel", COLORS[0]),
        (0.18, "pauli_observable_flip_probability", "Pauli approximation", COLORS[1]),
    ):
        values = [motif[state][key] for state in states]
        axes[0].bar(positions + offset, values, width=0.34, label=label, color=color)
    axes[0].set_ylabel("Observable flip probability")
    axes[0].set_title("Pauli noise loses ground-state preference")
    axes[0].legend(frameon=False, fontsize=9)
    variations = [motif[state]["joint_detection_observable_total_variation"] for state in states]
    axes[1].bar(positions, variations, width=0.55, color=COLORS[2])
    for index, value in enumerate(variations):
        axes[1].text(index, value + 0.004, f"{value:.4f}", ha="center")
    axes[1].set_ylabel("Total variation of joint outcome distribution")
    axes[1].set_title("Difference remains after repeated checks")
    for ax in axes:
        ax.set_xticks(positions, ("Prepared 00", "Prepared 11"))
        ax.set_ylim(0, 0.17)
        _style(ax)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    fig.text(
        0.055,
        0.02,
        "Illustrative 3-round parity motif: 1 us idle/round, T1=20 us, T2=30 us; ideal gates.\n"
        "Exact refers to two-level damping/dephasing; not complete transmon or hardware physics.\n"
        "Pauli approximation: 2.4385% spurious excitation in a single-qubit ground state.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    _save(fig, output, "pauli_reference_discrepancy")


def _calibration_plot(drift: dict[str, Any], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    families = ("T1", "T2", "sx_error", "cz_error", "readout_error")
    for offset, backend, color in (
        (-0.15, "ibm_fez", COLORS[0]),
        (0.15, "ibm_kingston", COLORS[1]),
    ):
        groups = {g["family"]: g for g in drift["groups"] if g["backend"] == backend}
        values = [
            groups[family]["held_out_forecast"]["series_mae_ratio_quantiles"]["median"]
            for family in families
        ]
        ax.scatter(
            np.arange(5) + offset,
            values,
            label=backend.removeprefix("ibm_").title(),
            s=64,
            color=color,
            zorder=3,
        )
        for index, value in enumerate(values):
            ax.annotate(
                f"{value:.2f}",
                (index + offset, value),
                (0, 8),
                textcoords="offset points",
                ha="center",
                color=color,
                fontsize=9,
            )
    ax.axhline(1, color="#57616C", ls="--", lw=1)
    ax.text(-0.4, 1.025, ">1: persistence is worse", ha="left", fontsize=10)
    ax.set_xticks(
        range(5), ("T1", "T2", "Single-qubit\ngate error", "CZ gate\nerror", "Readout error")
    )
    ax.set_ylim(0, 1.68)
    ax.set_ylabel("Median per-series MAE ratio\n(last observation / frozen training mean)")
    ax.set_title("Held-out calibration forecasts: last observation vs training mean")
    ax.legend(frameon=False, loc="lower left")
    _style(ax)
    fig.text(
        0.06,
        0.02,
        "Held-out IBM calibration reports; rolling forecasts use only already-observed values.\n"
        "Medians across correlated series; not uncertainty intervals or Willow transfer evidence.\n"
        "Repeated API reports do not establish independently measured physical drift.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    _save(fig, output, "calibration_forecast")


def _report_text(
    pilot: dict[str, Any],
    summary: dict[str, dict[str, Any]],
    model: dict[str, Any],
    reference: dict[str, Any],
    drift: dict[str, Any],
    replay: dict[str, Any],
) -> str:
    rows = []
    for arm, label in zip(ARMS, LABELS, strict=True):
        item = summary[arm]
        rows.append(
            f"| {label} | {', '.join(map(str, item['failures']))} | "
            f"{item['mean']:.4%} | {item['minimum']:.4%}-{item['maximum']:.4%} |"
        )
    difference_pp = (
        summary["heterogeneous_readout"]["mean"] - summary["uniform_matched"]["mean"]
    ) * 100
    stats = pilot["statistics"]
    uniform, heterogeneous = stats["uniform_matched"], stats["heterogeneous_readout"]
    throughput = {item["name"]: item for item in model["results"]}
    training_seconds = sum(run["summary"]["elapsed_seconds"] for run in pilot["training"])
    plateau_count = sum(run["summary"]["validation_plateau_observed"] for run in pilot["training"])
    cov_rows = []
    for key, label in (
        ("spatial", "Spatial"),
        ("temporal", "Temporal"),
        ("spacetime", "Space and time"),
    ):
        old, new = (item["pair_covariance_rmse"][key] for item in (uniform, heterogeneous))
        direction = "Better" if new < old else "Worse" if new > old else "Unchanged"
        cov_rows.append(f"| {label} | {old:.8f} | {new:.8f} | {direction} |")
    max_memory_mib = (
        max(
            throughput[name]["fresh_process_peak_working_set_bytes"]
            for name in ("static_uniform", "static_heterogeneous", "dynamic_illustration")
        )
        / 2**20
    )
    dynamic_rate_k = throughput["dynamic_illustration"]["shots_per_second"] / 1e3
    pairing = [
        b - a
        for a, b in zip(
            summary["uniform_matched"]["failures"],
            summary["heterogeneous_readout"]["failures"],
            strict=True,
        )
    ]
    improved_seeds = sum(change < 0 for change in pairing)
    worse_seeds = sum(change > 0 for change in pairing)
    mean_direction = "worse" if difference_pp > 0 else "better" if difference_pp < 0 else "equal"
    matching_counts = {arm: pilot["matching"][arm]["failures"] for arm in ARMS[1:]}
    if len(set(matching_counts.values())) != 1:
        raise ValueError("Divergent matching baselines require updating the checkpoint discussion")
    matching_count = matching_counts[ARMS[1]]
    quantiles_synthetic = ", ".join(
        f"{value:g}" for value in uniform["syndrome_weight_quantiles"]["synthetic"]
    )
    quantiles_real = ", ".join(
        f"{value:g}" for value in uniform["syndrome_weight_quantiles"]["validation"]
    )
    if uniform["syndrome_weight_quantiles"] != heterogeneous["syndrome_weight_quantiles"]:
        raise ValueError("Different synthetic tails require updating the checkpoint discussion")
    forecasts = [group["held_out_forecast"] for group in drift["groups"]]
    improved_groups = sum(
        row["persistence"]["pooled_mae"] < row["training_mean"]["pooled_mae"] for row in forecasts
    )
    improved_series = sum(row["series_improved_mae"] for row in forecasts)
    worse_series = sum(row["series_worse_mae"] for row in forecasts)
    return f"""# Gate 2: working model-design results

**The isolated pilot is complete. The full generator overhaul is not implemented.**
Adding fitted readout heterogeneity slightly improves detector-rate agreement but
does not improve mean decoder validation performance. The evidence supports
careful infrastructure integration and a stronger follow-up experiment, not a
claim that the new generator already produces hardware-equivalent data.

This is the approved model-design checkpoint. Full integration and final public
documentation remain separate approval gates. Physics sources and approximation
choices are in [EVIDENCE.md](EVIDENCE.md).

## What was built and acquired

- Frozen legacy references cover three existing noise models, both bases, two
  seed/chunk pairs and Contracts A/B. The existing default was already
  circuit-level noise; this work does not replace a code-capacity-only default.
- The isolated simulator supports operation/per-qubit/per-edge noise, timed T1/T2
  approximations, explicit correlations and persistent drift/burst/leakage-effect
  proxies. Transition frequency, temperatures and humidity are recorded covariates
  with zero automatic response. The proxies are assumptions, not identified causes.
- Acquisition recorded **14 assets, 58,895,044 bytes**, including four derived
  Willow cohorts totalling **200,000 shots**, a measured IBM calibration snapshot
  and calibration histories. Raw data, licences, checksums and receipts remain
  under `data/realism`; none are proposed for git. Original Google Zenodo
  downloads were blocked by HTTP 403. Willow shots are third-party-derived
  hardware data; original Google bytes were not independently compared.

Only one cohort was used for this transfer pilot: d=3, Z basis, 10 rounds,
orientation q10_7, 50,000 supplied shots. Guarded row blocks allocate **29,872
training**, **9,744 validation**, and **9,872 reserved test** shots. The 512 omitted
boundary rows reduce immediate adjacency; they do not prove independence. Source
row order is preserved, but acquisition chronology and independent run identities
are unverified. The final test partition was **not scored**.

## Decoder validation: no demonstrated improvement

A small recurrent neural network (GRU) was trained with the same architecture,
training count and maximum training budget for all arms. The synthetic arms use
the experimental circuit layout; both fit from real training data only. The richer
arm adds **effective readout heterogeneity only**. Calibrated leakage, thermal
state dependence, coherent errors and hardware-time drift were not fitted here.
Separate physical-process prototypes therefore cannot explain this result.

| Training data | Failures for seeds 17, 29, 43 / 9,744 | Mean rate | Seed range |
|---|---|---|---|
{chr(10).join(rows)}

The heterogeneous arm changes failure counts by **{pairing[0]}, {pairing[1]},
{pairing[2]}** relative to uniform: **{improved_seeds} improve, {worse_seeds} worsen**.
Its mean is **{abs(difference_pp):.5f} percentage points {mean_direction}**. These are descriptive
seed summaries on the same validation shots, not independent experiments or
confidence intervals. No significance or population-wide improvement is claimed.

Both fitted matching priors give
**{matching_count}/9,744 = {pilot["matching"]["uniform_matched"]["rate"]:.4%}**.
Even the real-trained GRU is worse. The GRU learns beyond the constant-zero
prediction ({pilot["constant_zero_validation_failures"]}/9,744), but competence and convergence
remain unresolved: only **{plateau_count}/9** runs observed the configured validation plateau,
and some hit the 80-epoch ceiling. Validation also selected checkpoints. These are
development results, not a locked final comparison. The earlier intake attempt
failed before training; the two completed runs used identical training settings.

Nine training runs took **{training_seconds:.1f} seconds** in total; the complete pilot
took **{pilot["elapsed_seconds"]:.1f} seconds**. Checkpoints, fitted profiles, input hashes,
seeds and training settings were recorded. The untouched legacy generator was
regression-tested but was **not** scored as an interchangeable input arm: its
generated circuit conventions must first be mapped to the experimental circuit.
Synthetic pretraining followed by real fine-tuning remains untested.

![All decoder seeds and means](../../data/realism/results/figures/decoder_validation.png)

## Distribution, physics and performance checks

Detector-rate RMSE changes from **{uniform["detector_marginal_rmse"]:.8f}** to
**{heterogeneous["detector_marginal_rmse"]:.8f}**, a small reduction. Pairwise covariance
RMSE moves in both directions:

| Pair group | Uniform | Heterogeneous readout | Direction |
|---|---|---|---|
{chr(10).join(cov_rows)}

These probabilities/covariances are dimensionless. Both synthetic arms have
syndrome-weight quantiles **{quantiles_synthetic}** at the 50th/90th/99th percentiles,
versus **{quantiles_real}** for validation. The observed upper tail remains underrepresented.
These statistics use only the first synthetic seed and have no sampling-uncertainty
estimate. Small marginal agreement does not establish accurate correlations or transfer.

![Detector statistics][detector-figure]

[detector-figure]: ../../data/realism/results/figures/detector_statistics.png

The exact small-system reference exposes a known approximation error rather than
hiding it. With illustrative 1 us idles, T1=20 us and T2=30 us, Pauli twirling
introduces **{reference["single_qubit"]["ground"]["pauli_excited_probability"]:.4%}**
excitation into a ground state that exact zero-temperature damping leaves alone.
In the three-round parity motif, the joint-outcome total variation is
**{reference["repeated_parity"]["00"]["joint_detection_observable_total_variation"]:.5f}**
for preparation 00 and
**{reference["repeated_parity"]["11"]["joint_detection_observable_total_variation"]:.5f}**
for 11. Exact means the stated two-level damping/dephasing reference, not complete
transmon physics. These are illustrative parameters, not hardware measurements.

![Pauli approximation discrepancy][reference-figure]

[reference-figure]: ../../data/realism/results/figures/pauli_reference_discrepancy.png

On the local CPU, one 100,000-shot d=3, 10-round trial measured approximately
**{throughput["static_uniform"]["shots_per_second"] / 1e6:.2f} million shots/s** for static
uniform noise,
**{throughput["static_heterogeneous"]["shots_per_second"] / 1e6:.2f} million shots/s**
for static heterogeneity, and **{dynamic_rate_k:.1f} thousand
shots/s** for the combined dynamic illustration. Fresh-process peak working set
was about **{max_memory_mib:.1f} MiB**,
including interpreter/imports. These single-run timings include uncontrolled
background activity and cannot be projected to large distances. The pilot
materializes output; production streaming still needs integration and testing.

Calibration-history analysis retained **{drift["retained_series"]:,}** sufficiently
populated series. Across all four backends, last-observation prediction improves
pooled held-out mean absolute error in **{improved_groups}/{len(forecasts)} groups**.
Across individual series, **{improved_series:,} improve and {worse_series:,} worsen**.
Torino gate-error forecasts supply large gains while Fez/Kingston coherence
forecasts worsen. Results therefore depend on the backend and calibration family.
For Fez/Kingston, a last-observation predictor often loses to a
frozen training mean, including T1 and T2; assuming universal smooth persistence
would be wrong. The figure shows median per-series forecast-error ratios, which
need not agree with a ratio of pooled errors. This is held-out **IBM calibration
forecasting**, separate from Willow decoder validation. API restamping, irregular
sampling and correlated series prevent interpreting persistence as direct evidence
of physical correlation. Weather columns were not used.

![Calibration forecasting ratios](../../data/realism/results/figures/calibration_forecast.png)

## Gate 2 recommendation and remaining limits

Approve a bounded next stage: integrate versioned configuration, provenance,
explicit legacy presets, checked hardware import and the static heterogeneous
backend into the existing staged exporter path. Keep dynamic mechanisms explicitly
experimental. Preserve the matching between circuit, detector order, observable
convention and supplied measurement preparation; reject unsupported mappings.

Before making a realism/transfer claim, establish an independently validated
hardware source and grouped acquisitions, improve the decoder until the real-data
baseline is credible, and repeat on both bases and multiple durations/cohorts.
Add the old-generator comparison only after its circuit mapping is audited, and
include the deferred pretraining/fine-tuning arm. Calibrate one mechanism family
at a time using training data, then freeze choices before any final test.

The completed **pilot-v3 replay** preserves driver/source snapshots and hashes.
Comparison with pilot-v2 found exact agreement in splits, noise fits, profiles,
distribution statistics, training-input hashes, checkpoint metrics and per-shot
validation failure outcomes across all nine arms. The original run lacked the full
driver snapshot; the replay supplies a captured implementation after reviewed
metadata/deadline changes that did not change these results. Combined recorded
training time was **{replay["recorded_training_wall_seconds_both_runs"]:.1f} seconds**.
This checks reproducibility on the same machine and pinned libraries; it does not
promise identical files or results on different hardware/library versions.
See `data/realism/results/replay-verification.json` for the individual checks.
Raw source verification, generator-format integration,
large-sample memory/performance checks, complete mechanism calibration, final-test
transfer and final public documentation remain outstanding.

Verification at this checkpoint: **774 existing fast tests passed** (8 deselected),
**132 research checks passed**, and repository lint/format plus strict mypy passed
across 77 source files. Research tests need the isolated environment, which
contains PyTorch 2.11.0+cu128; core library versions remain pinned in `pyproject.toml`.

```powershell
.\\data\\realism\\.venv\\Scripts\\python.exe -m pytest research/realism/tests -q
.\\data\\realism\\.venv\\Scripts\\python.exe -m mypy --strict qecgen tests research/realism
```

Figures are available as both PNG and PDF beside the result files. Regenerate this
working report and all figures from the completed JSON artifacts with:

```powershell
python -m research.realism.report --pilot pilot-v3
```

To replay training, preserve the checked configuration and change only the output
in a copy. The example target must not already exist: the driver deliberately
refuses to overwrite completed results. Run from the repository root:

```powershell
$pilotReplayConfig = Get-Content research/realism/pilot_config.json -Raw | ConvertFrom-Json
$pilotReplayConfig.output = "data/realism/results/pilot-replay"
$pilotReplayConfig | ConvertTo-Json -Depth 20 |
  Set-Content data/realism/replay-config.json -Encoding ascii
.\\data\\realism\\.venv\\Scripts\\python.exe -u -m research.realism.pilot `
  --config data/realism/replay-config.json
.\\data\\realism\\.venv\\Scripts\\python.exe -m research.realism.verify_replay `
  data/realism/results/pilot-v3 data/realism/results/pilot-replay `
  --output data/realism/results/new-replay-verification.json
```

The script records input SHA-256 values in the generated figure receipt and refuses
to present an incomplete set of training runs as complete. Final public docs have
not been written at this gate.
"""


def generate(results: Path, report_path: Path, pilot_name: str = "pilot-v3") -> dict[str, Any]:
    output = results / "figures"
    output.mkdir(parents=True, exist_ok=True)
    status_path = results / pilot_name / "status.json"
    status = _read(status_path) if status_path.exists() else {"status": "not_started"}
    if status.get("status") != "complete":
        note = (
            "# Gate 2: partial pilot\n\n"
            f"Pilot status is **{status.get('status')}**. Training is incomplete; "
            "no completed comparison is reported. Existing figures, if any, belong "
            "to an earlier run and must not be treated as current results.\n"
        )
        report_path.write_text(note, encoding="utf-8")
        return {"status": "partial", "pilot_status": status}
    paths = {
        "pilot": results / pilot_name / "results.json",
        "model": results / "model/report.json",
        "reference": results / "model/reference.json",
        "drift": results / "calibration_drift_summary.json",
        "replay": results / "replay-verification.json",
    }
    inputs = {name: _read(path) for name, path in paths.items()}
    pilot = inputs["pilot"]
    if pilot["final_test_scored"] is not False:
        raise ValueError("This report is restricted to validation-only pilot results")
    expected_source = {
        "distance": 3,
        "basis": "Z",
        "rounds": 10,
        "orientation": "q10_7",
        "shots": 50000,
    }
    if any(pilot["source"].get(key) != value for key, value in expected_source.items()):
        raise ValueError("The checkpoint narrative is specific to the d3 Z r010 q10_7 cohort")
    if pilot["validation_shots"] != 9744 or pilot["split"]["partitions"]["test"]["rows"] != 9872:
        raise ValueError("Changed partition sizes require a new checkpoint narrative")
    summary = arm_summary(pilot)
    replay = inputs["replay"]
    if replay.get("passed") is not True or not all(replay["checks"].values()):
        raise ValueError("The checkpoint replay discussion requires every replay check to pass")
    snapshot_root = results / "pilot-v3"
    source_hashes = _read(snapshot_root / "source_hashes.json")
    for filename, digest in source_hashes.items():
        if hashlib.sha256((snapshot_root / "source" / filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Captured replay source disagrees with its hash: {filename}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12})
    _decoder_plot(pilot, summary, output)
    _detector_plot(pilot, output)
    _reference_plot(inputs["reference"], output)
    _calibration_plot(inputs["drift"], output)
    report_path.write_text(
        _report_text(pilot, summary, inputs["model"], inputs["reference"], inputs["drift"], replay),
        encoding="utf-8",
    )
    receipt = {
        "status": "complete_validation_report",
        "final_test_scored": False,
        "inputs_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "outputs_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.iterdir())
            if path.suffix in {".png", ".pdf"}
        },
        "arm_summary": {
            arm: {key: value for key, value in item.items() if key != "runs"}
            for arm, item in summary.items()
        },
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("data/realism/results"))
    parser.add_argument("--report", type=Path, default=Path("research/realism/GATE2_REPORT.md"))
    parser.add_argument("--pilot", default="pilot-v3")
    args = parser.parse_args()
    receipt = generate(args.results, args.report, args.pilot)
    print(json.dumps({"status": receipt["status"], "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
