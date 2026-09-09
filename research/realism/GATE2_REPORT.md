# Gate 2: working model-design results

> Historical checkpoint, retained on 2026-09-09. The user subsequently authorized
> completing integration, evaluation and public documentation. The status and
> approval gates below describe the earlier checkpoint; current capabilities and
> results are in [docs/REALISM.md](../../docs/REALISM.md). Regenerating this historical
> report does not replace the final experiment.

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
| Real training | 899, 975, 912 | 9.5307% | 9.2262%-10.0062% |
| Uniform synthetic | 1016, 1076, 961 | 10.4440% | 9.8625%-11.0427% |
| Heterogeneous readout | 996, 1042, 1017 | 10.4509% | 10.2217%-10.6938% |

The heterogeneous arm changes failure counts by **-20, -34,
56** relative to uniform: **2 improve, 1 worsen**.
Its mean is **0.00684 percentage points worse**. These are descriptive
seed summaries on the same validation shots, not independent experiments or
confidence intervals. No significance or population-wide improvement is claimed.

Both fitted matching priors give
**799/9,744 = 8.1999%**.
Even the real-trained GRU is worse. The GRU learns beyond the constant-zero
prediction (2848/9,744), but competence and convergence
remain unresolved: only **4/9** runs observed the configured validation plateau,
and some hit the 80-epoch ceiling. Validation also selected checkpoints. These are
development results, not a locked final comparison. The earlier intake attempt
failed before training; the two completed runs used identical training settings.

Nine training runs took **186.4 seconds** in total; the complete pilot
took **197.1 seconds**. Checkpoints, fitted profiles, input hashes,
seeds and training settings were recorded. The untouched legacy generator was
regression-tested but was **not** scored as an interchangeable input arm: its
generated circuit conventions must first be mapped to the experimental circuit.
Synthetic pretraining followed by real fine-tuning remains untested.

![All decoder seeds and means](../../data/realism/results/figures/decoder_validation.png)

## Distribution, physics and performance checks

Detector-rate RMSE changes from **0.01209090** to
**0.01205215**, a small reduction. Pairwise covariance
RMSE moves in both directions:

| Pair group | Uniform | Heterogeneous readout | Direction |
|---|---|---|---|
| Spatial | 0.00194783 | 0.00200582 | Worse |
| Temporal | 0.00292484 | 0.00287558 | Better |
| Space and time | 0.00069166 | 0.00069570 | Worse |

These probabilities/covariances are dimensionless. Both synthetic arms have
syndrome-weight quantiles **4, 9, 13** at the 50th/90th/99th percentiles,
versus **4, 9, 14** for validation. The observed upper tail remains underrepresented.
These statistics use only the first synthetic seed and have no sampling-uncertainty
estimate. Small marginal agreement does not establish accurate correlations or transfer.

![Detector statistics][detector-figure]

[detector-figure]: ../../data/realism/results/figures/detector_statistics.png

The exact small-system reference exposes a known approximation error rather than
hiding it. With illustrative 1 us idles, T1=20 us and T2=30 us, Pauli twirling
introduces **2.4385%**
excitation into a ground state that exact zero-temperature damping leaves alone.
In the three-round parity motif, the joint-outcome total variation is
**0.13768**
for preparation 00 and
**0.12485**
for 11. Exact means the stated two-level damping/dephasing reference, not complete
transmon physics. These are illustrative parameters, not hardware measurements.

![Pauli approximation discrepancy][reference-figure]

[reference-figure]: ../../data/realism/results/figures/pauli_reference_discrepancy.png

On the local CPU, one 100,000-shot d=3, 10-round trial measured approximately
**2.07 million shots/s** for static
uniform noise,
**2.00 million shots/s**
for static heterogeneity, and **71.1 thousand
shots/s** for the combined dynamic illustration. Fresh-process peak working set
was about **51.7 MiB**,
including interpreter/imports. These single-run timings include uncontrolled
background activity and cannot be projected to large distances. The pilot
materializes output; production streaming still needs integration and testing.

Calibration-history analysis retained **3,686** sufficiently
populated series. Across all four backends, last-observation prediction improves
pooled held-out mean absolute error in **10/20 groups**.
Across individual series, **1,377 improve and 2,309 worsen**.
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
training time was **372.8 seconds**.
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
.\data\realism\.venv\Scripts\python.exe -m pytest research/realism/tests -q
.\data\realism\.venv\Scripts\python.exe -m mypy --strict qecgen tests research/realism
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
.\data\realism\.venv\Scripts\python.exe -u -m research.realism.pilot `
  --config data/realism/replay-config.json
.\data\realism\.venv\Scripts\python.exe -m research.realism.verify_replay `
  data/realism/results/pilot-v3 data/realism/results/pilot-replay `
  --output data/realism/results/new-replay-verification.json
```

The script records input SHA-256 values in the generated figure receipt and refuses
to present an incomplete set of training runs as complete. Final public docs have
not been written at this gate.
