# Realism pilot: working evidence register

> Historical research register, retained on 2026-09-09. The user subsequently
> authorized all remaining integration, evaluation and documentation work. Earlier
> approval boundaries below are historical. See [docs/REALISM.md](../../docs/REALISM.md)
> for current capabilities and final measurements; the retrieved sources below
> remain evidence for the model choices.

Research checkpoint, 2026-09-09. This is an internal model-design note, not the
final user guide or a claim that synthetic data transfers to hardware.

## Approval boundary and current baseline

The approved plan has three review gates. Gate 1 authorized a bounded acquisition,
an isolated model/import/decoder pilot, and fixed legacy references. Present the
pilot results at Gate 2 before full generator integration. Present integrated
experimental results at Gate 3 before writing final public documentation. Raw
hardware files stay outside git; acquisition receipts, licences, checksums and
preprocessing lineage belong with the acquisition catalogue. Limits for this
stage are 10 GB downloaded, 30 GB total working storage, and one local GPU-hour
for initial decoder feasibility. These limits are project choices, not physics.

Source inspection corrects the brief's starting assumption: existing qecgen
already has three explicit noise conventions. Code capacity has one round and
data depolarization only; phenomenological noise adds measurement flips; the
default uniform circuit-level path also includes reset and Clifford gate noise.
Its missing capabilities are per-qubit calibration and persistent noise processes,
not circuit-level simulation itself. Existing drift constructs separate
environments rather than one evolving device timeline. These statements follow
`qecgen/circuits.py`, `environments.py` and `sampling.py` as captured below.

`legacy_golden.json` freezes 24 pre-overhaul cases: all three conventions, both
memory bases, Contracts A and B, and two seed/chunk pairs. Each case records the
resolved channel vector, detector/observable widths, circuit and DEM SHA-256,
array SHA-256 with dtype and shape, and the existing BLAKE2b-256 content hash.
The parameters (d=3, p=0.013, 257 shots, seeds 0/20260909, chunks 31/64) are
**regression fixtures**, not measured device values. Five generator-source
fingerprints identify the precise working-tree baseline; the commit alone cannot
identify a dirty checkout.

The captured stream uses Python 3.13.6, Stim 1.16.0, NumPy 2.3.3, Windows AMD64,
little-endian bytes and AVX2. A single-environment seed is passed directly to its
sampler, without child-seed derivation. Contract B samples all three arrays from
one DEM sampler. Chunk boundaries and final remainders are part of the call
contract. Other platforms/SIMD implementations cannot be assumed to produce the
same seeded bytes; incompatible stream tests explicitly skip, while changed
library pins fail. Stim documents these seeded reproducibility restrictions in
its [versioned API reference](https://raw.githubusercontent.com/quantumlib/Stim/v1.16.0/doc/python_api_reference_vDev.md).

Verification on the captured runtime: **26 new legacy checks passed**, and
**81 total passed** when combined with the existing circuit, sampling and DEM
tests. Ruff checks, Ruff formatting checks and strict mypy passed for the two new
Python files. Run the pilot guard explicitly:

```powershell
python -m pytest research/realism/tests/test_legacy.py -q
python -m research.realism.legacy
```

The second command prints candidate fingerprints; it never overwrites the frozen
file. The current root pytest `testpaths` excludes research, so Stage 2 must
bridge this guard into normal CI before shipping the overhaul. No existing core
module, public documentation or test-discovery configuration was changed by this
baseline work.

## Mechanisms: what is supported, and what is an approximation

The target is superconducting surface-code hardware. A qubit is not simply a
classical bit: errors can alter its state, its relative phase, or move population
outside the two states used for computation. The decoder only receives evidence
from repeated parity checks, so different physical causes can produce the same
detection pattern. Contract A remains detection events as inputs and logical
observable flips as targets. Hidden causes in a simulator are not measured
physical-fault labels for a hardware dataset.

| Factor | Retrieved evidence | Modelling decision and limit |
|---|---|---|
| Relaxation and dephasing | Ghosh, Fowler and Geller [1], section II, maps amplitude/phase damping to an efficient Pauli approximation using T1, T2 and elapsed time. | Per-qubit duration-dependent channels, with a small density-matrix reference to quantify approximation error. Pauli twirling loses relaxation directionality and is not exact amplitude damping. |
| Gate, measurement, reset and idle errors | The Willow error budget [2], section III, separates local operation errors, idling during readout/reset, leakage and unwanted interactions. | Calibrate individual operations and qubit pairs. An aggregate gate error is not automatically an additional error after coherence loss; fitting must declare whether it includes that loss. |
| Qubit transition frequency | Carroll et al. [3] measure spectral and temporal T1 fluctuations consistent with coupling to nearby defects. Willow uses frequency selection and recalibration [2]. | A frequency identifies an operating point, not a universal failure probability. Use measured responses or explicitly hypothetical detuning/interaction scenarios. Do not infer coupling strength from two frequencies alone. |
| Syndrome extraction round rate | Time-dependent decoherence uses the time spent in each operation [1]. Willow's cycle includes gates, measurement, reset and leakage removal [2]. | Derive round rate from a complete duration schedule. Faster rounds cannot silently shorten fixed-duration gates; timing choices must remain visible in the configuration. |
| Temperature | Zhu et al. [4] vary temperature and study relaxation-rate fluctuations associated with defects and quasiparticles. The control-drift experiment [5] also distinguishes fluctuations in classical control instruments. | Distinguish cryostat, effective qubit and room/instrument temperatures. A measured temperature does not by itself specify T1/T2 or excited-state population. No universal temperature multiplier is justified. |
| Spatial proximity | Wilen et al. [6] observe correlated charge changes and relaxation degradation across a chip associated with particle impacts. Willow's unwanted interactions include correlated ZZ and swap-like errors [2]. | Model connectivity and calibrated interaction edges separately from a burst's spatial footprint. Physical distance or lattice adjacency alone does not establish a coupling law. |
| Leakage persistence | Miao et al. [7] show leakage spreading through multiqubit operations and reductions after removal each cycle. | Persistent hidden states can approximate correlated effects across rounds. Such a Pauli proxy does not simulate higher energy states, coherent leakage dynamics, or exact leakage-removal hardware. |
| Rare bursts | Wilen et al. [6] support particle-associated correlated errors; the earlier Google experiment [8] reports a damaging high-energy event. | Model event footprint, recovery and timing explicitly where observations support them. One observed event cannot estimate a population distribution reliably. A generic burst is not automatically a cosmic-ray event. |
| Nonstationary drift | Carroll et al. [3] demonstrate temporal variation. The later Willow study [5] separately studies natural drift and deliberately injected step/sinusoidal/stroboscopic control changes. | Prefer calibration histories; otherwise label stochastic or switching processes as assumptions. Injected drift and natural drift are separate benchmark conditions. |
| Ambient humidity | Di Giovanni et al. [9], Appendix G, varies temperature and humidity together by switching laboratory air conditioning and observes drift in XX-basis coherent measurement errors. | Keep a recorded covariate with zero default effect. This is evidence of an indirect association under combined environmental changes, not an isolated humidity coefficient or direct in-cryostat humidity effect. |

The humidity evidence refines the initial plan's wording. Saying there is *no
related experimental evidence* would be too strong. In [9], the authors identify
qubit manipulation as the likely source because ZZ-basis errors remain stable
while the extra rotations used for XX readout drift. Temperature and humidity
were not independently randomized. A nonzero humidity response therefore remains
an exploratory assumption requiring device-specific validation, not established
qubit physics.

The **actual isolated pilot** implements per-qubit/per-edge stochastic Pauli
overrides, explicit correlated Pauli event lists, and T1/T2 channels with specified
TICK-layer durations. Optional hidden processes supply across-shot drift and burst
occupancy and within-shot persistent leakage-effect proxies. Transition frequency,
cryostat/effective temperature and humidity are recorded covariates with no
automatic response. Round rates require explicit round boundaries and layer
durations. The pilot therefore supports controlled assumptions about heterogeneity
and correlation; it does not yet derive those responses from spectroscopy,
temperature measurements, or a microscopic leakage model.

For the local exponential decoherence approximation, [1], Eq. (10), gives

```text
pX = pY = (1 - exp(-duration / T1)) / 4
pZ = (1 - exp(-duration / T2)) / 2 - pX
```

This model requires positive T1/T2 and nonnegative duration, with T2 <= 2*T1 for
the underlying Markovian amplitude-damping plus dephasing interpretation. An input
outside that relation may reflect incompatible measurements or a model mismatch;
it must not be silently clipped into a different calibration. Ramsey, echo and
CPMG measurements probe different dephasing conditions: retain the protocol and
do not relabel a CPMG value as Ramsey T2. Finite-temperature population evolution
and coherent overrotations need richer reference calculations; this formula alone
does not reproduce them.

## Parameter provenance and identifiability

Every nonzero mechanism parameter must be tagged as one of: measured calibration
(source record, qubit/pair, time, units and protocol); fitted estimate (training
data and objective); or illustrative scenario (user assumption). Literature
numbers describe their cited experiment, not a generic modern processor. The
pilot may use illustrative parameters for unit tests and feasibility, but cannot
call those a calibrated Willow noise model.

Calibration JSON can contain averaged or fitted operational metrics instead of
microscopic channel probabilities. Their meaning, gate duration and calibration
timestamp matter. A snapshot transferred from one processor to another is only a
heterogeneity example. A time series with repeated poll responses is not a time
series of independent calibrations. Unit migrations, asynchronous per-qubit
measurement times, stale values, missing protocols and unavailable uncertainties
must be retained or explicitly rejected by preprocessing.

Our modelling inference is that detector statistics alone generally cannot identify
all microscopic mechanisms: several parameter settings can explain similar
observable patterns. Consequently, matching a histogram does not prove the
underlying cause has been recovered. The resulting uncertainty belongs in the
report and in any parameter sweep. Reference-model discrepancies are measured
errors of an approximation, not failing tests to be removed until agreement looks
good.

## Evaluation requirements for the next gate

The decisive question is whether training on synthetic data improves predictions
on held-out hardware shots. AlphaQubit [10] provides a primary experimental
precedent for synthetic pretraining followed by adaptation on experimental data;
this pilot's compact decoder is not a reproduction of AlphaQubit.

A newly retrieved preprint by Manor, Erhili and Jebbouri, submitted 2026-09-03,
directly examines synthetic-to-hardware decoder rankings. Its abstract reports
that operation-specific noise improved ranking agreement, while further device
calibration improved absolute rates without further improving the rankings. This
is a recent author-reported result, not a result reproduced here. It strengthens
the reason to report both transfer and failures to improve, rather than assuming
that adding detail always helps. [Study](https://arxiv.org/abs/2609.04557).

The authors also publish a
[derived Willow detector dataset](https://huggingface.co/datasets/ShayManor/willow-surface-code-detection-events).
This is a possible small-shard pilot source when direct Google archives are
unavailable. Preserve both its declared licence and the underlying Google
attribution. Label it third-party-derived hardware data; auditing its conversion
code is useful but does not replace comparing against original Google bytes.

Freeze acquisition-group train/validation/test splits before fitting anything.
Only use chronological splits when the acquisition ordering is established, and
keep overlapping subcodes or windows from one acquisition in one split. Fit noise
parameters on training data, select settings on validation, and freeze the whole
pipeline before test evaluation. A published prior of unknown fitting provenance
cannot count as a training-only prior.

Compare old qecgen, a uniform model on the experimental circuit, the calibrated
synthetic model, real-only training, and synthetic pretraining plus real
fine-tuning. All arms need the same detector/observable conventions and held-out
shots. Report logical failure probability versus duration, detection rates,
spatial/temporal correlations, weight tails and drift/burst behaviour. Literature
[2,8] motivates inspecting these beyond the mean error rate. Fitting a per-round
logical rate requires checking its assumptions; correlations can make one scalar
misleading. Use acquisition blocks for uncertainty and paired comparisons.

A competent real-trained decoder is necessary to interpret transfer. If limited
data or compute leaves it unconverged, results are feasibility measurements rather
than evidence of successful transfer. Report unchanged/worse outcomes, intervals,
failure counts, throughput, memory use and excluded mechanisms. Gate 2 must state
what actual raw files were verified; catalogue entries and cited papers alone do
not establish that their measurements have been imported correctly.

## Retrieved primary sources

Retrieved through the web tool on 2026-09-09; citations point to the original
papers or author manuscripts. No model parameter was copied from an unverified
search snippet.

1. Ghosh, Fowler and Geller (2012), *Surface code with decoherence: An analysis of
   three superconducting architectures*. [Paper and Eq. (10)](https://arxiv.org/pdf/1210.5799).
2. Google Quantum AI (2025 published volume), *Quantum error correction below the
   surface code threshold*. [Published article](https://www.nature.com/articles/s41586-024-08449-y);
   [earlier sections II-IV and supplement](https://arxiv.org/html/2408.13687v1).
   The published article reports a 14% overprediction of error suppression; the
   earlier arXiv v1 reported 20%. These are version-specific estimates, not
   interchangeable measurements. Richer modelling is not evidence of completeness.
3. Carroll et al. (2022), *Dynamics of superconducting qubit relaxation times*.
   [Author manuscript](https://arxiv.org/abs/2105.15201).
4. Zhu et al. (2025), *Disentangling the Impact of Quasiparticles and Two-Level
   Systems on the Statistics of Superconducting Qubit Lifetime*.
   [Author manuscript](https://arxiv.org/html/2409.09926v2).
5. Google Quantum AI (2026 version), *Reinforcement learning control of quantum
   error correction*. [Natural versus injected drift](https://arxiv.org/html/2511.08493v4).
6. Wilen et al. (2021), *Correlated Charge Noise and Relaxation Errors in
   Superconducting Qubits*. [Author manuscript](https://arxiv.org/abs/2012.06029),
   DOI [10.1038/s41586-021-03557-5](https://doi.org/10.1038/s41586-021-03557-5).
7. Miao et al. (2023), *Overcoming leakage in scalable quantum error correction*.
   [Author manuscript](https://arxiv.org/abs/2211.04728).
8. Google Quantum AI (2023), *Suppressing quantum errors by scaling a surface
   code logical qubit*. [Author manuscript](https://arxiv.org/abs/2207.06431).
9. Di Giovanni et al. (2025), *Benchmarking the quality of multiplexed qubit
   readout beyond assignment fidelity*, Phys. Rev. Applied 24, 044043.
   [Published Appendix G, p. 12 and Fig. 11, p. 13](https://journals.aps.org/prapplied/pdf/10.1103/dpft-lxtx);
   [accessible preprint Appendix F](https://arxiv.org/html/2502.08589v1).
10. Bausch et al. (2024), *Learning high-accuracy error decoding for quantum
    processors*. [Nature](https://www.nature.com/articles/s41586-024-08148-8).
