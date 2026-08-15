# qecgen — A Practical Guide

A walkthrough of every command and flag, plus a plain-English explanation of what the tool
is actually doing.

`README.md` is the reference and the evidence log. `DATA_CONTRACT.md` is the precise
statement of what the files mean. **This guide is the on-ramp.** Everything below was run
against the current tree, not transcribed from source.

---

## 1. The thirty-second version

`qecgen` simulates a quantum error correction experiment and writes down what happened, so
you can train or benchmark a **decoder**.

Each row of the output file is one run of the experiment ("one shot") and holds two things:

| Column | What it is | Role |
|---|---|---|
| `detectors` | which error-detection checks fired | the **input** a decoder sees |
| `observables` | whether the logical qubit actually flipped | the **answer** it must predict |

That's it. A dataset is a pile of (syndrome, did-it-break) pairs.

Here are five real shots from a distance-3 code, straight out of the tool:

```json
{"shot": 0, "detectors": "110000000000000000000000", "observables": "0"}
{"shot": 1, "detectors": "000000000000000000000000", "observables": "0"}
{"shot": 2, "detectors": "000000000000000000001100", "observables": "0"}
{"shot": 3, "detectors": "011000000000000000000000", "observables": "0"}
{"shot": 4, "detectors": "000000000010000000100000", "observables": "0"}
```

Shot 1 is a clean run — nothing fired, nothing broke. Shots 0, 2 and 3 had errors that the
checks caught, and in each case the logical qubit survived. A decoder's job is to look at
the left-hand string and predict the right-hand one.

---

## 2. Install

Python 3.13 or newer. Dependencies are pinned exactly, because the exact Stim version is
part of what makes a dataset reproducible.

```bash
pip install -e .                  # just the tool
pip install -e ".[dev]"           # + pytest, ruff, mypy
pip install -e ".[decoders]"      # + mwpf and fusion-blossom, only for `sweep --decoder`
```

The `decoders` extra is genuinely optional. You never need it to generate, validate, export
or score anything — only to run third-party decoder baselines. It is kept separate because
both packages are pre-1.0 Rust extensions, and a platform without a prebuilt wheel would
otherwise make the whole tool uninstallable.

Two ways to invoke it:

```bash
qecgen generate ...            # the installed console script
python -m qecgen.cli generate  # equivalent
```

`python -m qecgen` does **not** work — there is no `__main__.py`, only the guard in
`cli.py`.

---

## 3. The vocabulary, briefly

You can use the tool without this section, but the flags will make more sense with it.

**Physical qubits are unreliable.** Error correction fixes that by spreading one *logical*
qubit across many *physical* ones and repeatedly measuring parity checks that reveal errors
without revealing (and thus destroying) the stored information.

- **Distance (`d`)** — how big the code is. A rotated distance-`d` surface code uses `d²`
  data qubits: 9 at d=3, 25 at d=5, 81 at d=9. Bigger `d` corrects more errors and costs
  more qubits. **This is the main knob.**
- **Rounds** — how many times the parity checks are measured before reading out. Defaults
  to `distance`, which is the standard choice for a memory experiment.
- **`p`** — the physical error rate. `0.001` means roughly one error per thousand
  operations. Real hardware is somewhere around `10⁻³`.
- **Shot** — one complete run of the experiment. One row of your file.
- **Detector** — a parity check comparing two consecutive stabilizer measurements. It fires
  (`1`) when something changed, meaning an error happened nearby. A distance-3, 3-round
  memory experiment has **24 detectors**.
- **Observable** — the ground truth: did the encoded logical qubit end up flipped? A memory
  experiment has **1 observable**.
- **Syndrome** — the whole pattern of fired detectors for one shot. The decoder's input.
- **DEM (Detector Error Model)** — Stim's description of *how* errors relate to detectors:
  a list of independent error **mechanisms**, each with a probability and a set of detectors
  it flips. This is the graph PyMatching decodes on.
- **Threshold** — the physical error rate below which a bigger code helps and above which it
  hurts. Finding it is what `sweep` does.

A key fact that shapes the whole design: **many different physical errors produce the exact
same syndrome.** That ambiguity is not a flaw — it is why the code works. It also means
"which physical error occurred?" has no unique answer, which is why this tool labels the
*logical outcome* rather than the physical fault. More on that in §11.

---

## 4. What happens when you run `generate`

```
  circuits.py          Pick a Stim task string, resolve --noise into four channel
                       probabilities, call stim.Circuit.generated
        |
        v
  sampling.py          Compile a sampler and draw --shots in chunks of --chunk-size.
                       The ONLY place sample() is called.
        |
        v
  dem.py               Ask Stim for the detector error model, parse it into sparse
                       matrices H (detectors x mechanisms) and L (observables x mechanisms)
        |
        v
  validate.py          Check shapes, widths, hashes, DEM invariants
        |
        v
  exporters/           Write HDF5 / NPZ / Parquet / JSONL
```

Every command prints its fully resolved configuration as a table *before* doing any work —
including values you didn't type. A terminal log is therefore a complete record of the run.

---

## 5. The nine commands

| Command | What it does |
|---|---|
| `generate` | One dataset, one noise setting. **Start here.** |
| `multi-env` | Several noise settings pooled into one shuffled, labelled file |
| `drift` | A training file plus drifted test files, for generalisation studies |
| `sweep` | Run real decoders across distances and rates; find the threshold |
| `validate` | Check a file is structurally sound (`--qa` adds statistical checks) |
| `inspect` | Print a file's manifest without loading the shots |
| `formats` | List the registered export formats |
| `score` | Score a proposed physical correction by its logical effect |
| `ui` | Serve `generate`/`multi-env`/`drift` as a local web page — see §13 |

---

## 6. `generate` — your first dataset

```bash
qecgen generate --distance 3 --p 0.01 --shots 2000 --structure dem --out demo.h5
```

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--distance` | int | `5` | Code distance |
| `--p` | float | `0.005` | Physical error rate |
| `--shots` | int | `1000000` | Number of runs to sample |
| `--noise` | enum | `stim_uniform_circuit_level` | Which noise channels are active — see §7.1 |
| `--rounds` | int | *distance* | Measurement rounds |
| `--basis` | `z`\|`x` | `z` | Memory Z or memory X |
| `--rotated` / `--no-rotated` | bool | `--rotated` | Rotated layout (`d²` qubits) vs unrotated (`d² + (d-1)²`) |
| `--format` | str | `hdf5` | `hdf5`, `npz`, `parquet` or `jsonl` |
| `--structure` | enum | `none` | How much DEM structure to ship — see §7.2 |
| `--emit-mechanisms` / `--no-emit-mechanisms` | bool | off | Add Contract B labels — see §7.4 |
| `--seed` | int | `0` | Master seed |
| `--chunk-size` | int | `100000` | Shots per `sample()` call — part of the reproducibility contract |
| `--out` | path | `data/dataset.h5` | Output file |

> **`--structure` defaults to `none`.** A bare `qecgen generate` produces a file with *no*
> `/dem` group — just shots. If you want the decoding graph in the file, you must ask for
> `--structure dem`. (`drift` is the exception: it defaults to `dem`.)

**What you get.** The run above prints its config, a progress bar, then:

```
wrote demo.h5  shots=2,000  content_hash=a1352c86a3521fe2...
```

and the HDF5 file looks like this:

```
/detectors                    uint8   (2000, 3)     <- 24 detectors packed into 3 bytes
/observables                  uint8   (2000, 1)     <- 1 observable packed into 1 byte
/dem/H_indices, H_indptr, H_shape     CSC of (24, 286)   detectors x mechanisms
/dem/L_indices, L_indptr, L_shape     CSC of (1, 286)    observables x mechanisms
/dem/priors                   float64 (286,)        probability of each mechanism
/dem/detector_coords          float64 (24, 3)       (x, y, t) per detector
/dem/component_*                                    decomposed error components
root attrs: manifest (full JSON), bit_order, contract, n_detectors, n_observables,
            shots, distance, rounds, chunk_size, drift_condition, qecgen_version
```

Note `(2000, 3)`: 24 detectors packed into 3 bytes. **Never infer the detector count from
the byte width** — 3 bytes could be anywhere from 17 to 24 detectors. Read `n_detectors`
from the manifest.

---

## 7. The flags that decide what's in your file

### 7.1 `--noise` — which things can go wrong

Stim's generated surface code exposes exactly four noise probabilities. The three named
models are conventions over those four. Measured at `p = 0.01`:

| `--noise` | `before_round_data_depolarization` | `before_measure_flip_probability` | `after_clifford_depolarization` | `after_reset_flip_probability` |
|---|---|---|---|---|
| `code_capacity` | 0.01 | 0 | 0 | 0 |
| `phenomenological` | 0.01 | 0.01 | 0 | 0 |
| `stim_uniform_circuit_level` | 0.01 | 0.01 | 0.01 | 0.01 |

- **`code_capacity`** — data qubits get noisy, measurements are perfect. The simplest
  textbook setting. **It forces `rounds = 1`** and *raises* if you ask for anything else,
  rather than silently overriding you.
- **`phenomenological`** — data noise plus measurement noise. Needs multiple rounds to be
  meaningful.
- **`stim_uniform_circuit_level`** (default) — everything is noisy, including the gates
  that implement the checks. The realistic one.

### 7.2 `--structure` — how much of the decoding graph you ship

| Level | Contents | Use it when |
|---|---|---|
| `none` *(default)* | shots only | The decoder is expected to learn the structure |
| `coords` | detector `(x, y, t)` positions | You want geometry but no error model |
| `dem` | `+ H`, `L`, `priors`, components | The normal choice — this is what PyMatching gets for free |
| `full` | `+ /provenance/` with circuit and DEM **text** | Auditing |

**The manifest is decoder-visible; `/provenance/` is not.** Circuit and DEM text live only
in the provenance group, physically apart, and it carries a `warning` attribute saying so.
Under a frozen-prior study that text is precisely what the condition is withholding.

### 7.3 `--format` — where it goes

Run `qecgen formats` to see this live:

| Format | Extension | Streaming | Round-trips structure |
|---|---|---|---|
| `hdf5` | `.h5` | **yes** | yes |
| `jsonl` | `.jsonl` | no | yes |
| `npz` | `.npz` | no | yes |
| `parquet` | `.parquet` | no | **no** (arrays + manifest only) |

**Use `hdf5` for anything real.** It is the only format with an incremental writer, so it
is the only one that will not materialise the whole dataset in memory first. Parquet cannot
reconstruct structure on read, so it honestly *downgrades* the level it records rather than
over-claiming.

`jsonl` is for eyeballing and fixtures. It warns above 100,000 shots. Its layout is:

```
line 1     {"__manifest__": {...}}
line 2     {"__structure__": {...}}      (only when structure_level != none)
line 3..N  {"shot": 0, "detectors": "0100...", "observables": "1"}
```

> **Do not parse those strings with `int(s, 2)`.** They are written index-order —
> `s[i]` is detector `i` — so `int(s, 2)` reverses the detector index across the entire file.

### 7.4 `--emit-mechanisms` — Contract A vs Contract B

| | Target the decoder predicts | Manifest `contract` |
|---|---|---|
| default | logical observable flips | `logical_frame` |
| `--emit-mechanisms` | + which DEM mechanisms fired | `dem_mechanism` |

Turning this on **switches the sampler**: all three arrays then come from the DEM sampler
in one draw. Sampling the circuit and the DEM separately would give two independent RNG
streams, and the labels would not explain the events sitting next to them.

> Contract B labels are indices into *Stim's DEM construction order*. They are not physical
> faults and they do not carry across noise models, distances or Stim versions. A model
> trained on them is learning to invert Stim's DEM construction. Use Contract A unless you
> specifically know why you want B.

### 7.5 `--seed` and `--chunk-size` — reproducibility

Seeds are threaded explicitly through `numpy.random.SeedSequence` spawning; no global RNG
state is touched. Same inputs → same `content_hash` (BLAKE2b-256), which `validate` checks.

**`--chunk-size` is part of the contract.** It changes the sequence of `sample()` calls and
therefore the sample stream, so reproducing a file means matching its chunk size. It's
recorded in every manifest for that reason. Determinism is guaranteed on array contents and
the content hash — never on byte-identical files.

---

## 8. `multi-env` — several noise settings in one file

Pools multiple environments into one shuffled file with an `environment_ids` column, so a
model can't learn the order.

```bash
qecgen multi-env --distance 3 --p 0.004 --p 0.008 --p 0.012 \
    --shots-per-env 500 --out multi.h5
```

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--distance` | int | `5` | |
| `--p` | float, **repeatable** | `0.003 0.005 0.008 0.012` | One value per environment |
| `--shots-per-env` | int | `250000` | Total shots = this × number of `--p` values |
| `--noise` | enum | `stim_uniform_circuit_level` | |
| `--drift-axis`, `--axis` | enum | `p` | Which property varies — see §9.1 |
| `--base-p` | float | none | **Required for non-`p` axes** |
| `--rounds` | int | *distance* | |
| `--basis` | `z`\|`x` | `z` | |
| `--format` | str | `hdf5` | |
| `--structure` | enum | `none` | |
| `--emit-mechanisms` | bool | off | Rejected if environments have different mechanism counts |
| `--seed` | int | `0` | |
| `--chunk-size` | int | `100000` | |
| `--out` | path | `data/train_multi.h5` | |

On a non-`p` axis the repeated `--p` values are **axis values, not error rates** — the
physical rate comes from `--base-p`:

```bash
qecgen multi-env --axis xz_bias --base-p 0.008 --p 0.5 --p 4.0 --p 16.0 \
    --shots-per-env 100000 --out train_bias.h5
```

---

## 9. `drift` — train here, test over there

Writes a training file plus one test file per `--test-p`, into a **directory**.

```bash
qecgen drift --distance 3 --train-p 0.005 --test-p 0.008 --test-p 0.012 \
    --shots 500 --condition frozen_prior --out data/drift
```

```
data/drift/train.h5        condition=oracle_calibrated  structure_from_env=0
data/drift/test_0.008.h5   condition=frozen_prior       structure_from_env=0
data/drift/test_0.012.h5   condition=frozen_prior       structure_from_env=0
```

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--distance` | int | `5` | |
| `--train-p` | float | `0.005` | Training rate; also the base rate on non-`p` axes |
| `--test-p` | float, **repeatable** | `0.007 0.010 0.014` | |
| `--condition` | enum | `frozen_prior` | See below |
| `--drift-axis`, `--axis` | enum | `p` | |
| `--shots` | int | `100000` | Per file |
| `--noise` | enum | `stim_uniform_circuit_level` | |
| `--rounds` | int | *distance* | |
| `--basis` | `z`\|`x` | `z` | |
| `--format` | str | `hdf5` | |
| `--structure` | enum | **`dem`** | Note: differs from the other commands |
| `--seed` | int | `0` | |
| `--chunk-size` | int | `100000` | |
| `--out` | path | `data/drift` | A **directory** |

Two things bite people here:

- **`train.h5` is always `oracle_calibrated`,** even under `--condition frozen_prior`. Only
  the *test* files are frozen. The training file legitimately ships its own DEM.
- **Filenames use `%g` formatting, which trims trailing zeros.** `--test-p 0.010` writes
  `test_0.01.h5`. Two test values that collide into one filename raise rather than
  silently overwriting — and the check runs **before** any sampling, so a long run
  cannot complete and then refuse to write.

There is no `--base-p` and no `--emit-mechanisms` on `drift`.

### The condition — the whole point of this command

| Condition | The test file ships… | What it measures |
|---|---|---|
| `oracle_calibrated` | its **own** DEM | Nothing about generalisation — the decoder has been handed the true test-time noise |
| `frozen_prior` | the **training** DEM | Real generalisation: the decoder must cope with noise it was never told about |
| `not_applicable` | — | Single-environment files |

This is **stated in the manifest, never inferred**. `structure_source_environment_id` and
`structure_dem_sha` make it auditable: under `frozen_prior`, a test file's
`structure_dem_sha` equals the training file's.

### 9.1 Drift axes

| Axis | Axis value means | Domain | Unbiased point |
|---|---|---|---|
| `p` | the physical error rate itself | `[0, 1]` | — |
| `xz_bias` | η = pz / (px + py) on single-qubit noise | η > 0 | **0.5** |
| `measurement_ratio` | `before_measure_flip_probability = base_p × value` | ≥ 0, effective ceiling `1/base_p` | **1.0** |

- **η = 0.5 is unbiased, not 1.0** — there are two channels in the denominator, so equal
  rates give ½. That's the depolarising point, `px = py = pz = p/3`. η > 0.5 is Z-biased.
- `xz_bias` rewrites **every** `DEPOLARIZE1`, which means data noise *and* single-qubit gate
  noise. Two-qubit `DEPOLARIZE2` is left symmetric. The manifest records the true scope in
  `bias_scope`, so never describe such a dataset as simply "biased noise" without the
  qualifier.
- On a non-`p` axis, `drift` places the *training* environment at the unbiased point and
  drifts the test values away from it.
- `measurement_ratio` overshoot is caught late and the error names the *channel*, not the
  axis: `channel before_measure_flip_probability must lie in [0, 1], got 2.0`.

---

## 10. Checking what you made

### `validate`

```bash
qecgen validate demo.h5           # fast, deterministic
qecgen validate demo.h5 --qa      # + slow statistical checks
```

| Flag | Default | Notes |
|---|---|---|
| `--format` | inferred from extension | |
| `--qa` / `--no-qa` | `--no-qa` | Adds sampling-based checks |

Real output:

```
[PASS] detectors.dtype: uint8
[PASS] detectors.packed_width: ceil(24/8) = 3, got 3
[PASS] manifest.shots: manifest says 2000, arrays hold 2000
[PASS] bit_order: recorded as 'little'
[PASS] contract_a.no_mechanisms: contract=logical_frame, mechanisms array absent
[PASS] content_hash: manifest a1352c86a3521fe2... vs actual a1352c86a3521fe2...
[PASS] dem.H_shape: expected (24, 286), got (24, 286)
[PASS] dem.priors_in_range: range [0.00067, 0.0522]
[PASS] dem.component_weight_le_2: component weight histogram {1: 368, 2: 188}
[PASS] dem.mechanism_weight_distribution: reported, not asserted: {1: 24, 2: 115, 3: 98, 4: 49}
all structural checks passed
```

Note the last two lines. The ≤2-detector bound is asserted on **components**; mechanism
weights legitimately run 1–4, so that histogram is *reported*, not asserted. Asserting it
would fail on correct data.

`validate` and `validate --qa` are deliberately separate: deterministic checks must never
fail for sampling reasons. If structural checks fail, QA is skipped with
`skipping --qa: structural checks failed first` and exit code 1.

### `inspect`

```bash
qecgen inspect demo.h5              # manifest only
qecgen inspect demo.h5 --show-text  # + circuit and DEM text (needs --structure full)
```

Prints all 27 manifest fields plus a per-environment table showing the axis, axis value,
`p`, noise model, shot count and the resolved channel dictionary. For HDF5 it reads only
the manifest attribute, so it stays fast on a million-shot file.

### `formats`

No flags. Prints the registry table from §7.3.

---

## 11. `score` — grading a proposed correction

This answers: *"here is the correction my decoder proposes; did it work?"*

```bash
qecgen score demo.h5 --correction proposed.npz --unpacked
```

| Flag | Type | Default | Notes |
|---|---|---|---|
| `path` | path | **required** | Dataset holding the true observables |
| `--correction` | path | **required** | NPZ with `correction_x` and `correction_z` |
| `--format` | str | inferred | |
| `--unpacked` / `--no-unpacked` | bool | `--no-unpacked` | Arrays are bool `(shots, n_data_qubits)` |
| `--alpha` | float | `0.05` | 1 − confidence level |

**The NPZ must contain exactly two arrays**, `correction_x` and `correction_z`:

| Mode | dtype | Shape |
|---|---|---|
| default (packed) | `uint8` | `(shots, ceil(n_data_qubits/8))`, little-endian |
| `--unpacked` | `bool` | `(shots, n_data_qubits)` |

`n_data_qubits` is `d²` for a rotated code (9 at d=3, 25 at d=5) and `d² + (d-1)²`
unrotated. **Use `--unpacked` unless you have a reason not to** — then qecgen packs for
you and you can't accidentally take NumPy's big-endian default.

A complete worked example, the "do nothing" baseline:

```python
import numpy as np

shots, n_data = 2000, 9  # rotated d=3
np.savez(
    "proposed.npz",
    correction_x=np.zeros((shots, n_data), dtype=bool),
    correction_z=np.zeros((shots, n_data), dtype=bool),
)
```

```
logical error rate under the supplied correction
  logical=0.18150 [0.16483, 0.19910] (363/2000) n_data=9 n_obs=1 schema=41e83ddd20ea
```

18.15% — and since a zero correction induces no flips, that is exactly the raw logical
flip rate of the data. **That is your floor: any real decoder has to beat it.**

**The scoring rule.** For observable `k` with symplectic masks `(L_x[k], L_z[k])`:

```
predicted_flip[k] = parity(popcount(C_x & L_z[k]) + popcount(C_z & L_x[k]))
success(shot)     = all(predicted_flip[k] == true_flip[k] for every k)
```

That's just anticommutation of two Pauli operators. A `Y` correction sets both `C_x` and
`C_z` and falls out of the same expression.

**The schema digest matters.** It identifies the qubit ordering the score was computed
under. The same array under a different ordering gives a different, equally plausible
number — so a score without a digest is unfalsifiable. The schema is derived on demand
from the manifest's decoder-visible parameters (it depends on the *code*, not the noise),
which means scoring a frozen-prior test file never reads that file's own error model.

### Why this isn't "predicting the physical error"

`score` runs the map **forwards**: you supply a correction, it reports the logical effect.
Single-valued, deterministic, no tie-breaking.

The **inverse** — *given a syndrome, which physical fault occurred?* — is refused, and not
out of laziness. Codes are degenerate: many different physical faults give the identical
syndrome and the identical logical effect, and that indistinguishability is exactly what
makes the code work. There is no unique answer to learn. The degeneracy that makes the
inverse ill-posed is what makes the forward direction well-posed: every member of an
equivalence class has the same logical effect, so the answer doesn't depend on which
representative you hold.

No file gains a label column from `score`; the manifest's `contract` stays `logical_frame`.

---

## 12. `sweep` — finding the threshold

Runs real decoders through `sinter` across a grid of distances and error rates. This path
is independent of the dataset path — it doesn't read or write dataset files.

```bash
qecgen sweep --distances 3 --distances 5 --p-range 0.004:0.016:4 \
    --max-errors 40 --max-shots 20000 --out results/sweep.csv
```

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--distances` | int, **repeatable** | `3 5 7` | |
| `--p-range` | str | `0.001:0.020:8` | `low:high:count`, **linearly** spaced |
| `--max-errors` | int | `500` | Primary stopping condition |
| `--max-shots` | int | `100000000` | Ceiling |
| `--workers` | int | `4` | Parallel processes |
| `--noise` | enum | `stim_uniform_circuit_level` | |
| `--basis` | `z`\|`x` | `z` | |
| `--decoder` | str, **repeatable** | `pymatching` | See below |
| `--out` | path | `results/sweep.csv` | Plot and JSON go alongside |

Three files come out: `sweep.csv`, `sweep.png`, `sweep.threshold.json`.

```csv
decoder,distance,p,rounds,noise_model,basis,shots,errors,discards,logical_error_rate,ci_low,ci_high
pymatching,3,0.004,3,stim_uniform_circuit_level,z,4369,42,0,0.0096131838,0.0069368735,0.012972239
pymatching,5,0.004,5,stim_uniform_circuit_level,z,5393,41,0,0.0076024476,0.0054610121,0.010299585
```

and a summary:

```
pymatching  crossing: p ~ 0.008
  p=0.004    Lambda=1.26 [0.81, 1.97] d=3,5
  p=0.008    Lambda=0.746 [0.501, 1.11] d=3,5  (at or above crossing)
```

**Reading it.** Λ is the error-suppression factor from raising the distance. Λ > 1 means a
bigger code helps — you're below threshold. Λ < 1 means it hurts. The crossing is where
the curves for different distances meet.

The run above used `--max-errors 40`, which is far too few — look at how wide those
intervals are, and that the first one straddles 1. **Use the default 500 for anything you
intend to quote.** Results are reported as results, never asserted, and both the crossing
and Λ depend on the channel convention, rounds, basis and decoder.

### Decoders

With the pinned sinter 1.16.0, `--decoder` accepts exactly these:

| Name | Needs | Notes |
|---|---|---|
| `pymatching` | — | The default. MWPM. |
| `pymatching-correlated` | — | Correlated variant. **The only name with a hyphen.** |
| `vacuous` | — | Never predicts a flip |
| `fusion_blossom` | `pip install fusion-blossom` | Note module/package spelling differs |
| `hypergraph_union_find` | `pip install mwpf` | |
| `mw_parity_factor` | `pip install mwpf` | Same package as above |

- **`vacuous` is a ceiling, not a floor.** It never corrects anything, so it produces the
  *highest* curve — a no-decoding control. Measured at d=3, p=0.01: vacuous `0.1978`
  vs pymatching-correlated `0.0513`.
- **`hypergraph_union_find` is not the union-find you may be expecting.** It's MWPF's
  hypergraph solver, not Delfosse–Nickerson. Measured in this repo it is *more* accurate
  than PyMatching at every sampled point.
- Names are matched **exactly** — case-sensitive, no whitespace trimming, no aliases.
  `PyMatching` and `pymatching_correlated` are both rejected.

Names are validated *before* collection starts, which turns what sinter would report as a
multiprocessing traceback from inside a worker into one actionable line:

```
unknown decoder 'mwpm'; the installed sinter provides: fusion_blossom,
hypergraph_union_find, mw_parity_factor, pymatching, pymatching-correlated, vacuous
```

```
decoder 'fusion_blossom' is valid but its backend is not installed: no module named
'fusion_blossom'. Install it with: pip install -e ".[decoders]" (or pip install
fusion-blossom). Nothing else about qecgen requires it.
```

Note the sweep's `--decoder` does **not** change the QA oracle in `validate --qa`, which is
always PyMatching.

---

## 13. `ui` — the same three commands in a browser

```bash
pip install -e ".[ui]"
qecgen ui                      # http://127.0.0.1:8765; opens your browser once it's up
qecgen ui --data-root runs     # confine reads and writes to a different directory
```

`ui` serves a local web page for `generate`, `multi-env` and `drift` — the three
dataset-producing commands — with a cost preview before you commit, a live progress bar,
a run history that survives restarts, and a browser for the files under the data root.
Every run goes through the same `run.py` layer the CLI uses, so a file made in the
browser is exactly the file the terminal would have made, staged writes and all.

Three properties are deliberate and not configurable:

- **Loopback only.** The API writes files and spawns processes for whoever can reach it,
  with no authentication. `--host` accepts `127.0.0.1` and `localhost` and refuses
  everything else — including `::1`, because the host-header check cannot match IPv6
  literals, so binding it would serve a UI that rejects every request. Use an SSH tunnel
  if you need it from another machine.
- **Confined to `--data-root`** (default `data/`). Every path the browser sends is
  resolved and checked: `..`, absolute paths elsewhere, and the data root itself are all
  refused.
- **The frontend is built on demand.** If the bundle is missing, the page names the
  build command (`cd frontend && npm ci && npm run build`) instead of serving a blank
  screen, and the API keeps working either way.

Frontend development is two processes, not a rebuild loop: `qecgen ui --dev` serves the
API and allows exactly the Vite dev origins; `npm run dev` in `frontend/` serves the
pages on 5173 and proxies `/api` across. Without `--dev`, a cross-origin request is
refused outright.

---

## 14. Traps

These produce **well-formed files containing wrong data** — nothing errors, and casual
inspection looks fine.

1. **Bit order is little-endian, everywhere.** NumPy's `packbits`/`unpackbits` default to
   `bitorder='big'`, the opposite of Stim and PyMatching. Use
   `qecgen.sampling.unpack_bits(...)` and `qecgen.correction.pack_correction(...)`. Taking
   NumPy's default maps qubit 0 onto bit 7 and silently produces a wrong answer.
2. **Packed width never implies true width.** 3 bytes could be 17–24 detectors. Always read
   `n_detectors` / `n_observables` / `n_mechanisms` from the manifest.
3. **`int(s, 2)` reverses JSONL detector strings.** `s[i]` is detector `i`.
4. **`--structure` defaults to `none`** on `generate` and `multi-env`. No `/dem` group
   unless you ask.
5. **`drift` writes `train.h5` as `oracle_calibrated`** regardless of `--condition`.
6. **`%g` filenames trim zeros**: `--test-p 0.010` → `test_0.01.h5`.
7. **η = 0.5, not 1.0, is unbiased** on the `xz_bias` axis.
8. **Reproducing a file requires matching `--chunk-size`,** not just the seed.
9. **Go's `bufio.Scanner` defaults to a 64 KB token cap** and will fail on the JSONL
   structure line of every d≥3 file. Fix: `scanner.Buffer(make([]byte, 0, 64<<10), 64<<20)`.
   Python, jq, Node and Java need nothing.

---

## 15. What this tool deliberately does not do

Not built, and not stubbed in a way that implies otherwise:

- **Any Nexus client, import or exporter.** The Nexus input format is not known.
- **Decoder implementations or adapters.** `--decoder` resolves names and hands them to
  sinter. There is deliberately no `Decoder` protocol — `sinter.Decoder` already is one.
- **A latency harness.** Sinter's timing is *throughput*, not per-shot decoder latency.
- **Contract C** — physical Pauli fault labels as a training target. See §11.
- **On-disk export of the correction schema.** `score` derives it on demand.
- **Codes other than the surface code**, real hardware data, custom MWPM, or `pickle`.

---

## 16. Cookbook

```bash
# Smallest useful thing — inspect it by hand
qecgen generate --distance 3 --p 0.01 --shots 20 --format jsonl --out tiny.jsonl

# A real training set with the decoding graph attached
qecgen generate --distance 5 --p 0.005 --shots 1000000 --structure dem \
    --out data/d5_p005.h5

# Same, plus Contract B mechanism labels
qecgen generate --distance 5 --p 0.005 --shots 1000000 --structure dem \
    --emit-mechanisms --out data/d5_mech.h5

# Pool four error rates, shuffled and labelled
qecgen multi-env --distance 5 --p 0.003 --p 0.005 --p 0.008 --p 0.012 \
    --shots-per-env 250000 --out data/train_multi.h5

# Generalisation study: train at 0.005, test at three unseen rates
qecgen drift --distance 5 --train-p 0.005 --test-p 0.007 --test-p 0.010 \
    --test-p 0.014 --condition frozen_prior --out data/drift

# Bias drift instead of rate drift
qecgen multi-env --axis xz_bias --base-p 0.008 --p 0.5 --p 4.0 --p 16.0 \
    --shots-per-env 100000 --out data/train_bias.h5

# Always check what you made
qecgen validate data/d5_p005.h5 --qa
qecgen inspect  data/d5_p005.h5

# Threshold sweep with two decoders
qecgen sweep --decoder pymatching --decoder hypergraph_union_find \
    --distances 3 --distances 5 --distances 7 --p-range 0.001:0.020:8

# Grade a proposed correction
qecgen score data/d5_p005.h5 --correction proposed.npz --unpacked
```

A reasonable first session: **generate → inspect → validate → sweep**.
