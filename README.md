# qecgen

Surface code QEC dataset generator for decoder benchmarking. Builds circuits with Stim,
samples detection events, parses detector error models, validates the output, and exports
it through a pluggable exporter layer.

**Every file this tool produces is a syndrome-to-logical-frame dataset**: per-shot
detection events mapped to per-shot logical observable flips. It contains no physical
Pauli fault labels. Read [`DATA_CONTRACT.md`](DATA_CONTRACT.md) before using the output
for anything — the three things a decoder can be asked to predict are not
interchangeable, and conflating them is the most likely way this project produces
confident, wrong results.

> **Nexus compatibility is not claimed.** The Nexus input format is unknown. No Nexus
> exporter exists. This README will not claim compatibility until a Nexus exporter passes
> a fixture supplied by the Nexus team. Adding one is a single new file — see
> [Adding an exporter](#adding-an-exporter).

**New to surface codes?** [qecgen-learn](https://github.com/HaiderAli3D/qecgen-learn) is a
six-lesson interactive course covering what a decoder sees, what the distance buys you, what
an error rate means, where the labels come from, what is in a file, and how to run a study
whose result supports the claim you want to make. About an hour, in the browser. This README
is the reference; [`GUIDE.md`](GUIDE.md) is the task-oriented walkthrough.

---

## Installation

Requires Python 3.13+.

```bash
pip install -e .              # runtime
pip install -e ".[dev]"       # plus pytest, ruff, mypy, httpx
pip install -e ".[ui]"        # plus fastapi and uvicorn, for `qecgen ui`
pip install -e ".[decoders]"  # plus mwpf and fusion-blossom, for `sweep --decoder`
```

The `decoders` extra is **not** required to generate, validate or export a dataset — only
to run a decoder baseline against one. It is deliberately separate because both packages
are pre-1.0 Rust extensions, and a platform without a wheel would otherwise make `qecgen
generate` uninstallable, taking eight commands that never decode anything down with it.

The `ui` extra follows the same rule for the same reason: eight commands work without a
web stack, and none of them should become uninstallable if one fails to resolve. All four
of its packages ship a PEP 561 `py.typed`, so none of them needed an entry in the mypy
`ignore_missing_imports` list — the absence is the notable part.

Dependencies are pinned exactly, because dataset reproducibility depends on the Stim
version:

| Package | Pinned | Package | Pinned |
|---|---|---|---|
| stim | 1.16.0 | h5py | 3.15.0 |
| sinter | 1.16.0 | pyarrow | 25.0.1 |
| pymatching | 2.4.0 | matplotlib | 3.10.8 |
| numpy | 2.3.3 | typer | 0.27.1 |
| scipy | 1.16.2 | rich | 14.2.0 |

Verify the install:

```bash
ruff check . && ruff format --check .
mypy --strict qecgen tests
pytest -m "not slow"        # 398 fast structural tests
pytest -m slow              # 7 statistical and integration tests
```

The web UI needs one more step, because its bundle is built rather than committed:

```bash
cd frontend && npm ci && npm run build   # writes qecgen/ui/static
```

The suite is green without the `decoders` extra: tests needing `mwpf` are gated on
`importlib.util.find_spec`.

---

## Bit ordering, and why it matters

**Every packed array in every file is little-endian.** Every manifest records
`bit_order="little"`.

NumPy's `packbits`/`unpackbits` default to `bitorder='big'`, which is the **opposite** of
the convention Stim and PyMatching use. Taking the default silently reverses the bits
within every byte and produces a well-formed file containing wrong data — detector 0
becomes detector 7, and nothing crashes.

Verified empirically against the installed libraries:

```
np.unpackbits(packed, axis=1, count=n_detectors, bitorder="little") == unpacked sample  -> True
np.unpackbits(packed, axis=1, count=n_detectors, bitorder="big")    == unpacked sample  -> False
```

Sinter's own decoder contract states it directly: *"All data taken and returned must be
bit packed with `bitorder='little'`."*

Rules followed throughout:

- Sampling always uses `sample(shots, separate_observables=True, bit_packed=True)`. Stim
  does the packing; we never pack ourselves.
- Any code path touching `packbits`/`unpackbits` passes `bitorder="little"` explicitly.
- Use `qecgen.sampling.unpack_bits(packed, n_bits)` rather than calling NumPy directly.

**Packed width does not tell you the true width.** Packing pads to a byte boundary, so a
3-byte row could hold anywhere from 17 to 24 detectors. `n_detectors` and `n_observables`
are therefore explicit manifest fields, never inferred from array shape.

---

## Noise models: exactly which channels each one sets

The full channel vector is stored per environment in every manifest, so a results table
can state precisely which channels were active. A model name alone is not enough.

| Model | `after_clifford_depolarization` | `after_reset_flip_probability` | `before_measure_flip_probability` | `before_round_data_depolarization` | rounds |
|---|---|---|---|---|---|
| `code_capacity` | 0 | 0 | 0 | **p** | forced to 1 |
| `phenomenological` | 0 | 0 | **p** | **p** | distance |
| `stim_uniform_circuit_level` (default) | **p** | **p** | **p** | **p** | distance |

`STIM_UNIFORM_CIRCUIT_LEVEL` is named that way deliberately. Setting all four channels to
one value is **one valid synthetic convention, not a universal definition of
"circuit-level noise"**. Hardware-motivated models weight these channels differently, and
the threshold location moves when the convention changes.

`CODE_CAPACITY` forces `rounds=1` and **raises** if a conflicting `rounds` is passed,
rather than silently overriding it — a silent override would make the manifest disagree
with the file.

---

## CLI

Every command prints its fully resolved configuration before doing any work, so a terminal
log is a complete record of the run, including values that were defaulted rather than
passed.

### `qecgen generate`

Single-environment dataset.

```bash
qecgen generate --distance 5 --p 0.005 --shots 1000000 \
    --noise stim_uniform_circuit_level --format hdf5 --structure dem \
    --out data/d5_p005.h5
```

Add `--emit-mechanisms` for Contract B labels. Note this switches sampling to the DEM
sampler so all arrays come from one consistent draw (see [Contract B](#contract-b)).

### `qecgen multi-env`

Pooled dataset spanning several environments, shuffled with a seeded permutation so row
order carries no environment signal.

```bash
qecgen multi-env --distance 5 --p 0.003 --p 0.005 --p 0.008 --p 0.012 \
    --shots-per-env 250000 --out data/train_multi.h5
```

Each rate gets its own circuit, DEM and `EnvironmentSpec`; every shot carries an
`environment_id`. Non-`p` axes need `--base-p`:

```bash
qecgen multi-env --drift-axis xz_bias --base-p 0.008 --p 0.5 --p 4.0 --p 16.0 \
    --shots-per-env 100000 --out data/train_bias.h5
```

### `qecgen drift`

Training set plus drifted test sets, under an **explicit, recorded** condition.

```bash
qecgen drift --distance 5 --train-p 0.005 --test-p 0.007 --test-p 0.010 --test-p 0.014 \
    --condition frozen_prior --out data/drift/
```

### `qecgen sweep`

Sinter-driven threshold sweep. `--max-errors` is the primary stopping condition;
`--max-shots` is a ceiling.

```bash
qecgen sweep --distances 3 --distances 5 --distances 7 \
    --p-range 0.001:0.020:8 --max-errors 500 --workers 8 --out results/sweep.csv
```

Emits three files, all staged and committed together like every dataset write: the CSV, a
log-y plot with Clopper-Pearson error bars on every point (one curve per (decoder,
distance); zero-error points appear as downward carets at their upper bound), and a
threshold summary. The plot and summary are named by *replacing* the `.csv` suffix —
`results/sweep.csv` is joined by `results/sweep.png` and `results/sweep.threshold.json`.
A rate is never emitted without an interval.

**Decoder baselines.** `--decoder` is repeatable and defaults to `pymatching`:

```bash
qecgen sweep --decoder pymatching --decoder hypergraph_union_find \
    --distances 3 --distances 5 --p-range 0.005:0.015:3 --max-errors 500
```

Names are validated against `sinter.BUILT_IN_DECODERS` **before** any collection starts,
and a name whose backing package is absent is reported as an install instruction rather
than as a typo. sinter discovers both problems only inside a worker, after every circuit
in the grid has been built.

> **`hypergraph_union_find` is not the union-find you may be expecting.** It is MWPF's
> hypergraph solver, not the Delfosse-Nickerson union-find usually described as "faster
> and slightly less accurate than MWPM". Measured here it is *more* accurate than
> PyMatching at every sampled point (d=3 and d=5, p ∈ {0.005, 0.010, 0.015}). Nothing in
> the test suite asserts an accuracy ordering between the two, because the ordering is a
> property of these particular decoders rather than of this code.

Both decoders receive the **same** DEM: sinter derives one per task with
`decompose_errors=True, approximate_disjoint_errors=True` and shares it. That DEM is
byte-identical to the one `qecgen` exports, verified for uniform circuit-level noise and
for the `xz_bias` `PAULI_CHANNEL_1` rewrite.

### Threshold crossing and exponential suppression

Both are **reported, never asserted**. The `.threshold.json` sidecar holds, per decoder,
the bracketing crossing and one suppression fit per `p`. The crossing is claimed only
from *separated* intervals: the largest distance's interval must sit entirely below the
smallest's at some lower `p` and entirely above it at the reported one — overlapping
intervals claim nothing, so two statistically indistinguishable points (or two zero-error
points in a quick sweep) are never reported as a crossing:

```
pymatching  crossing: p ~ 0.009
  p=0.001    Lambda=7.01 [5.33, 9.22] d=3,5,7 chi2/dof=1.02
  p=0.009    Lambda=0.785 [0.727, 0.847] d=3,5,7  (at or above crossing)
```

The model is `p_L(d) = A * Lambda^(-(d+1)/2)`, fitted by weighted least squares on
`log p_L`, weighting each point by its Clopper-Pearson interval width in log space. With
exactly two distances it degenerates to the pairwise ratio `p_L(d) / p_L(d+2)`, so nothing
is lost against reporting ratios; with three or more it also yields a residual, which is
the only honest signal that the points are in the exponential regime at all.

Three deliberate choices:

- **Zero-error points are excluded, never pseudo-counted.** No `0.5/n`, no Jeffreys, no
  `max(k, 1)` — substituting a count fabricates a measurement. The excluded distances are
  named in `excluded_zero_error`. It is almost always the *largest* distance that produces
  one, so dropping it removes the most-suppressed point and biases `Lambda` **downward**:
  a reported `Lambda` under-states suppression whenever that list is non-empty.
- **`Lambda < 1` is reported, not clamped.** Above threshold, larger distance genuinely
  hurts.
- **`suppressing` is true only when the whole interval lies above 1.** An interval
  straddling 1 means the data do not establish suppression, which is the honest answer.

Points that hit `--max-shots` before reaching `--max-errors` are listed as
`censored_points` — a measured fact about what happened, not a heuristic warning about
what might.

### `qecgen score`

Score a **supplied** physical Pauli correction by its logical effect — the brief's
"apply the correction, check the logical qubit matches", done exactly.

```bash
qecgen score data/d5_p005.h5 --correction proposed.npz
```

`proposed.npz` holds `correction_x` and `correction_z`, each little-endian bit-packed
`uint8 (shots, ceil(n_data_qubits/8))` over the data qubits. Pass `--unpacked` to supply
bool `(shots, n_data_qubits)` arrays instead and let `qecgen` pack them, so a third party
never calls `np.packbits` themselves and cannot take NumPy's big-endian default.

Measured on a d=3, p=0.01, 20k-shot dataset:

| correction | logical error rate |
|---|---|
| identity (all zeros) | `0.19245 [0.18701, 0.19798] (3849/20000)` — exactly the raw uncorrected rate |
| oracle (flip iff the observable flipped) | `0.00000 [0.00000, 0.00018] (0/20000)` |
| MWPM prediction lifted to a Pauli | `0.05525 [0.05212, 0.05851] (1105/20000)` |

**This is not Contract C.** Contract C — inferring which physical fault occurred from a
syndrome — remains refused, because it inverts a deliberately many-to-one map. Scoring a
correction someone else proposes is the forward direction of the same arrow: a
deterministic, single-valued function with no tie-breaking anywhere. See
[`DATA_CONTRACT.md`](DATA_CONTRACT.md).

The logical operators are rebuilt from a **noiseless** circuit generated from the
dataset's own decoder-visible manifest parameters, which is sound because the correction
schema is a property of the code rather than of the noise — asserted as a test across
noise model, `p` and the `xz_bias` rewrite. So scoring a `frozen_prior` test file never
reads that file's own error model.

### `qecgen validate` / `inspect` / `formats`

```bash
qecgen validate data/d5_p005.h5          # fast structural checks
qecgen validate data/d5_p005.h5 --qa     # plus slow statistical checks
qecgen inspect  data/d5_p005.h5          # manifest and environment table
qecgen formats                           # registered exporters
```

### `qecgen ui`

The three dataset-producing commands in a browser, for when you would rather not remember
which flags interact.

```bash
cd frontend && npm ci && npm run build   # once, and after any frontend change
qecgen ui --data-root data               # http://127.0.0.1:8765
```

Three pages: a run form with a live cost preview, a run list with live progress and
cancellation, and a dataset browser with manifests and validation. It covers `generate`,
`multi-env` and `drift`; `sweep`, `score` and `inspect` stay terminal-only.

Three things about it are deliberate rather than incidental.

**It is loopback-only, and that is not configurable.** The API writes files and starts
processes for whoever can reach it, with no authentication. A non-loopback `--host` is
refused by name rather than quietly accepted; use an SSH tunnel if you need it elsewhere.
Every path that arrives from the browser is resolved and required to sit under
`--data-root` before it is used.

**Each run is a subprocess, not a thread.** Stim's sampler holds the GIL. Measured here
with the job on a thread and an asyncio loop beside it: median event-loop lag 102 ms at
d=9 with a 100k chunk, worst case 1033 ms at a 1M chunk. The same work in a subprocess
left the loop at 13 ms, which is Windows timer granularity. A server that freezes for a
second at a time cannot serve the cancel request, which is the one request that matters
during a long run.

**Nothing is written to the path you will read until the run finishes.** Output is staged
in a sibling directory and moved into place with `os.replace`, so a cancelled or crashed
run leaves no file rather than a plausible-looking broken one. The browser also never
lists a dataset by reading all of it: manifests come from the cheap place in each format,
and a file that cannot be read is listed with the reason attached rather than hidden.

The preview panel is the one thing the terminal cannot do. It builds the circuit and DEM
without sampling, so before you commit it can tell you the detector and mechanism counts,
the packed row width, the file size, and whether the run will stream or hold every shot
in memory.

Same inputs produce the same bytes as the CLI — verified by matching `content_hash`
between a `qecgen generate` run and the same run submitted through the browser.

---

## HDF5 schema

The primary format, and the only streaming-capable one.

```
/detectors        uint8  (shots, ceil(n_detectors/8))    little-endian bit-packed
/observables      uint8  (shots, ceil(n_observables/8))
/environment_ids  int32  (shots,)                        present for multi-environment
/mechanisms       uint8  (shots, ceil(n_mechanisms/8))   present with --emit-mechanisms
/dem/                                                    present per --structure
    H_indices, H_indptr, H_shape       CSC of (n_detectors, n_mechanisms)
    L_indices, L_indptr, L_shape       CSC of (n_observables, n_mechanisms)
    priors               float64 (n_mechanisms,)
    detector_coords      float64 (n_detectors, coord_dim)   (x, y, t)
    component_parent     int32   (n_components,)            parent_mechanism_id
    component_index      int32   (n_components,)
    component_det_flat   int32   ragged detector ids, concatenated
    component_det_offset int64   (n_components + 1,)
    component_obs_flat   int32
    component_obs_offset int64   (n_components + 1,)
/provenance/                                             present only at --structure full
    environments         JSON attribute: per-environment circuit and DEM text
    warning              "PROVENANCE ONLY - a decoder must not read this group"
```

Root attributes: `manifest` (full JSON), plus `bit_order`, `contract`, `n_detectors`,
`n_observables`, `shots`, `distance`, `rounds`, `chunk_size`, `drift_condition`,
`qecgen_version` mirrored as scalars so `h5dump -A` is readable without parsing JSON.

CSC `data` values are not stored: `H` and `L` are binary incidence matrices, so every
stored entry is 1 by construction.

`--structure {none,coords,dem,full}` controls how much of this is written, so a decoder
can be run both with and without the graph structure PyMatching gets for free.

### One column per mechanism, not per component

`H` has **one column per independent `error(...)` instruction**. Stim's decomposed DEM
writes correlated mechanisms as graphlike components joined by `^` separators; those
components share a single probability and belong to one mechanism.

Measured on d=3, r=3, uniform circuit-level p=0.01:

| Quantity | Value |
|---|---|
| `error(...)` instructions = `n_mechanisms` | **286** |
| Total components | 556 |
| Mechanisms containing a `^` separator | 208 |

Emitting one column per component would give **556 columns instead of 286**, each
carrying a duplicated copy of the parent probability — a different noise model from the
one Stim simulated. Components are preserved separately, keyed by `parent_mechanism_id`.

### Column weight is checked per component, not per mechanism

This corrects a natural but wrong expectation. Measured mechanism detector-weights:

```
weight 1 -> 24    weight 2 -> 115    weight 3 -> 98    weight 4 -> 49
```

Only 139 of 286 mechanisms touch one or two detectors. That is **not** a parsing bug: a
decomposed mechanism with two components legitimately touches up to four detectors. The
graphlike guarantee applies to components (each ≤ 2 detectors), which is what
`validate.py` asserts. Mechanism weight is *reported*, never asserted — asserting it would
produce a validator that fails on correct data.

---

## JSONL schema

The JSON-facing format. Line-delimited, so a consumer can read the manifest and the full
DEM structure without touching a single shot.

```
line 1        {"__manifest__": { … }}
line 2        {"__structure__": { … }}          present iff structure_level != none
lines 3..N    {"shot": 0, "detectors": "0100…", "observables": "1",
               "environment_id": 3, "mechanisms": "0010…"}
```

The structure object mirrors the HDF5 group key for key — `H_indices`, `H_indptr`,
`H_shape`, the same for `L`, `priors`, `detector_coords`, plus `level`, `n_detectors`,
`n_observables`, `n_mechanisms`, `coord_dim`. CSC `data` values are omitted for the same
reason HDF5 omits them. `components` are **nested objects**, not the flattened
value/offset pairs HDF5 and NPZ use: those exist only because HDF5 has no ragged type, and
the offset form carries a specific silent failure — an off-by-one re-segments every
component after it while still round-tripping, because the read side is symmetrically
wrong.

**Bit strings: `s[0]` is index 0.** The leftmost character is the *lowest* index, matching
the little-endian packing everywhere else. **Do not use `int(s, 2)`** — that treats the
leftmost character as most-significant and reverses the index across the whole file. This
is the string-form counterpart of the `bitorder` trap above, and a round-trip test cannot
catch it: reversing on write and again on read is perfectly self-consistent. It is
asserted against the packed bytes instead.

Floats round-trip **bit-exactly**. Python's JSON encoder uses `repr`, which is
shortest-round-trip for IEEE754 doubles; verified over 10k random float64 plus denormals
and 1e±300. `allow_nan=False` on write and a rejecting `parse_constant` on read, so a bare
`NaN`/`Infinity` token — not valid JSON, and rejected by Go and serde_json — can neither be
written nor silently accepted. Ragged detector coordinates, which `parse_dem` pads with
NaN, are encoded as JSON `null`.

Structure-line size depends on distance, never on shot count: 61 KiB at d=3, ~434 KiB at
d=5, ~1.5 MB at d=7.

> **Reader note.** Go's `bufio.Scanner` defaults to a 64 KB token cap and fails on the
> structure line of *every* d≥3 file. One call fixes it:
> `scanner.Buffer(make([]byte, 0, 64<<10), 64<<20)`. Python, `jq`, Node and Java need
> nothing.

JSONL is a JSON interface. **It is not the Nexus interface** — no fixture from the Nexus
team has been run against it.

---

## Oracle-calibrated vs frozen-prior

The distinction that decides whether the drift study means anything.

- **`ORACLE_CALIBRATED`** — structure exported with a test file is derived from **that
  file's own environment**.
- **`FROZEN_PRIOR`** — structure exported with a test file is derived from the **nominated
  training environment**.

If a test set at p = 0.010 ships with `--structure dem` where the DEM was built at
p = 0.010, the decoder has been handed the true test-time noise distribution. Any claim of
generalisation to unseen noise is then unsupported, because nothing was unseen.

Both are legitimate experiments. `ORACLE_CALIBRATED` measures a ceiling;
`FROZEN_PRIOR` measures generalisation. The condition is recorded in the manifest as
`drift_condition`, alongside `structure_source_environment_id` so the choice is auditable
from the file rather than taken on trust. It is never inferred from whichever DEM happened
to be in scope.

Verifiable directly:

```python
train = get_exporter("hdf5").read(Path("data/drift/train.h5"))
test = get_exporter("hdf5").read(Path("data/drift/test_0.014.h5"))
np.allclose(test.structure.priors, train.structure.priors)  # True under frozen_prior
test.meta.structure_dem_sha == train.meta.structure_dem_sha  # True: same source DEM
```

### The manifest is decoder-visible; provenance is not

Freezing the *numbers* is not sufficient. A test file also has to avoid describing its own
error model in words. The manifest therefore **never** contains circuit or DEM text:

| Location | Contents | Decoder may read |
|---|---|---|
| root `manifest` attribute | all parameters: distance, rounds, channels, axis, seed, hashes | **yes** |
| `dem/` group | H, L, priors, components, coordinates, from whichever environment the condition nominates | **yes** |
| `provenance/` group | per-environment circuit and DEM **text**, written only at `--structure full` | **no** |

Under `FROZEN_PRIOR` the `provenance/` group contains the *test* environment's own DEM,
which is exactly what the condition withholds. It is kept for auditing but separated
physically, so reading the manifest cannot expose it. `structure_dem_sha` lets a reviewer
confirm which environment supplied the structure without reading any text.

The parameters in the manifest remain sufficient to regenerate every environment, so
dropping the text costs no reproducibility.

### Drift axes

`--drift-axis` is not limited to `p`, because sweeping the uniform rate alone mostly tests
interpolation within one synthetic family. Held-out noise *families* are stronger evidence
of invariance than a held-out rate.

| Axis | Meaning | Unbiased point |
|---|---|---|
| `p` | the uniform rate itself | — |
| `measurement_ratio` | measurement error scaled independently of gate error | 1.0 |
| `xz_bias` | `pz / (px + py)` on single-qubit data noise | 0.5 |

Adding an axis is one function plus one entry in `AXIS_BUILDERS`.

**`xz_bias` scope, stated precisely.** `stim.Circuit.generated` exposes only four
*symmetric* scalar probabilities and cannot express unequal X/Z rates at all. The bias axis
therefore rewrites `DEPOLARIZE1(p)` into `PAULI_CHANNEL_1(px, py, pz)` on Stim's generated
circuit, preserving the total single-qubit error probability.

It rewrites **every** `DEPOLARIZE1`, which in a generated surface code means both
`before_round_data_depolarization` (data noise) **and** `after_clifford_depolarization`
applied after single-qubit Cliffords. An earlier version of this README claimed only data
noise was affected; that was wrong, and a gate-noise-only circuit demonstrates it. The
manifest records the true scope as `bias_scope`.

Two-qubit `DEPOLARIZE2` noise **is** left symmetric: biasing it correctly needs the
15-parameter `PAULI_CHANNEL_2` and a convention for correlated two-qubit bias that nobody
has specified. So a dataset on this axis carries biased single-qubit noise (data and gate)
and unbiased two-qubit gate noise.

At `eta = 0.5` the rewrite gives `px = py = pz = p/3`, the depolarising point. The
resulting DEM is not bit-identical to the one built from `DEPOLARIZE1` — it agrees on
mechanism support exactly and to first order in p, with an O(p²) residual from Stim's
channel-composition arithmetic (`diff/p² ≈ 0.356`, constant across p = 1e-2, 1e-3, 1e-4).
An isolated single-channel comparison agrees to 1e-16, confirming the formula is exact and
only the composition differs.

---

## Contract B

`--emit-mechanisms` records which DEM mechanisms fired per shot.

These are **abstract mechanisms in the decomposed noise model, not gate-level physical
Pauli faults.** The mechanism index is an artifact of Stim's DEM construction order and is
not portable across noise models or distances. A decoder trained on these targets is
learning to invert Stim's DEM construction.

With `--emit-mechanisms`, **all three arrays come from the DEM sampler**
(`dem.compile_sampler(...).sample(n, bit_packed=True, return_errors=True)`), not from the
circuit's detector sampler. Sampling the circuit and the DEM separately would give two
independent RNG streams, so the mechanism labels would not explain the detection events
stored beside them. Sampling the DEM reproduces the same detector/observable distribution,
so Contract A targets remain valid.

Pooling Contract B labels across environments is rejected when their mechanism counts
differ, since column *k* would then mean a different mechanism in different environments.

---

## Reproducibility

Determinism is **content-based, not file-based**. Manifests contain timestamps and git
commits, so byte-identical files are not a meaningful test. Every manifest carries a
`content_hash` over array dtype, shape and bytes only, alongside
`content_hash_algorithm: "blake2b-256"`.

The algorithm is named because it is externally checkable. This field was previously called
`content_sha256` while computing BLAKE2b, so any independent verification attempted with
the advertised algorithm would have failed.

**Chunk size is part of the reproducibility contract.** Stim guarantees seeded
reproducibility only under matching version, machine characteristics and call structure.
Changing the chunk size changes the sequence of `sample()` calls against one seeded
sampler, and therefore changes the sample stream. `chunk_size` is recorded in every
manifest; reproducing a file requires matching it.

Seeds are threaded explicitly everywhere. No global RNG state is touched. Per-environment
and shuffle seeds are derived from the master seed via `numpy.random.SeedSequence`
spawning, so the whole run is reproducible from one integer.

Memory is bounded where it matters. Sampling is chunked (default 100,000 shots), and
`qecgen generate` routes through the streaming HDF5 writer whenever the format is HDF5 and
`--shots` exceeds one chunk, so peak memory stays flat regardless of dataset size.

Two honest limits. Non-HDF5 formats have no incremental writer and still materialise, so
keep NPZ, Parquet and JSONL runs modest. `multi-env` needs all shots resident to apply the
seeded permutation, so its peak memory scales with total shot count; that is the price of a
shuffle that provably preserves per-shot correspondence.

---

## Validation vs QA

Deliberately separate modules. Mixing deterministic structural checks with statistical ones
produces a default suite that fails for sampling reasons, and a suite that fails
intermittently is a suite that gets ignored.

**`validate.py`** — fast, deterministic, runs by default. Shapes and dtypes against the
manifest, packed widths equal `ceil(n/8)`, bit order recorded, all-zero detectors and
observables at p = 0, content hash, DEM shape agreement, per-component weight bound,
environment id correspondence.

**`qa.py`** — slow, opt-in via `--qa` or `pytest -m slow`. Every comparison is between
**Clopper-Pearson intervals**, never point estimates, with adaptive shot counts targeting a
minimum number of observed logical failures.

The estimated threshold crossing is reported as a **result**, never asserted against a
hardcoded value. The commonly quoted 0.5–1% figure depends on the channel convention,
rounds, basis and decoder, so it is a sanity range to print, not a test to fail on.

The validation oracle is built with `pymatching.Matching.from_detector_error_model(dem)` on
the original decomposed DEM object — never reconstructed from our own `H`, which would only
validate the parser against itself.

---

## Adding an exporter

Once the Nexus format is known, adding it is one file.

1. Create `qecgen/exporters/nexus.py` with a class satisfying the `Exporter` protocol
   (`qecgen/exporters/base.py`): properties `format_name`, `extension`, `streaming`,
   `structure_round_trip`, and methods `write(dataset, path, structure_level)` and
   `read(path)`. `structure_round_trip` is live, not decorative — the parametrised tests
   and `qecgen formats` both read it, so declaring it `True` enrols the format in the
   exact-equality structure round-trip contract.
2. Register it in `qecgen/exporters/__init__.py` by adding one entry to `EXPORTERS`.
3. The parametrised round-trip tests in `tests/test_exporters.py` pick it up
   automatically from the registry — no test changes needed.

The round-trip contract is: `read(write(d))` reproduces every array and every manifest
field exactly. Structure round-trip is required only for formats that claim it.

```python
class NexusExporter:
    @property
    def format_name(self) -> str:
        return "nexus"

    @property
    def extension(self) -> str:
        return ".nxs"

    @property
    def streaming(self) -> bool:
        return False

    @property
    def structure_round_trip(self) -> bool:
        return False

    def write(self, dataset, path, structure_level=StructureLevel.NONE): ...
    def read(self, path): ...
```

---

## Comparison with `decoder-bench`

`decoder-bench` (Maurya, Viszlai, Raveendran, Das, Tannu, IISWC 2025) is a published
Stim-based framework producing HDF5 QEC traces for this kind of benchmarking. `qecgen`
takes **no dependency on it**; the comparison exists because alignment with a published
convention is cheap insurance if the client turns out to expect something shaped like it.

**Scope of this comparison:** the repository's landing page documents its HDF5 files at the
level of *syndromes*, *observables* and *metadata* groups. Field-level schema is in its
source (`sampler.py`, `eval.py`, `decoders.py`) and the associated Zenodo dataset, which
were **not read directly**. The agreements below are therefore at the conceptual level, and
this section should be re-checked against the actual files before being relied on.

| Aspect | decoder-bench | qecgen | Why |
|---|---|---|---|
| Container | HDF5 | HDF5 | Agrees. Chosen for the same reasons: chunked, compressible, self-describing. |
| Syndrome data | binary syndrome measurements | `detectors`, bit-packed uint8 | Agrees conceptually. We store **detection events**, and pack them, to keep files ~8x smaller and match Stim/PyMatching's native format exactly. |
| Observables | logical measurement outcomes | `observables`, separate array | Agrees. `separate_observables=True` on every sample call; observables are never mixed into the detector array. |
| Metadata | code parameters, noise model, simulation settings | JSON `manifest` attribute | Agrees in intent. We additionally require the manifest be **sufficient to regenerate the file**, including library versions, git commit, seed and chunk size. |
| Noise models | code-capacity, phenomenological, circuit-level | same three | Agrees. We rename the third `stim_uniform_circuit_level` because setting all four channels to one `p` is a convention, not a definition. |
| Per-environment structure | not described | `EnvironmentSpec` list, no top-level `p` | **Diverges.** Multi-environment datasets are a first-class requirement here; promoting `p`/`circuit`/`dem` to the dataset level breaks the moment there is more than one environment. |
| DEM export | not described | `dem/` group, one column per `error(...)` | **Diverges.** Needed for the frozen-prior condition and to give a learning decoder the same structure PyMatching receives. |
| Drift condition | not described | `drift_condition` + `structure_source_environment_id` | **Diverges.** Specific to this contract's generalisation study. |

---

## API discrepancies and verification notes

Signatures were verified by introspecting the installed libraries rather than assumed from
documentation.

- **`CompiledDetectorSampler.sample`** — installed signature is
  `sample(shots, *, prepend_observables=False, append_observables=False,
  separate_observables=False, bit_packed=False, dets_out=None, obs_out=None)`. Matches the
  assumed usage.
- **`CompiledDemSampler.sample`** — installed signature is
  `sample(shots, *, bit_packed=False, return_errors=False, recorded_errors_to_replay=None)`.
  `return_errors` exists as assumed. **It always returns a 3-tuple**; with
  `return_errors=False` the third element is `None` rather than the tuple being shorter.
- **`stim.Circuit.generated`** — the four noise parameters are the *only* noise controls.
  There is no parameter for asymmetric X/Z rates, which is why `xz_bias` rewrites the
  circuit.
- **`Matching.decode_batch`** — supports `bit_packed_shots` and `bit_packed_predictions`.
  The packed and unpacked paths were checked to give identical predictions.
- **`dem.flattened()`** — contains no `DemRepeatBlock`; instruction types observed are
  `error` and `detector` only. The parser still checks `instruction.type == "error"`
  rather than assuming.
- **Duplicate detector targets** — a detector repeated within one `error(...)` would cancel
  mod 2. Audited across all 286 mechanisms of the d=3 model: zero occurrences. The parser
  uses XOR accumulation anyway, because it is correct regardless and the audit covers one
  configuration rather than all of them.
- **No PEP 561 markers** — `stim`, `pymatching`, `sinter`, `h5py`, `pyarrow` and `scipy` all
  ship without a `py.typed` marker (`stim` ships an `__init__.pyi` but no marker, so mypy
  will not consult it). `pyproject.toml` carries narrowly scoped
  `ignore_missing_imports` overrides for exactly these, listed explicitly rather than as a
  blanket ignore. `matplotlib` ships `py.typed` (verified on the pinned 3.10.8) and needs
  no override; an entry for it sat in the list anyway until a review caught it — dead
  config that made the list read as "whatever mypy complained about once".

---

## Sinter timing is throughput, not latency

`sinter.collect` parallelises across workers to maximise shots per second. Its timing is a
**throughput** measure and says nothing about per-shot decoder latency. Latency
benchmarking belongs in the separate benchmark repository and is out of scope here.

---

## Out of scope

Not built, and not stubbed in a way that implies they exist:

- **Any `nexus` client, import or exporter.** The Nexus input format is still unknown.
- **A latency harness.** No per-shot decoder timing. `sinter`'s timing is throughput, and
  latency benchmarking belongs in the separate benchmark repository.
- **Decoder *implementations* or adapters.** `qecgen` implements no decoder. `--decoder`
  resolves names and hands them to `sinter`, which owns the dispatch; `qecgen/decoders.py`
  exists only to answer "is this name valid and is its backend installed" before a
  collection starts. There is deliberately no `Decoder` protocol — `sinter.Decoder` already
  is one, and `sinter.collect(custom_decoders=...)` is the extension point.
- **A custom surface code builder, or a custom MWPM implementation.**
- **`pickle` as an output format.**
- **Codes other than the surface code.** Stim supports `color_code:memory_xyz` and
  `repetition_code:memory` out of the box, so adding them is cheap, but they are not here.
- **Real hardware syndrome data.** No Zenodo/Google/IBM ingest; synthetic drift only.
- **`decoder-bench` compatibility.** The comparison below is conceptual and its caveat
  stands.
- **On-disk export of the correction schema.** `qecgen score` derives it on demand from
  manifest parameters, which is sound (the schema is noise-independent), but it means the
  schema is not auditable from the file itself. A `/correction/` HDF5 group is the
  outstanding piece.
- **Contract C.** Physical Pauli fault *labels* as a training target remain refused. See
  `DATA_CONTRACT.md`; note that *scoring* a supplied correction is a different operation
  and is implemented.
- **A multi-user or network-reachable web UI.** `qecgen ui` binds loopback and refuses
  anything else. There is no authentication, no per-user isolation and no disk quota,
  because it is a local tool for the person at the keyboard.
- **`sweep`, `score` and `inspect` in the browser.** The UI covers the three commands that
  produce datasets. The rest stay terminal-only.
