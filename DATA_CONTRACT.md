# DATA CONTRACT

**Status:** Phase 0, resolved. Governs every file `qecgen` writes.

This document exists because "decoder training data" is not one thing. There are three
mutually incompatible things a quantum error correction decoder can be asked to output,
they require different targets, and conflating them is the single most likely way this
project produces confident, wrong results.

Every file produced by `qecgen` is a **syndrome-to-logical-frame dataset**. It must never
be described, in a paper, a README, a filename or a conversation with the client, as a
physical-error-labelled dataset. It does not contain physical error labels. See Contract C.

---

## The one-line summary

> `qecgen` maps **per-shot detection events** to **per-shot logical observable flips**.
> Optionally it also records **which abstract DEM mechanisms fired**. It never records
> physical Pauli faults on data qubits.

---

## Contract A — logical frame prediction (APPROVED DEFAULT, IMPLEMENTED)

| | |
|---|---|
| **Input** | Per-shot detection events, `(shots, n_detectors)` |
| **Target** | Per-shot logical observable flips, `(shots, n_observables)` |
| **Produced by** | `circuit.compile_detector_sampler(seed=...).sample(n, separate_observables=True, bit_packed=True)` |
| **Predicted by** | PyMatching, and every other MWPM/UF/BP decoder |
| **Measured by** | Every published surface code threshold plot |

This is the contract the benchmark should use. It is what PyMatching predicts, so the
validation oracle and the decoder under test are answering the *same question*, which is
the only way the comparison means anything.

A decoder succeeds on a shot when its predicted observable flip equals the true observable
flip. Logical error rate is the fraction of shots where they differ. This is well-defined,
standard, and directly comparable to published numbers.

**Verified behaviour** (stim 1.16.0, `surface_code:rotated_memory_z`, d=3, r=3):

```
sample(shots, separate_observables=True, bit_packed=True)
  -> (uint8 (shots, 3), uint8 (shots, 1))       # d=3: 24 detectors, 1 observable
```

Packed widths are `ceil(24/8) = 3` and `ceil(1/8) = 1`. The packed width does **not**
determine the true width — 3 bytes could mean anywhere from 17 to 24 detectors — so
`n_detectors` and `n_observables` are stored explicitly in every manifest.

---

## Contract B — DEM mechanism prediction (OPTIONAL, BEHIND A FLAG)

| | |
|---|---|
| **Input** | Per-shot detection events |
| **Target** | Which detector-error-model mechanisms fired, `(shots, n_mechanisms)` |
| **Produced by** | `dem.compile_sampler(seed=...).sample(n, bit_packed=True, return_errors=True)` |
| **Enabled by** | `--emit-mechanisms` |

**Read this before using Contract B.**

DEM mechanisms are **abstract error mechanisms in the decomposed noise model**. They are
**not** the original gate-level physical Pauli faults. Specifically:

- Stim's DEM is derived *from* the circuit by propagating faults to detectors. Many
  physically distinct gate-level faults that produce identical detector and observable
  signatures are collapsed into a single mechanism.
- With `decompose_errors=True`, mechanisms are further rewritten into graphlike components
  joined by `^` separators, chosen to suit matching decoders. The decomposition is a
  decoder-facing convenience, not a physical claim.
- The mechanism index is an artifact of Stim's DEM construction order. It has no meaning
  outside the exact circuit and Stim version that produced it, and it is **not** portable
  across noise models or distances.

Consequently a decoder trained to predict Contract B targets is learning to invert Stim's
DEM construction, not to identify physical faults. That may still be a legitimate research
target, but it must be named accurately.

**Verified behaviour** (same circuit):

```
dem.compile_sampler(seed=...).sample(n, bit_packed=True, return_errors=True)
  -> (uint8 (n, 3), uint8 (n, 1), uint8 (n, 36))   # 286 mechanisms -> ceil(286/8) = 36
```

The call returns a **3-tuple in both cases**; with `return_errors=False` the third element
is `None` rather than the tuple being shorter.

---

## Contract C — physical Pauli correction (NOT IMPLEMENTED, AND NOT IMPLEMENTABLE AS SPECIFIED)

| | |
|---|---|
| **Input** | Per-shot detection events |
| **Target** | Per-data-qubit X/Z corrections |
| **Status** | **Will not be implemented** |

Two independent reasons, either of which is sufficient.

**1. It is underdetermined by quantum mechanics, not merely unspecified.**

Quantum error correction codes are *degenerate*. Distinct physical fault configurations
produce **identical syndromes** and identical logical effects. They are not
distinguishable, even in principle, from the measurement record — that indistinguishability
is precisely what makes the code work. Asking a decoder to recover "the" physical fault
from a syndrome is asking it to invert a deliberately many-to-one map.

There is no ground truth to train against, because for a given syndrome there is no unique
correct physical answer. A dataset claiming to provide one has silently picked a
representative from an equivalence class and called it truth. Any accuracy number computed
against such labels measures agreement with an arbitrary tie-breaking convention.

**2. The pieces needed to even define it do not exist.**

Nobody has specified a physical correction schema (qubit indexing, X/Z/Y encoding, frame
convention, time slice) or a mapping from a proposed physical correction to logical
success. Without the second, "correct" is undefined even if the first were fixed.

**If the client asks for Contract C**, the response is not to build it. It is to establish
which of these they actually want:
- logical frame prediction (Contract A — they almost certainly want this), or
- a *coset* / equivalence-class target, which is well-defined but is a different and
  larger design problem, or
- DEM mechanism identification (Contract B), correctly named, or
- **scoring a correction they supply**, which is well-defined and is implemented — see
  Contract C-scoring below.

Both pieces named above as missing now exist. Supplying them does **not** make Contract C
implementable. They define how to *evaluate* a correction someone else proposes; they
produce no training target, and the degeneracy argument in (1) is untouched by them.

---

## Contract C-scoring — evaluating a supplied physical correction (IMPLEMENTED)

| | |
|---|---|
| **Input** | A proposed per-data-qubit Pauli correction, supplied by a third party |
| **Output** | Per-shot logical success/failure, and a rate with a Clopper-Pearson interval |
| **Produced by** | `qecgen.correction`, `qecgen score` |
| **Is it a dataset target?** | **No.** No file gains a label column. `contract` stays `logical_frame`. |

**This is not a reversal of Contract C. It is the other direction of the same arrow.**

Contract C asks: *given a syndrome, which physical fault occurred?* That inverts a
deliberately many-to-one map, has no unique answer, and remains refused. Nothing here
weakens that.

Contract C-scoring asks: *given a correction, what was its logical effect?* That is a
**forward evaluation of a deterministic function**, single-valued for every input, with no
tie-breaking anywhere. The degeneracy that makes the inverse ill-posed is exactly what
makes the forward map well-posed: every member of an equivalence class has the same
logical effect, so the answer does not depend on which representative is held.

The client brief's *"Nexus outputs a proposed correction. Apply and Measure. If the final
state matches the pristine state, the decoding was a success"* is **this** operation, not
Contract C. It needs no ground-truth fault label, because the correction is an **input**.

### The rule

For observable `k` with symplectic masks `(L_x[k], L_z[k])` over data qubits, and a
proposed correction `(C_x, C_z)`:

```
predicted_flip[k] = parity(popcount(C_x & L_z[k]) + popcount(C_z & L_x[k]))
success(shot)     = all(predicted_flip[k] == true_observable_flip[k] for every k)
```

This is anticommutation of two Pauli operators. It is exact for a memory experiment,
because the logical observable is fully determined by the final data measurements. A `Y`
correction sets both `C_x` and `C_z` and needs no special case.

### The schema — the pieces Contract C named as missing

- **Qubit indexing.** Data qubits are those measured by the final **non-resetting**
  measurement layer (`M`/`MX`/`MY`), indexed `0 .. n_data-1` in ascending stim qubit order.
  The map to stim qubit index, coordinates, per-qubit measurement basis and
  measurement-record position is exported explicitly, never left to convention. Rotated
  d=3 allocates 26 qubit slots for 9 data qubits, so compact indexing also makes a
  correction on an ancilla *unrepresentable* rather than silently ignored.
- **Encoding.** Two parallel little-endian bit-packed `uint8` arrays, `correction_x` and
  `correction_z`, each `(shots, ceil(n_data/8))`. `Y = x & z`. Chosen over a
  2-bit-per-qubit layout because it *is* the representation the scoring rule consumes, and
  it introduces no new bit-layout convention.
- **Frame and time slice.** `final_data_layer`: immediately before the final data
  measurement layer. **Out of scope:** mid-circuit Pauli frames (they propagate through
  the remaining CX layers, so their final-layer residue is not the operator written),
  measurement-result corrections, non-Pauli corrections, and corrections on ancillas.

### Why this is safe under `FROZEN_PRIOR`

The schema and the logical operators are properties of the **code**, not of the noise.
They are determined by `distance`, `rounds`, `basis` and `rotated` — all already
decoder-visible manifest fields — so a reader could reconstruct them anyway. There is no
probability-valued field anywhere in the schema.

Verified invariant across noise model, `p ∈ {0, 1e-4, 0.01, 0.05}` and the `xz_bias`
rewrite, and asserted as a test rather than claimed
(`test_schema_digest_is_stable_and_noise_independent`). Consequently `qecgen score`
rebuilds the operators from a **noiseless** circuit generated from manifest parameters, so
scoring a frozen-prior test file never reads that file's own error model.

**One caveat, stated rather than glossed:** if a future drift axis varied the *code*
rather than the noise, train and test schemas would differ and the schema would then carry
test-side information. Every axis today (`p`, `xz_bias`, `measurement_ratio`) varies noise
only, and the digest test above fails loudly if that changes.

### Verified behaviour

```
rotated_memory_z   d=3 -> Z-type, support (stim ids) [1, 3, 5],   weight 3
rotated_memory_x   d=3 -> X-type, support (stim ids) [1, 8, 15],  weight 3
unrotated_memory_z d=3 -> Z-type, support (stim ids) [0, 2, 4],   weight 3
rotated_memory_z   d=5 -> Z-type, support (stim ids) [1,3,5,7,9], weight 5
```

Cross-checked three ways: `stim.Circuit.has_flow` with `included_observables` **and a
negative control** that must fail; literal `X_ERROR(1)`/`Z_ERROR(1)` injection before the
final layer, exhaustively over every data qubit × {X, Z, Y} in four layouts; and exact
integer agreement with PyMatching's Contract A failure count when its logical prediction is
lifted onto a support qubit.

**A trap worth recording.** A *deterministic* `X` gate injected before the final
measurement does **not** move the observable `compile_detector_sampler` reports — the
sampler defines observables relative to a reference sample of the same circuit and folds
the gate into the reference. Measured: `X 1` gives observable 0 and **zero** detection
events; `X_ERROR(1) 1` gives observable 1 and one detection event. An oracle built on
deterministic gates passes every test while measuring nothing.

### What PyMatching cannot do here

`Matching.decode_batch` predicts observable flips; its `num_fault_ids` equals
`num_observables`. `decode_to_edges_array` returns matching-graph edges, which are
spacetime DEM mechanisms — many of them measurement-flip mechanisms with **no data-qubit
support at all**. There is no route from a matching edge to a data-qubit Pauli without the
refused inversion. `stim.Circuit.explain_detector_error_model_errors` does map mechanisms
back to representative circuit faults, but turning those into a final-frame correction
requires propagating mid-circuit faults forward, which is not implemented.

---

## Verified facts underpinning these contracts

Established by direct introspection of the installed libraries, not assumed from
documentation. Environment: Python 3.13.6, win_amd64, stim 1.16.0, pymatching 2.4.0,
sinter 1.16.0.

### Bit ordering is little-endian, proven empirically

```
np.unpackbits(packed, axis=1, count=n_detectors, bitorder="little") == unpacked sample  -> True
np.unpackbits(packed, axis=1, count=n_detectors, bitorder="big")    == unpacked sample  -> False
```

NumPy's `packbits`/`unpackbits` default is `bitorder='big'`, which is the **opposite** of
what Stim and PyMatching use. Sinter's own decoder contract states it explicitly: *"All
data taken and returned must be bit packed with `bitorder='little'`."* Every manifest
records `bit_order="little"`. Any code path touching `packbits`/`unpackbits` passes
`bitorder="little"` explicitly.

### One column per `error(...)`, not one per component

For d=3, r=3, uniform circuit-level p=0.01:

| Quantity | Value |
|---|---|
| `error(...)` instructions = **`n_mechanisms`** | **286** |
| Total graphlike components across all mechanisms | 556 |
| Mechanisms containing at least one `^` separator | 208 |
| Components per mechanism | 1→78, 2→150, 3→54, 4→4 |

Emitting one column per component would produce **556 columns instead of 286**, a 94%
inflation, each carrying a duplicated copy of the parent probability. That is a different
noise model from the one Stim simulated. Components are therefore recorded in a parallel
structure keyed by `parent_mechanism_id`.

### Column weight sanity must be checked per component, not per mechanism

This corrects an assumption in the original brief. The brief expects "most mechanisms
should touch one or two detectors". Measured detector-weight histogram across whole
mechanisms:

```
weight 1 -> 24    weight 2 -> 115    weight 3 -> 98    weight 4 -> 49
```

Only 139 of 286 mechanisms touch one or two detectors. This is **not** a parsing bug. A
decomposed mechanism with 2 components legitimately touches up to 4 detectors. The
graphlike guarantee applies to **components** (each ≤ 2 detectors), not to mechanisms.

`validate.py` therefore asserts the weight bound on components and merely *reports* the
mechanism-weight distribution. Checking the bound on mechanisms would produce a validator
that fails on correct data.

### `dem.flattened()` contains no repeat blocks

Instruction types observed: `error` (286) and `detector` (24), all of concrete type
`DemInstruction`. No `DemRepeatBlock` survives flattening, so the parser does not need to
recurse. It still checks `instruction.type == "error"` rather than assuming.

### Duplicate-detector cancellation did not occur here

A detector appearing twice within one `error(...)` instruction would cancel mod 2.
Audited across all 286 mechanisms: **0 duplicates within a component, 0 across
components**. The parser still uses XOR/symmetric-difference accumulation, because it is
correct regardless and the audit covers one configuration rather than all of them — but
this trap is, on present evidence, latent rather than active.

### Zero noise gives zero signal

At p = 0, across 256 shots: no detector fires and no observable flips. Used as a
validation invariant.

---

## OPEN QUESTIONS

Everything below cannot be resolved without the client. Each carries the current working
assumption, which is what the code does today.

| # | Question | Working assumption |
|---|---|---|
| 1 | **What is the Nexus input format?** | Unknown. No Nexus exporter exists. The canonical model plus `Exporter` registry means adding one is a single new file. **The README does not and must not claim Nexus compatibility until a Nexus exporter passes a fixture supplied by the Nexus team.** |
| 2 | Does Nexus want detection events bit-packed or as bool? | Both are exportable. HDF5 stores packed uint8 with `bit_order="little"` recorded; unpacking is one documented call. |
| 3 | Does Nexus consume graph structure (coords), the full DEM, or neither? | `--structure {none,coords,dem,full}` makes this a run-time choice so the decoder can be run with and without the structure PyMatching gets for free. |
| 4 | Is the target Contract A or Contract B? | Contract A. Contract B is opt-in via `--emit-mechanisms` and clearly labelled. |
| 5 | **Can `xz_bias` drift be applied to two-qubit gate noise?** | **No, not in this iteration.** `stim.Circuit.generated` exposes only four *symmetric* scalar probabilities and cannot express unequal X/Z rates at all. Bias is applied by rewriting **every** single-qubit `DEPOLARIZE1` into `PAULI_CHANNEL_1` — which covers both data noise and after-Clifford single-qubit gate noise, not data noise alone as first documented. `DEPOLARIZE2` is left symmetric. The true scope is recorded per file as `bias_scope`. |
| 12 | **In what layout does Nexus emit a proposed correction?** | Unknown. `qecgen` defines and exports a schema (Contract C-scoring) and scores anything conforming to it. It does **not** claim Nexus emits this layout. If Nexus emits something else, the adapter is a converter into this schema, not a change to the scoring rule. |
| 11 | Is a mechanism index comparable across environments? | Only when the DEMs enumerate mechanisms identically. Pooling Contract B labels checks mechanism **topology**, not just count, and refuses environments whose indexing differs. Under `FROZEN_PRIOR`, `--emit-mechanisms` is refused outright when the training and test DEMs disagree in count **or enumeration order** — the same topology comparison as pooling — since the labels index one DEM while the shipped structure describes another. |
| 6 | Which noise model does the client consider "circuit-level"? | `STIM_UNIFORM_CIRCUIT_LEVEL` sets all four channels to a single `p`. This is **one valid synthetic convention, not a universal definition**, hence the name. The full channel vector is stored per environment so results can state exactly which channels were active. |
| 7 | What counts as a "distinct environment" for invariant causal learning? | Each distinct parameter point on the chosen drift axis gets its own circuit, DEM and `EnvironmentSpec`. Shots carry `environment_id`. |
| 8 | For the drift study, is structure oracle-calibrated or frozen? | Must be stated explicitly per file via `ORACLE_CALIBRATED` / `FROZEN_PRIOR`, recorded in the manifest. Never an accident of which DEM was in scope. |
| 9 | How should multi-observable codes pack observables? | Same little-endian packing as detectors, width `ceil(n_observables/8)`. Surface code memory has 1 observable, so this is currently untested at width > 1. |
| 10 | Is a fixed decoder prior acceptable across drifted test sets? | This is the whole point of `FROZEN_PRIOR`. Exporting a p=0.010 test set with a DEM derived from p=0.010 hands the decoder the true test noise distribution and invalidates any generalisation claim. |

---

## The distinction that decides the drift study

**`ORACLE_CALIBRATED`** — structural information exported alongside a test file is derived
from **that file's own environment**.

**`FROZEN_PRIOR`** — structural information exported alongside a test file is derived from
the **nominated training environment**.

If a test set at p = 0.010 ships with `--structure dem` where the DEM was built at
p = 0.010, the decoder has been handed the true test-time noise distribution. Any claim of
generalisation to unseen noise is then unsupported, because nothing was unseen. Under
`FROZEN_PRIOR` the decoder gets the training environment's DEM and must genuinely
generalise.

Both conditions are legitimate experiments. Which one a file represents is recorded in its
manifest and is never inferred.

### Decoder-visible payload vs provenance

Freezing the numeric structure is necessary but **not sufficient**. A test file must also
not describe its own error model in words. An external review found that a
`--structure full` frozen-prior file, while correctly shipping the training environment's
priors, still embedded the *test* environment's DEM text in its manifest — enough to
reconstruct the withheld distribution exactly.

The file is therefore split, and the split is part of this contract:

| Location | Contents | May a decoder read it? |
|---|---|---|
| `manifest` | all parameters: distance, rounds, channels, axis, seed, hashes, `structure_dem_sha` (a BLAKE2b-128 digest — the algorithm is named in `structure_dem_algorithm`; the historical "sha" key is kept for compatibility) | **Yes** |
| `dem/` | H, L, priors, components, coordinates, from the environment the condition nominates | **Yes** |
| `provenance/` | per-environment circuit and DEM **text**, written only at `--structure full` | **No** |

Per format, the same three payloads land here:

| Payload | HDF5 | NPZ | Parquet | JSONL | CSV |
|---|---|---|---|---|---|
| manifest | root attr `manifest` | `manifest` array | schema KV metadata | line 1 `__manifest__` | comment line 2 `#__manifest__` |
| structure | `/dem` group | `dem_*` arrays | not round-tripped | line 2 `__structure__` | comment line 3 `#__structure__` |
| provenance | `/provenance` group | `provenance` array | not written | not written | comment line 4 `#__provenance__` |

**A format that will not carry a payload must not record a level that claims it.**
Parquet cannot reconstruct structure on read, so it records `structure_level: none`.
JSONL will not carry provenance, so asking it for `full` produces a file recording `dem`.
Both are the same rule — a manifest never claims more than its file holds — and it is one
function, `exporters.base.recorded_structure_level`, rather than a convention each
exporter restates. The in-memory manifest is never altered; only the persisted copy is.

The JSON structure encoding is normative, not incidental: the exact HDF5 key names; CSC
`data` omitted because every stored entry is 1 by construction; `components` **nested**
rather than flattened into value/offset pairs; NaN coordinates encoded as `null`, with
`allow_nan=False` on write and a rejecting `parse_constant` on read; and the bit-string
convention `s[0] == index 0`. JSONL and CSV share one implementation of it
(`exporters/structure_json.py`) so the two cannot drift apart; Parquet keeps an older,
divergent copy that is written but never read back.

**CSV header order is normative.** Line 1 is the literal magic line `#qecgen-csv v1`;
the manifest is line 2 so a reader obtains it in two `readline` calls without touching
the structure line, which is 1.5 MB at d=7. The first line not beginning with `#` is the
column header. A `.csv` without the magic line is **not a qecgen dataset** — `qecgen
sweep` writes its threshold-results table with the same extension — and readers refuse it
by name rather than inferring a schema from its columns.

**CSV is the only text format that carries provenance, and its wall is thinner.** In HDF5
the block is a separate group and in NPZ a separate array; in CSV it is a comment line in
the same byte stream as the decoder-visible rows. The contract is unchanged — `read()`
never returns it, no manifest path exposes it, and **a decoder, or any harness feeding
one, must never read `provenance/`** — and the hazard that keeps JSONL out is genuinely
absent here, because a reader that does not filter `#` lines does not find the table at
all. But a `--structure full` CSV still puts a frozen-prior test file's own DEM text one
`head` away from its shots. Prefer HDF5 for a `full` file a decoder will be pointed at,
and reserve full-level CSV for audit.

**A decoder, or any harness feeding one, must never read `provenance/`.** Under
`FROZEN_PRIOR` it contains precisely the distribution the condition exists to withhold.
It is retained so a reviewer can audit what was generated, and separated physically so
that reading the manifest cannot expose it.

Dropping the text from the manifest costs no reproducibility: distance, rounds, basis,
rotated, the full channel vector, axis, axis value, seed and chunk size are all recorded,
and together they regenerate each environment exactly.

---

## What a manifest must contain

Enough to regenerate the file exactly:

- Per dataset: distance, rounds, basis, rotated, total shots, seed, **chunk size**,
  `bit_order`, `contract`, `n_detectors`, `n_observables`, stim/sinter/pymatching versions,
  `qecgen` version, git commit, UTC timestamp, content hash.
- Per environment (`EnvironmentSpec`): `environment_id`, `p`, noise model, **full channel
  vector**, circuit text, DEM text, shots.

There is deliberately **no top-level `p`, `circuit` or `dem`**. Those are per-environment
properties, and promoting them to the dataset level is wrong the moment there is more than
one environment.

The content digest is **BLAKE2b with a 32-byte digest**, recorded as `content_hash` with
`content_hash_algorithm: "blake2b-256"`. It was previously named `content_sha256`, which
was simply wrong: anyone verifying it with SHA-256 would have concluded the file was
corrupt.

**Chunk size is part of the reproducibility contract.** Stim guarantees seeded
reproducibility only under matching version, machine characteristics and call structure.
Changing the chunk size changes the sequence of `sample()` calls and therefore the sample
stream. Determinism is asserted on **array contents and a content hash**, never on
byte-identical files — manifests contain timestamps.
