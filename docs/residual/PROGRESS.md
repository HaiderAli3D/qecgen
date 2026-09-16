# Residual-error datasets — progress log

Living record for the PyMatching residual-error dataset build (`qecgen/residual/`).
Dates are UTC. Everything under `data/` is ignored by git; this file is the tracked
record of what was measured and decided.

## Work completed

- 2026-09-15 — Phase 0 inventory (read-only): every candidate source identified by
  manifest sidecar + recomputed `content_hash`, not by filename (see "Source identities").
- 2026-09-15 — Google Willow release (Zenodo record 13273331) inspected without
  downloading the 5.7 GB archive: the ZIP64 central directory was read by HTTP range
  requests, the members for `d3_at_q10_7/Z/r10` were fetched individually (~1.3 MB) and
  cross-checked against the local mirror (see "Willow provenance").
- 2026-09-15 — Prefix-safety of the planned extended sampling streams verified by content
  hash for all three simulated sources (see "Scientific decisions", item 6).
- 2026-09-15 — Pilot decode of the first 2,000 rows of the d=9 and d=25 sources (see
  "Timings and storage estimates").
- 2026-09-15 — Implementation plan reviewed by three independent read-only critics
  (feasibility, science/leakage, completeness); findings folded into the plan.
- 2026-09-16 — `qecgen/residual/` implemented (config, splits, graph, features, decoder,
  remote_zip + zenodo, sources, checkpoint, writers + report, validation, sanity,
  pipeline + cli) with 438 focused tests; every module independently reviewed and the
  review findings fixed (see "Command log").
- 2026-09-16 — Inventory and pilots for all five configurations (see "Pilot results").
- 2026-09-16 — All five datasets built, validated and published (see "Build results" and
  "Validation results"); `data/residual/MANIFEST.md` rendered.
- 2026-09-16 — Documentation: `docs/residual/README.md`, README pointer, this log.

## Current blockers

- None. All four required datasets and the additional Willow RL-prior dataset are
  published and validated. All four required sources are identified, verified and decodable. The Willow
  decoder model is Google's shipped `error_model.dem` (verified, see below), so no
  data-fitted matching weights are needed and no `calibration` split is used.

## Source identities

All paths are absolute on this machine; the tree was moved from `C:\Projects\qecgen` to
`C:\Projects\quantum\qecgen`, so recorded paths inside old manifests/run records carry the
old prefix and must be rebased (they are historical, not live inputs).

### 1. `indep_d9_r200_p0005` — independent-noise distance 9

| field | value |
|---|---|
| source path | `C:\Projects\quantum\qecgen\data\qecgen-dataset-16000.shots.ml.csv` (512,356,902 B) |
| manifest path | `C:\Projects\quantum\qecgen\data\qecgen-dataset-16000.shots.ml.manifest.json` (sha256 `910e87dd932e75b6d97b82ebecbfd8fa44ada797b1b50c7515c26d1bffd7e182`) |
| content hash | `84ec11b1cac62e74eead3177a0dd01b8b77b145f23f993fe5101f5f7a426dab0` (blake2b-256, qecgen `content_hash`) |
| distance / rounds / basis / layout | 9 / 200 / Z / rotated |
| noise model | `stim_uniform_circuit_level`, p = 0.005 (all four channels 0.005) |
| detectors / observables | 16,000 / 1 |
| shots / seed / chunk size | 16,000 / 0 / 100,000 (one `sample(16000)` call) |
| circuit identity | `environments.build_environment(0, 9, 0.005, DriftAxis.P, 0.005, 16000, STIM_UNIFORM_CIRCUIT_LEVEL, rounds=200, basis=Z, rotated=True)`; rebuilt channels equal the recorded channel vector |
| DEM availability | exact: `circuit.detector_error_model(decompose_errors=True)` — 360,015 errors, 82,990 matching edges |
| run record | `data/runs/47f36dbaefdb.json` (generated 2026-09-01T21:59:30Z, git 51cb5ae9) |
| provenance limitations | none beyond simulation; no prior PyMatching benchmark exists for this file |

### 2. `indep_d25_r25_p0005` — independent-noise distance 25

| field | value |
|---|---|
| source path | `C:\Projects\quantum\qecgen\data\d25.ml.csv` (499,550,902 B) |
| manifest path | `C:\Projects\quantum\qecgen\data\d25.ml.manifest.json` (sha256 `9b6cd50afa160fd7de1bc8f47686b316c61f9d4130be0a2e295c78eb8017153f`) |
| content hash | `521c3dac32227ffff15c3992c66bebc58a717fdfa91734b153ed66dc3097639c` |
| distance / rounds / basis / layout | 25 / 25 / Z / rotated |
| noise model | `stim_uniform_circuit_level`, p = 0.005 |
| detectors / observables | 15,600 / 1 |
| shots / seed / chunk size | 16,000 / 0 / 100,000 |
| circuit identity | `build_environment(...)` as above with distance 25, rounds 25 |
| DEM availability | exact — 365,558 errors, 87,022 matching edges |
| duplicates | `data/dataset.ml.csv`, `data/large-dataset.csv`, `data/large-dataset2.csv`, `data/large-dataset-cords.csv` are byte-identical (sha256 `998f363c148e76c8…`); `d25.ml.csv` has no run record of its own and is a copy of one of them (manifest `generated_at` 2026-09-09T21:29:54Z) |
| provenance limitations | as above |

### 3. `device_static_d3_r3` — the recently shared realistic-noise small test set

| field | value |
|---|---|
| source path | `C:\Projects\quantum\qecgen\data\small-2000\test-real-shots.ml.csv` (109,190 B) |
| manifest path | `C:\Projects\quantum\qecgen\data\small-2000\test-real-shots.ml.manifest.json` (sha256 `4e124eb4a0bb153f97c6781c7a49225a45b4fe832269362c4fbf532de054a434`) |
| content hash | `d6033e32f977f830ece28cddad40d6e8f13a060378500ef445b55796324810d8` |
| identification evidence | run record `data/runs/f7011a6ad078.json` (mode `configured`, created 2026-09-09T22:46:40Z, output `…\small-2000\shots.ml.csv`, same content hash — the file was renamed on disk afterwards); manifest v2 `generated_at` 2026-09-09T22:46:41Z; it is the most recent device-profile dataset in the repository and the only one in `ml_csv` form at 2,000 shots. The equal-size look-alikes `formated-dataset-test*.ml.csv` / `demo/qecgen-sample.ml.csv` (also 109,190 B) are legacy p=0.005 seed-1 files with content hash `c8c4051c…` — different content. |
| distance / rounds / basis / layout | 3 / 3 / Z / rotated |
| noise model | device profile, **static** (`drift`, `bursts`, `leakage`, `coherence` all `enabled: false`): probabilities one_qubit_gate 0.001, two_qubit_gate 0.003, measurement 0.01, reset 0.002, idle 0.0001; qubit_overrides `1: idle 0.0005`, `2: measurement 0.02`; edge_overrides `2,3: 0.005`; spatial `XX` on qubits [1,3] p=0.0002 (`scenario_assumption`); `profile_sha256 15e4de32b92b6c461712cd5fa13eba2f501ded56f384be4f25a3937664c5c281`; `parameter_provenance.kind = scenario` (not fitted to hardware) |
| detectors / observables | 24 / 1 |
| shots / seed / chunk size | 2,000 / 20260909 / 2,000 |
| circuit identity | `configuration.ideal_circuit(generation_config)` + `noise.build_noisy_circuit`; `generation_audit.circuit_sha256 = 1310adbb06c13a8f1fe0d09d4c75c4cd081744354d64326a9192c804e60b314b`, `ideal_circuit_sha256 = 169e6e44…` |
| DEM availability | exact static DEM of the noisy circuit (286 errors, 208 with `^` decompositions, 0 undecomposable) |
| same-profile sibling | `data/device-static.h5` (100,000 shots, seed 12345, same `profile_sha256`) |
| provenance limitations | scenario parameters, not a calibrated device |

### 4. `willow_d3_z_r10_si1000` (required) and `willow_d3_z_r10_rlprior` (additional)

| field | value |
|---|---|
| formatted sample | `C:\Projects\quantum\qecgen\data\real-world-2000\willow-shots.ml.csv` (333,862 B), manifest sha256 `3db56d0831f6ea4d0503a62f033a0c9ae1723cf6ea240890e749d38a137e67ab`, content hash `61daf0218f9e6260ff47ff2ab5a2b0aab352231699b291d45fa92abb756e6368`, run `data/runs/337e37018aba.json`, `hardware.offset 0`, `source_rows {offset 0, count 2000}` |
| source cohort (table) | `C:\Projects\quantum\qecgen\data\realism\raw\willow-derived-z-r010\d3_at_q10_7__Z__r010.parquet` (308,494 B, sha256 `2324ecb77a859d005850341a442396d6c3e9f8a2f12a4928364f27933b8ac696`), 50,000 rows |
| source cohort (circuit) | `C:\Projects\quantum\qecgen\data\realism\raw\willow-derived-z-r010-circuit\d3_at_q10_7__Z__r010.stim` (8,525 B, sha256 `fba4d5575c0afa11ce2126acbbe7d3a2546609ecac66195ea2ba696c45ef085e`), ideal (noiseless) circuit |
| mirror | HuggingFace `ShayManor/willow-surface-code-detection-events` @ `79ea4cbbc278047c9ce5d4d74d79a0f38aa28c7c` (intake receipts in `data/realism/intake/`) |
| prefix verification | the 2,000-row file equals parquet rows 0–1,999 (every detector bit and observable, row order) and `willow-z10.h5[:2000]` — verified 2026-09-15 |
| distance / rounds / basis / orientation | 3 / 10 / Z / q10_7, rotated (XZZX surface code per the Google README) |
| detectors / observables | 80 / 1 |
| shots / seed / chunk size | 50,000 recorded rows / n/a (seed unused) / 10,000 (processing only) |
| circuit identity | catalogue-verified Willow circuit (`hardware.WILLOW_CIRCUITS`); `mapping-z10.json` audit ties it to qecgen's canonical d=3/Z/10 circuit through 359 exact fault signatures |
| DEM availability | Google's shipped `error_model.dem` files (see below); not derivable from the ideal circuit alone |
| provenance limitations | source row order preserved; acquisition chronology unverified; archive-level MD5 not re-verified |

#### Willow provenance (verified 2026-09-15)

- Zenodo record 13273331, "Data for 'Quantum error correction below the surface code
  threshold'", version 1.0.0, published 2024-08-26, CC BY 4.0. Archive
  `google_105Q_surface_code_d3_d5_d7.zip`, 5,716,907,033 B, published MD5
  `21fa6ad35b395d838ebcdbc92e364a12` (not recomputed — would require the full download).
- The archive's ZIP64 central directory (9,959 members) was read by HTTP range requests
  (`https://zenodo.org/records/13273331/files/google_105Q_surface_code_d3_d5_d7.zip?download=1`,
  `Accept-Ranges: bytes`). Members under
  `google_105Q_surface_code_d3_d5_d7/d3_at_q10_7/Z/r10/` were fetched individually and
  CRC32-verified against the central directory:

| member | uncompressed bytes | sha256 |
|---|---:|---|
| `circuit_ideal.stim` | 8,525 | `fba4d5575c0afa11ce2126acbbe7d3a2546609ecac66195ea2ba696c45ef085e` (byte-identical to the local mirror circuit) |
| `circuit_noisy_si1000.stim` | 19,050 | `3a3003d145c0961899f165fb826d9928cad19823db3b6bdf794550fb94c4c017` |
| `detection_events.b8` | 500,000 | `c9f6ad73d64eba7c5f588dec3f3d13cdc3cd20baa9b696f10e06188fa67da2d7` |
| `obs_flips_actual.b8` | 50,000 | `184364ae0483af36e00c868af89fb4032b2e78cc0eb32aac5a96dc55cff2f6a9` |
| `sweep_bits.b8` | 100,000 | `7249f5a3b91a253e5d25e079d012b2b10e36fa56c5b7eac6801213ab63a08171` |
| `metadata.json` | 252 | `4eec7e52dff0cb04fa45d0d207295ab03b8b4ae8eb017728942acbadcb62008f` |
| `decoding_results/correlated_matching_decoder_with_si1000_prior/error_model.dem` | 47,004 | `b7758707f3e1df6f61f98721b445325b590bd4e045688b159a42b95ffa4cd08e` |
| `decoding_results/correlated_matching_decoder_with_rl_optimized_prior/error_model.dem` | 46,726 | `5352565ddc6704152e04258a0c4e2911fd26fb7da5a5b470c09a08ce3c251578` |
| `README/README.md` | 13,070 | `2159442ec57b661718d7276ac841a7736ec462a0177766b3d7060e6c6ae19561` |

- The local parquet's detectors and observables equal `detection_events.b8` /
  `obs_flips_actual.b8` for **all 50,000 rows** (0 differing rows). The mirror's
  `dems/*.dem.gz` gunzip to bytes identical to the Zenodo `error_model.dem` members.
- Both shipped DEMs: 80 detectors, 1 observable, 1,003 error instructions (701 with `^`
  decompositions), 0 hyperedges ignored by PyMatching, detector coordinates identical to
  `circuit_ideal.stim`; PyMatching graph 302 edges, 80 nodes, 80 boundary edges, 22 edges
  flipping observable 0. The si1000 DEM is byte-identical across the harmony and
  correlated-matching pathways; the RL DEM across harmony/correlated/libra.
- The si1000 DEM is **not** reproducible from `circuit_noisy_si1000.stim` via
  `detector_error_model(decompose_errors=True)` (1,353 errors, not approx-equal); its
  construction is undocumented. It is taken from a correlated-matching pathway but is
  decoded here with standard MWPM (`enable_correlations=False`).
- The RL prior was "optimized jointly for all distance-3 and distance-5 patches using the
  13-cycle calibration data" (archive README) with a logical-error-rate objective (Sivak
  et al., arXiv:2406.02700): same-device, logical-outcome-tuned; overlap with this 10-cycle
  cohort unknown. It is an additional, disclosed dataset, not the required one.
- Standard PyMatching on all 50,000 rows (planning probe): si1000 prior 3,992 failures =
  7.984 % [7.748, 8.225]; RL prior 3,630 = 7.260 % [7.034, 7.491]. Google's own correlated
  matching pathways: 6.324 % / 5.622 % (diagnostic only; predictions never used).

## Scientific decisions

1. Dataset names, sources and decoders are exactly those in the plan
   (`~/.claude/plans/task-build-pymatching-humble-turtle.md`, Decision 1): d9 and d25 use
   the decomposed DEM of the rebuilt legacy circuit; the device dataset uses the exact
   static DEM of the noisy circuit; Willow uses Google's shipped si1000 DEM (required) and
   RL DEM (additional).
2. Time slices: a detector's slice is the rank of its latest time coordinate
   (`qecgen.hardware.detector_anchors`). Slice counts: d9 201, d25 26, device 4, Willow 11.
3. Graph node sets are derived from the decomposed DEM's graphlike components, not from
   `matching.edges()`, because PyMatching 2.4.0 keeps only the first parallel edge's fault
   ids. A DEM whose detector pairs carry conflicting observable sets is refused (0 such
   pairs measured for all five DEMs).
4. Feature schema version 1 = the brief's 24 input columns + `truth, pm_wrong, run_id,
   split`; the optional edge features are prototyped but excluded unless every brief
   condition holds (decision recorded below after the prototype).
5. Splits: simulated datasets use a seeded permutation (`split_seed 20260915`,
   70/15/15); Willow uses contiguous source-row blocks (60/20/20), chronology unverified.
   No calibration split is needed because no matching weights are fitted here.
6. Extension is prefix-safe under the call-size-sequence rule. Verified 2026-09-15 by
   content hash: `iter_chunks(circuit, 32000, seed=0, chunk_size=16000)` rows 0–15,999
   reproduce `84ec11b1…` (d9) and `521c3dac…` (d25);
   `iter_profile_chunks(ideal, profile, 4000, 20260909, 2000)` rows 0–1,999 reproduce
   `d6033e32…` (device). Planned runs: d9/d25 304,000 shots at chunk 16,000 seed 0;
   device 300,000 shots at chunk 2,000 seed 20260909.
7. Willow decoder provenance as recorded above; the phrase "no data-fitted weights" is
   never used; the RL prior's third-party fitting is disclosed in every artifact.
8. Dynamic profiles (no deliverable): frozen reference = static blocks kept, drift
   baseline represented as static single-qubit spatial terms, bursts/leakage omitted and
   said so; supported and tested but not built at scale.
9. Raw HDF5 arrays live under group `/residual` so the file cannot be mistaken for an
   interrupted qecgen write (root `detectors` without a manifest).
10. Expectation bands: d9 measured 16.0 % [14.4, 17.7] on the first 2,000 rows (accuracy
    84.0 %, the edge of both the 6–16 % and 84–94 % bands; consistent with 200 rounds of
    accumulation at p = 0.005). d25 measured 0/2,000 (95 % upper bound 0.18 %) — far below
    the band; the scaled dataset will contain very few positives. Nothing is adjusted; the
    alignment investigation is run before scaling. The "84–94 %" figure matches the Willow
    cohort MWPM results in `data/realism/results/final-evaluation/results.json`; no prior
    benchmark existed for d9/r200 or d25/r25.
11. `CLAUDE.md`/`AGENTS.md` are not edited (they carry unrelated uncommitted edits).

## Deviations from the brief

- `.ml.csv` files carry no in-band `#` manifest lines (that is the `csv` format); the
  manifest is the `*.ml.manifest.json` sidecar. Sources are identified by sidecar plus a
  recomputed `content_hash`.
- The device dataset's `sample()` call size is 2,000 (below the suggested 10,000–50,000)
  because only that size reproduces the 2,000-shot source; checkpoint chunks accumulate
  calls to 10,000 rows, and no unpacked full matrix is ever held.
- Raw HDF5 arrays under group `/residual` (see decision 9).
- A fifth, additional dataset (Willow RL prior) beyond the four required.
- `inventory`/`pilot` records live under `data/residual/.checkpoints/<name>/`, not in
  the published directory (which only the staged commit may populate).
- Development happens in per-task commits on `feat/residual-datasets`, squashed into one
  focused commit after the full gate run; only that hash is reported.

## Timings and storage estimates

Planning-phase measurements (scratch scripts, read-only; 32 CPUs, 32 GiB RAM, 115 GiB free):

| quantity | d9/r200 | d25/r25 | Willow |
|---|---|---|---|
| circuit rebuild + DEM | 0.03 s | 0.21 s | n/a |
| matcher build | 0.07 s | 0.17 s | 0.002 s |
| PyMatching decode (batch) | 0.58 ms/shot | 0.62 ms/shot | 0.0013 ms/shot |
| ml_csv streaming read | ≈1.0 ms/row | ≈1.0 ms/row | n/a |
| Stim sampling, 32,000 shots | 0.63 s | 0.62 s | n/a |
| fired detectors per shot (mean) | 1,284 | 1,329 | — |
| projected decode, 304,000 shots | ≈3 min | ≈3.2 min | — |
| projected raw HDF5 (uncompressed) | ≈608 MB | ≈593 MB | 0.5 MB |
| projected feature CSV | ≈60 MB | ≈60 MB | ≈10 MB |

Feature extraction is vectorised; per 2,000-row block on 16,000 detectors the dense
neighbour-count product is ≈128 MB. Estimates are refreshed by the 10,000-shot pilots
before each scaled build (see "Pilot results").

## Pilot results

All five configurations were inventoried and piloted on 2026-09-16 with
`python -m qecgen.residual.cli inventory|pilot --config examples/residual/<name>.json`
(records under `data/residual/.checkpoints/<name>/`). The pilot decodes **every** existing
source row, runs the alignment investigation, and times the stages on 10,000 freshly
generated rows.

### PyMatching on the existing source rows

| dataset | rows | `pm_wrong` | error rate | 95 % CI (Clopper–Pearson) | accuracy | expectation |
|---|---:|---:|---:|---|---:|---|
| `indep_d9_r200_p0005` | 16,000 | 2,511 | 15.694 % | [15.133 %, 16.267 %] | 84.31 % | inside the 6–16 % band and the 84–94 % accuracy range (at the edge) |
| `indep_d25_r25_p0005` | 16,000 | 3 | 0.019 % | [0.004 %, 0.055 %] | 99.98 % | **far below** the 6–16 % band; see below |
| `device_static_d3_r3` | 2,000 | 16 | 0.800 % | [0.458 %, 1.296 %] | 99.20 % | band not applied (device data) |
| `willow_d3_z_r10_si1000` | 50,000 | 3,992 | 7.984 % | [7.748 %, 8.225 %] | 92.02 % | band not applied (hardware); inside the historical 84–94 % Willow range |
| `willow_d3_z_r10_rlprior` | 50,000 | 3,630 | 7.260 % | [7.034 %, 7.491 %] | 92.74 % | as above |

### Alignment investigation (every configuration)

| dataset | self-consistency (`qa.decode_stored_shots`) | permuted columns | big-endian unpack | per-detector marginal r / max abs diff | re-sample (16,000 fresh rows) | detection-event rate stored vs fresh |
|---|---|---:|---:|---|---|---|
| d9 | equal (2,511 = 2,511) | 50.29 % | 50.04 % | 0.990 / 0.0092 | 2,510/16,000 = 15.69 %, CI overlaps | 0.08027 vs 0.08023 (0.05 %) |
| d25 | equal (3 = 3) | 50.38 % | 50.02 % | 0.984 / 0.0091 | 3/16,000, CI overlaps | 0.08515 vs 0.08511 (0.05 %) |
| device | equal (16 = 16) | 21.65 % | 19.80 % | 0.934 / 0.0128 | 121/16,000 = 0.756 %, CI overlaps | 0.04685 vs 0.04425 (5.9 %) |
| Willow si1000 | n/a (official DEM) | 41.60 % | 43.48 % | 0.789 / 0.0595 | n/a (recorded rows) | n/a |
| Willow RL prior | n/a | 41.50 % | 43.05 % | 0.912 / 0.0323 | n/a | n/a |

Reading: the negative controls sit at 50 % for the two large simulated configurations and
far above the aligned rate everywhere (d=3 syndromes are sparse, so a misordered d=3
decode still returns 0 often); the per-detector firing rates agree with the DEM-predicted
marginals; the fresh re-samples reproduce the stored rates. The self-consistency row shares
the decoder's construction path and is not alignment evidence. The d25 rate is far below
the 6–16 % expectation: every control confirms alignment, the fresh re-sample gives the
same 3/16,000, and the rate is what d=25 at p=0.005 (well below threshold) yields after 25
rounds. Nothing was adjusted; the scaled d25 dataset will contain only tens of positives
(see "Scientific limitations"). For Willow the si1000 prior's marginal agreement (r=0.79)
is below the 0.9 heuristic while the RL prior, on the *same* detector ordering, gives
0.91: the discrepancy is the generic SI1000 prior's calibration, not ordering — ordering
is established by the byte-equality with the archive members, the metadata/coordinate
checks, the near-50 % negative controls and the mapping audit.

### Throughput, projections and the edge-feature prototype

| quantity | d9 | d25 | device | Willow |
|---|---|---|---|---|
| sampling / decoding / features / writing per row | 0.035 / 0.491 / 0.279 / 0.073 ms | 0.035 / 0.545 / 0.252 / 0.076 ms | 0.002 / <0.001 / <0.001 / 0.005 ms | — / 0.001 / 0.001 / 0.006 ms |
| peak working set (10,000-row pilot) | 1.20 GB | 1.17 GB | 0.16 GB | — |
| raw HDF5 gzip ratio | 0.50 | 0.52 | 1.11 | — |
| projected full run | 304,000 rows, 4.5 min, 1,005 MiB | 304,000 rows, 4.6 min, 1,004 MiB | 300,000 rows, <1 min, 106 MiB | 50,000 rows, <1 min, 21 MiB |
| free disk at pilot time | 101.4 GiB | | | |
| `resources_sufficient` | true | true | true | true |

Edge-feature prototype (`decode_to_edges_array`, 10,000 d9 rows): 0.730 ms/row (≈3.7 min
for 304,000 rows), mean 704 matched edges per row, `merged_pairs_with_differing_observables`
= 0 for this DEM (and `n_conflicting_pairs` = 0 for all five DEMs). **Decision: the
optional `pm_*` edge features stay out of schema version 1.** Throughput and fault-id
recovery would allow them, but the brief also requires a distance metric "defined
consistently" across every dataset, and the Stim-generated coordinates (half-integer
lattice units, `t` per round) and Google's hardware coordinates (qubit-grid units with
multi-triple detectors) do not share one metric; adopting one would change the schema for
every dataset. The measurement is recorded so a schema-version-2 decision can be taken
later without repeating it.

## Build results (2026-09-16)

`python -m qecgen.residual.cli build-all --config-dir examples/residual` built the five
datasets in 11 min 5 s wall-clock (device → Willow si1000 → d9 → d25 → Willow RL prior),
each behind its pilot gate. Every `extend` build reproduced its source as an exact prefix
(content hash of rows `0..source_shots-1` equal to the manifest's) before continuing the
seeded stream; every build passed the fifteen validation checks and the spot checks on
the staged copy before publication; `MANIFEST.md` reports 5 completed, 0 blocked, 0 failed.

### PyMatching baseline on the full datasets

| dataset | runs | `pm_wrong` | PyMatching error rate | 95 % CI | always-zero accuracy | split sizes (train/val/test) |
|---|---:|---:|---:|---|---:|---|
| `indep_d9_r200_p0005` | 304,000 | 47,156 | 15.5118 % | [15.3833 %, 15.6410 %] | 84.4882 % | 212,800/45,600/45,600 |
| `indep_d25_r25_p0005` | 304,000 | 54 | 0.0178 % | [0.0133 %, 0.0232 %] | 99.9822 % | 212,800/45,600/45,600 |
| `device_static_d3_r3` | 300,000 | 2,437 | 0.8123 % | [0.7805 %, 0.8451 %] | 99.1877 % | 210,000/45,000/45,000 |
| `willow_d3_z_r10_si1000` | 50,000 | 3,992 | 7.9840 % | [7.7479 %, 8.2250 %] | 92.0160 % | 30,000/10,000/10,000 |
| `willow_d3_z_r10_rlprior` | 50,000 | 3,630 | 7.2600 % | [7.0340 %, 7.4909 %] | 92.7400 % | 30,000/10,000/10,000 |

Positive counts: d9 47,156; d25 **54**; device 2,437; Willow si1000 3,992; Willow RL 3,630.
The d9 rate on 304,000 rows (15.51 %) sits inside the 6–16 % expectation; the d25 rate
(0.018 %) is far below it, as the pilot predicted, and the scaled dataset carries only 54
positives (10 in the test split). This is the property of d=25 at p=0.005, not an
alignment defect (see "Pilot results"); the dataset is delivered as specified and the
sparsity is reported as a scientific limitation.

### Artifact sizes (published directories)

| dataset | total size | files |
|---|---:|---|
| `indep_d9_r200_p0005` | 384.4 MB | decoder.dem 0.5 MB; decoder_metadata.json 0.0 MB; features.csv 74.9 MB; note.md 0.0 MB; raw.h5 309.0 MB; resolved_config.json 0.0 MB; sanity.json 0.0 MB; summary.json 0.0 MB; validation.json 0.0 MB |
| `indep_d25_r25_p0005` | 401.0 MB | decoder.dem 3.3 MB; decoder_metadata.json 0.0 MB; features.csv 83.2 MB; note.md 0.0 MB; raw.h5 314.4 MB; resolved_config.json 0.0 MB; sanity.json 0.0 MB; summary.json 0.0 MB; validation.json 0.0 MB |
| `device_static_d3_r3` | 42.6 MB | decoder.dem 0.0 MB; decoder_metadata.json 0.0 MB; features.csv 41.7 MB; note.md 0.0 MB; raw.h5 0.9 MB; resolved_config.json 0.0 MB; sanity.json 0.0 MB; summary.json 0.0 MB; validation.json 0.0 MB |
| `willow_d3_z_r10_si1000` | 10.0 MB | decoder.dem 0.0 MB; decoder_metadata.json 0.0 MB; features.csv 9.7 MB; note.md 0.0 MB; raw.h5 0.3 MB; resolved_config.json 0.0 MB; sanity.json 0.0 MB; summary.json 0.0 MB; validation.json 0.0 MB |
| `willow_d3_z_r10_rlprior` | 10.0 MB | decoder.dem 0.0 MB; decoder_metadata.json 0.0 MB; features.csv 9.7 MB; note.md 0.0 MB; raw.h5 0.3 MB; resolved_config.json 0.0 MB; sanity.json 0.0 MB; summary.json 0.0 MB; validation.json 0.0 MB |

Checkpoint chunks (`data/residual/.checkpoints/<name>/*.chk`, ignored by git) were left
in place for resumability: 640 MB (d9), 625 MB (d25), 60 MB (device), 11 MB each (Willow).
They can be deleted once the published artifacts are archived.

### Sanity residual models (train-fit, validation-threshold, single test evaluation)

| dataset | model | positive prevalence (test) | balanced accuracy | ROC AUC | PR AUC | precision / recall (pm_wrong=1) | confusion matrix (test) | PyMatching test LER [95 % CI] | corrected test LER [95 % CI] | change | flips |
|---|---|---:|---:|---:|---:|---|---|---|---|---|---|
| `indep_d9_r200_p0005` | hist_gradient_boosting | 15.601 % | 0.500 | 0.573 | 0.192 | n/a / 0.000 | tp 0, fp 0, fn 7114, tn 38486 | 15.6009 % [15.2689, 15.9372] | 15.6009 % [15.2689, 15.9372] | +0.0000 pp (+0.00 %) | 0 (0 correcting, 0 harmful); McNemar p=1.000 |
| `indep_d9_r200_p0005` | logistic_regression | 15.601 % | 0.500 | 0.581 | 0.197 | n/a / 0.000 | tp 0, fp 0, fn 7114, tn 38486 | 15.6009 % [15.2689, 15.9372] | 15.6009 % [15.2689, 15.9372] | +0.0000 pp (+0.00 %) | 0 (0 correcting, 0 harmful); McNemar p=1.000 |
| `indep_d25_r25_p0005` | hist_gradient_boosting | 0.022 % | 0.500 | 0.584 | 0.001 | n/a / 0.000 | tp 0, fp 0, fn 10, tn 45590 | 0.0219 % [0.0105, 0.0403] | 0.0219 % [0.0105, 0.0403] | +0.0000 pp (+0.00 %) | 0 (0 correcting, 0 harmful); McNemar p=1.000 |
| `indep_d25_r25_p0005` | logistic_regression | 0.022 % | 0.500 | 0.691 | 0.0 | n/a / 0.000 | tp 0, fp 0, fn 10, tn 45590 | 0.0219 % [0.0105, 0.0403] | 0.0219 % [0.0105, 0.0403] | +0.0000 pp (+0.00 %) | 0 (0 correcting, 0 harmful); McNemar p=1.000 |
| `device_static_d3_r3` | hist_gradient_boosting | 0.831 % | 0.533 | 0.943 | 0.218 | 0.556 / 0.067 | tp 25, fp 20, fn 349, tn 44606 | 0.8311 % [0.7493, 0.9194] | 0.8200 % [0.7388, 0.9077] | -0.0111 pp (-1.34 %) | 45 (25 correcting, 20 harmful); McNemar p=0.551 |
| `device_static_d3_r3` | logistic_regression | 0.831 % | 0.500 | 0.942 | 0.121 | n/a / 0.000 | tp 0, fp 0, fn 374, tn 44626 | 0.8311 % [0.7493, 0.9194] | 0.8311 % [0.7493, 0.9194] | +0.0000 pp (+0.00 %) | 0 (0 correcting, 0 harmful); McNemar p=1.000 |
| `willow_d3_z_r10_si1000` | hist_gradient_boosting | 8.210 % | 0.505 | 0.826 | 0.274 | 0.615 / 0.010 | tp 8, fp 5, fn 813, tn 9174 | 8.2100 % [7.6792, 8.7653] | 8.1800 % [7.6501, 8.7344] | -0.0300 pp (-0.37 %) | 13 (8 correcting, 5 harmful); McNemar p=0.581 |
| `willow_d3_z_r10_si1000` | logistic_regression | 8.210 % | 0.500 | 0.821 | 0.265 | n/a / 0.000 | tp 0, fp 0, fn 821, tn 9179 | 8.2100 % [7.6792, 8.7653] | 8.2100 % [7.6792, 8.7653] | +0.0000 pp (+0.00 %) | 0 (0 correcting, 0 harmful); McNemar p=1.000 |
| `willow_d3_z_r10_rlprior` | hist_gradient_boosting | 7.400 % | 0.500 | 0.808 | 0.235 | n/a / 0.000 | tp 0, fp 0, fn 740, tn 9260 | 7.4000 % [6.8944, 7.9306] | 7.4000 % [6.8944, 7.9306] | +0.0000 pp (+0.00 %) | 0 (0 correcting, 0 harmful); McNemar p=1.000 |
| `willow_d3_z_r10_rlprior` | logistic_regression | 7.400 % | 0.500 | 0.813 | 0.234 | n/a / 0.000 | tp 0, fp 0, fn 740, tn 9260 | 7.4000 % [6.8944, 7.9306] | 7.4000 % [6.8944, 7.9306] | +0.0000 pp (+0.00 %) | 0 (0 correcting, 0 harmful); McNemar p=1.000 |

Reading, honestly: no configuration shows a statistically meaningful reduction of the
held-out logical error rate. The gradient-boosted model reaches a modest ROC AUC on the
d=3 datasets (0.94 device, 0.83 / 0.81 Willow) but almost no precision–recall AUC, and the
validation-chosen thresholds flip few or no test rows: −1.34 % relative on the device split
(25 correcting vs 20 harmful flips, McNemar p = 0.55) and −0.37 % on Willow si1000
(8 vs 5, p = 0.58) — both consistent with chance. On d9 the features are barely
informative (ROC AUC 0.57–0.58) and both models select "flip nothing"; on d25 there are
too few positives for any conclusion; on the RL-prior Willow dataset both models select
"flip nothing". Logistic regression selects "flip nothing" everywhere. The always-zero
baseline (never flip) is therefore never beaten by more than noise. The datasets exist so
that stronger residual models can be tried; this result says nothing beyond these two
baselines on schema-version-1 features.

## Validation results (2026-09-16)

Every dataset was validated twice: on the staged copy inside `build` (before publication)
and afterwards in a fresh process with `python -m qecgen.residual.cli validate --output
data/residual/<name>`. All fifteen named checks passed for all five datasets
(`row_count_csv_equals_raw`, `run_id_unique_contiguous_identical`,
`raw_widths_match_true_widths`, `bit_packing_little_endian`, `binary_columns`,
`pm_wrong_consistent`, `batch_vs_single_decode`, `no_nan_inf`, `schema_exact`,
`fractions_in_range`, `note_matches_data`, `checksums_match`, `source_prefix_matches`,
`feature_inputs_audit`, `no_fit_on_held_out`), each with five random spot rows decoded
one at a time and every feature recomputed by the pure-Python reference extractor
(max feature difference 0.0), plus the big-endian negative witness. Fresh-process
wall-clock: device 2.9 s, Willow 1.7–1.8 s, d9 28.6 s, d25 28.4 s (the d9/d25 time is the
re-streaming of the 500 MB source files for `checksums_match`).

Alignment with the earlier 84–94 % accuracy range (brief): d9 84.49 % (inside); d25
99.98 % (outside, investigated — genuine low failure rate at d=25, p=0.005, every control
passed, nothing adjusted); Willow 92.02 % / 92.74 % (inside, and the range is the one the
repository's own Willow MWPM study reported).

## Scientific limitations

- **d25 sparsity.** 54 `pm_wrong` rows in 304,000 (10 in the test split). Any residual
  claim on this dataset is statistically empty; a higher-noise d=25 configuration would be
  a new configuration and is left to the user.
- **The features are weak.** On d9 both baselines have ROC AUC ≈ 0.58 and choose to flip
  nothing; the schema-version-1 summary features do not separate PyMatching's failures at
  this scale. The datasets are delivered so that richer residual models can be tried; the
  sanity result is not evidence for or against the hypothesis beyond these baselines.
- **Willow provenance.** Detector ordering is established structurally (archive-member
  byte equality, metadata and coordinate partition, near-50 % negative controls, the
  mapping audit), but acquisition chronology is unknown; splits are contiguous source-row
  blocks. The si1000 DEM is Google's shipped file (not reproducible from the shipped noisy
  circuit) decoded here with standard rather than correlated matching; the RL prior was
  tuned by Google on same-device 13-cycle data and is an additional dataset, not an
  independent baseline. The 5.7 GB archive MD5 was not recomputed.
- **d=3 graphs.** Every node is boundary-adjacent, so `frac_fired_boundary_adjacent`
  duplicates `frac_fired` for the device and Willow datasets.
- No Nexus compatibility is claimed; no column is a physical Pauli fault label;
  `pm_weight` is a sum of matching-edge weights.

## Gates and commit (2026-09-16)

Run from the repository root on the final tree (verbatim results):

| command | result |
|---|---|
| `python -m pytest -q -m "not slow"` | 1410 passed, 8 deselected in 62 s |
| `python -m pytest -q -m slow` | 8 passed |
| `python -m pytest research/realism/tests -q` | 129 passed, 2 skipped (`test_decoder.py`, `test_evaluate.py`: `torch` is not installed in the global interpreter) |
| `python -m pytest tests/test_residual_*.py -q` | 438 passed in 8 s |
| `ruff check .` | All checks passed! |
| `ruff format --check .` | 206 files already formatted |
| `mypy --strict qecgen tests` | Success: no issues found in 93 source files |
| `python -m qecgen.cli --help` | usage printed |
| `python -m qecgen.residual.cli --help` | usage printed (inventory, pilot, build, validate, build-all, manifest) |

Commit: the development commits on `feat/residual-datasets` were squashed with
`git reset --soft 80e7a89` into one focused commit containing `qecgen/residual/`,
`tests/test_residual_*.py`, `examples/residual/`, `docs/residual/`, the README pointer
and the `pyproject.toml` `residual` extra + mypy override. `AGENTS.md`, `CLAUDE.md`,
`.agents/`, `.codex/` and everything under `data/` were left untouched and uncommitted
(`git status --ignored -- data/residual` lists the outputs as ignored). Not pushed. The
commit hash is reported in the delivery message.

## Command log

- 2026-09-15 planning probes (scratchpad, read-only): `pilot_probe.py` (d9, d25 first
  2,000 rows), `willow_probe.py` (Zenodo range fetch + cross-checks), `prefix_check.py`
  (prefix-safety by content hash), `merge_check.py` (PyMatching parallel-edge merge rule).
- 2026-09-16 implementation on branch `feat/residual-datasets` (base `main` @ 80e7a89):
  development commits `6eedbad` (phase 0), `6c8ce85` (modules + tests), `6b95b21`
  (post-review fixes, README pointer), `efe31a3` (docs), then the PROGRESS.md updates;
  squashed into one focused commit before delivery (see "Gates and commit").
- 2026-09-16 real runs, all from the repository root:
  `python -m qecgen.residual.cli inventory --config examples/residual/<name>.json` ×5
  (device 0.95 s; Willow si1000 8.2 s incl. the live Zenodo range fetch; Willow RL 3.2 s;
  d9 10.5 s; d25 11.1 s);
  `python -m qecgen.residual.cli pilot --config …` ×5 (device 1.0 s; Willow 3.6 s each;
  d25 2 min 43 s; d9 2 min 9 s with `--edge-prototype-rows 10000`);
  `python -m qecgen.residual.cli build-all --config-dir examples/residual` (11 min 5 s);
  `python -m qecgen.residual.cli validate --output data/residual/<name>` ×5 (all ok).
