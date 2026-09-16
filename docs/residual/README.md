# Residual-error datasets (research pipeline)

`qecgen/residual/` builds datasets for testing a research hypothesis about two-stage
decoding: PyMatching predicts the logical observable, a small residual model predicts
whether that prediction is wrong, and the final decoder flips PyMatching's answer where
the residual model says so. **Nothing here claims the hypothesis holds.** The pipeline
produces reproducible datasets with which it can be tested, plus a sanity baseline that
reports weak, neutral or harmful outcomes as they are.

The build record — every source hash, measurement and decision — is
[`PROGRESS.md`](PROGRESS.md). This file documents the pipeline itself.

## Contents

1. [Vocabulary and claims](#1-vocabulary-and-claims)
2. [Commands](#2-commands)
3. [Configuration schema](#3-configuration-schema)
4. [Source identification](#4-source-identification)
5. [Decoder-model rules per source kind](#5-decoder-model-rules-per-source-kind)
6. [Time slices and graph node sets](#6-time-slices-and-graph-node-sets)
7. [Feature schema version 1](#7-feature-schema-version-1)
8. [Data partitions](#8-data-partitions)
9. [Output layout](#9-output-layout)
10. [Crash safety and resumability](#10-crash-safety-and-resumability)
11. [Validation](#11-validation)
12. [Sanity residual model](#12-sanity-residual-model)
13. [Limitations](#13-limitations)

## 1. Vocabulary and claims

- A **detector error model (DEM)** is a probabilistic model of which detectors and
  observables an error mechanism flips. It is not a per-shot answer key.
- `pm_guess` is PyMatching's predicted logical-observable flip, `truth` the sampled or
  recorded flip, `pm_wrong = int(pm_guess != truth)`.
- `pm_weight` is PyMatching's returned total solution weight: the **sum of matching-edge
  weights**, not a physical fault count.
- The artifacts are **not qecgen datasets**: no qecgen manifest, no exporter-registry
  entry, no Nexus compatibility claim (the Nexus input format is unknown; nothing in this
  package may imply otherwise).
- No column anywhere is a physical Pauli fault label (`DATA_CONTRACT.md`, Contract C).

## 2. Commands

Run from the repository root with `python -m` (the installed `qecgen` console script may
resolve a different checkout):

```bash
python -m qecgen.residual.cli inventory --config examples/residual/<name>.json
python -m qecgen.residual.cli pilot     --config examples/residual/<name>.json [--generated-rows N] [--resample-rows N] [--edge-prototype-rows N]
python -m qecgen.residual.cli build     --config examples/residual/<name>.json [--fresh] [--skip-pilot-gate] [--pilot-rows N]
python -m qecgen.residual.cli validate  --output data/residual/<name>
python -m qecgen.residual.cli build-all --config-dir examples/residual
python -m qecgen.residual.cli manifest  --output-root data/residual
```

- `inventory` resolves the source and the decoder model without decoding anything and
  writes `data/residual/.checkpoints/<name>/inventory.json` (identity table, decoder
  statistics, graph digest, every verification the source kind performs). A source that
  cannot be resolved is reported as blocked; no dataset directory is created.
- `pilot` decodes **all** existing source rows, prints `pm_wrong: k / n (fraction)`, the
  PyMatching logical error rate with its 95 % Clopper–Pearson interval, runs the alignment
  investigation (section 11), times sampling / decoding / feature extraction / writing on
  freshly generated rows, projects the full run and free disk, and records a
  `resources_sufficient` verdict in `.checkpoints/<name>/pilot.json`.
- `build` refuses to start unless a pilot record for the *same* configuration, source and
  decoder says `resources_sufficient: true` (it runs the pilot inline when the record is
  missing; `--skip-pilot-gate` exists for the test suite only). It then runs the three
  stages of section 10 and publishes the artifact set atomically. `--fresh` discards the
  dataset's checkpoints first.
- `validate` re-runs the fifteen named checks and the spot checks on a published
  directory and prints `ok: True (k spot rows)` or the failing checks.
- `build-all` builds every config in the directory, required datasets first and
  `"additional": true` datasets last, continues past a blocked or failed dataset (the
  reason is recorded in `.checkpoints/<name>/blocked.json`), and re-renders `MANIFEST.md`.

Every command prints its fully resolved configuration and the configuration hash first,
so a terminal log is a complete record of the run.

## 3. Configuration schema

The reviewed configurations live in `examples/residual/`. Paths are relative to the
repository root and are resolved to absolute paths at load time.

```json
{
  "version": 1,
  "dataset_name": "indep_d9_r200_p0005",
  "output_root": "data/residual",
  "additional": false,
  "source": {"kind": "legacy_ml_csv", "path": "data/….ml.csv", "expected_content_hash": "<blake2b-256 hex>"},
  "generation": {"mode": "extend", "shots": 304000, "seed": 0, "chunk_size": 16000, "require_source_prefix": true},
  "decoder": {"kind": "circuit_dem", "enable_correlations": false},
  "splits": {"method": "seeded_permutation", "seed": 20260915, "fractions": {"train": 0.7, "validation": 0.15, "test": 0.15}},
  "pipeline": {"checkpoint_rows": 16000, "feature_rows": 2000, "spot_check_rows": 8},
  "sanity_model": {"enabled": true, "seed": 0}
}
```

| block | keys and rules |
|---|---|
| `source.kind` | `legacy_ml_csv` (manifest v1, uniform circuit-level noise), `device_ml_csv` (manifest v2 device profile), `device_config` (a version-1 `generate-config` JSON with no existing rows; `fresh` mode only), `hardware_willow` (`table`, `circuit`, `expected {table_sha256, circuit_sha256, distance, basis, rounds, orientation}`, `formatted_prefix {path, expected_content_hash, offset}`, `zenodo {record, archive, archive_md5_published, cohort_prefix, cache_dir}`) |
| `generation.mode` | `extend` (existing rows are reproduced as an exact prefix, then the seeded stream continues), `source_rows` (recorded rows only; hardware), `fresh` (no existing rows). `extend` requires `shots`, `seed`, `chunk_size` and, when `require_source_prefix` is true, the call-size-sequence rule of section 5 |
| `decoder.kind` | `circuit_dem` (legacy), `static_profile_dem` / `frozen_reference_dem` (device), `official_dem` (`member`, `expected_sha256`; Willow). `enable_correlations` must be `false` in schema version 1: a correlated-matching run is a separate named configuration, never mixed |
| `splits` | `seeded_permutation` (requires `seed`) or `contiguous_blocks` (forbids it); fractions over `calibration`, `train`, `validation`, `test`, each > 0, summing to 1 |
| `pipeline` | `checkpoint_rows` must be a positive multiple of `generation.chunk_size` (every checkpoint holds whole `sample()` calls); `feature_rows` bounds the unpacked block; `spot_check_rows ≥ 5` |

The **configuration hash** is the sha256 of the canonical JSON form (sorted keys,
repository-relative POSIX paths, `output_root` excluded), so it is identical across output
roots and checkouts. It is written into `<name>_resolved_config.json`, every checkpoint,
the raw HDF5 attributes and the note, and validation recomputes it.

## 4. Source identification

Sources are identified by content, never by filename:

- `.ml.csv` files carry **no** in-band manifest lines; the manifest is the
  `<stem>.ml.manifest.json` sidecar. The pipeline streams the table row by row
  (`shot` must equal the row index; cells are literal `0`/`1`), packs little-endian, and
  refuses the file unless the recomputed qecgen `content_hash` (BLAKE2b-256) equals the
  sidecar's and the configured `expected_content_hash`.
- Legacy sources are rebuilt through `qecgen.environments.build_environment` and the
  rebuilt channel vector must equal the recorded one; device sources through
  `qecgen.configuration.ideal_circuit` + `qecgen.noise.build_noisy_circuit`, and the noisy
  circuit's sha256 must equal the manifest's `generation_audit.circuit_sha256`.
- Willow sources go through `qecgen.hardware.load_willow_derived` (sha256 of table and
  circuit, identity columns, row order), the formatted 2,000-row file is compared bit for
  bit with the cohort rows at the recorded offset, and the official Zenodo archive members
  are fetched and compared (section 5).
- Every source must have exactly one logical observable. A multi-observable source stops
  with `MultiObservableError`, whose message proposes an explicit multi-observable schema
  instead of reducing the array to one bit.

## 5. Decoder-model rules per source kind

One frozen `pymatching.Matching` per dataset, built with
`Matching.from_detector_error_model(dem)` on a real `stim.DetectorErrorModel`, standard
matching (`enable_correlations=False`), never from qecgen's parsed `H` matrix. The DEM is
serialised first and the matcher is built from the parsed text, so the published
`<name>_decoder.dem` is bit-for-bit the model that produced the labels (a DEM re-parsed
from its own text is not equal to the in-memory object at the 1e-18 level; the text is
the model).

| source kind | decoder model |
|---|---|
| legacy (independent Stim noise) | `circuit.detector_error_model(decompose_errors=True)` of the rebuilt circuit — an exact model of the sampled noise |
| static device profile | the decomposed DEM of the exact noisy circuit the shots were sampled from; whether correlated mechanisms decomposed is recorded (`dem_stats`) |
| dynamic device profile | **no exact DEM exists.** One frozen reference model for the whole dataset: the static blocks of the profile are kept, `drift`, `bursts` and `leakage` are disabled, and the drift stationary point (`baseline_probability`) is represented as static single-qubit spatial terms; bursts and leakage have no static representation and are omitted, which the metadata states. Hidden drift, burst and leakage state is never exposed to the decoder, no per-shot or per-chunk oracle DEM is built, and the generated noise is not changed. The transformation text and both profile hashes are recorded. (Supported and tested; no deliverable uses it.) |
| Google Willow | the official `error_model.dem` shipped in the Zenodo record 13273331 archive, fetched by HTTPS range requests against the ZIP64 central directory (no multi-gigabyte download); the receipts record the record metadata, archive size and published MD5 (not re-verified), and per member the path, sizes, CRC32 from the central directory, sha256 and fetch time. The local mirror's detectors and observables must equal the archive's `detection_events.b8` / `obs_flips_actual.b8` for every row, the circuit must be byte-identical, `metadata.json` and the `QUBIT_COORDS` partition must agree. Google's own predicted flips are fetched only as a diagnostic (agreement rate, their pathway's error rate) and are never a feature or target. |

**Extension prefix rule.** An `extend` run is admissible only when the source's
`sample()` call-size sequence is a prefix of the new run's — e.g. a source written as one
`sample(16000)` call is reproduced by a new run whose `chunk_size` is 16,000, whatever
`chunk_size` the source recorded. The pipeline then hashes the first `source_shots`
generated rows with qecgen's streaming content hasher and refuses to publish unless the
digest equals the source `content_hash`; validation re-checks it from the raw HDF5.
Dynamic profiles are refused in `extend` mode.

The decoder metadata records `fitted_in_this_pipeline: false` for every kind (this
pipeline estimates no matching weights from data) and, for third-party priors, a
`third_party_fitting` block naming the method, objective, data and the unknown overlap
with the cohort.

## 6. Time slices and graph node sets

**Time slices.** A detector's slice is the rank of its *latest* time coordinate among the
sorted distinct latest-time values, computed with `qecgen.hardware.detector_anchors`
(the unique earliest-time position identifies the stabiliser; the latest time the slice).
Stim-generated circuits carry one `(x, y, t)` triple per detector; the Willow circuit
carries one triple in the first slice, two in the bulk and three to five in the final
slice, and the latest time is used. The number of slices is `rounds + 1` for these
memory experiments (an initial and a terminal detector layer exist), and per-slice sizes
are recorded in the decoder metadata.

**Graph node sets are derived from the decomposed DEM, not from `matching.edges()`.**
PyMatching 2.4.0 merges parallel edges and keeps only the *first-inserted* edge's fault
ids, so `edges()` cannot preserve fault-identifier information separately and its logical
markings depend on DEM instruction order. Instead every graphlike component (≤ 2
detectors, split at `^` separators, ids XOR-reduced) becomes an `EdgeRecord(u, v|None,
observables, probability)`:

- boundary-adjacent nodes = endpoints of one-detector components;
- logical-edge-adjacent nodes = endpoints of components whose observable set contains 0;
- adjacency = unordered two-detector pairs, parallel components collapsed, no self loops;
- `n_conflicting_pairs` counts pairs or boundary nodes whose components disagree on the
  observable set; a DEM with any such pair is **refused** (all five built DEMs have zero);
- the pair and boundary sets from `matching.edges()` must equal the DEM-derived ones
  (asserted), and its per-edge weights are recorded for provenance.

The build fails if the logical-edge-adjacent set is empty.

## 7. Feature schema version 1

One CSV row per shot, columns in exactly this order. Integer columns are written as
integers; floats with shortest round-trip `repr`.

| column | definition |
|---|---|
| `n_fired_total` | fired detectors |
| `frac_fired` | `n_fired_total / n_detectors` |
| `n_fired_first_round`, `frac_fired_first_round` | fired detectors in the earliest slice; divided by that slice's size |
| `n_fired_final_round`, `frac_fired_final_round` | same for the latest slice |
| `fired_per_round_mean`, `fired_per_round_max`, `fired_per_round_std`, `fired_per_round_range` | over per-slice counts; `std` is the population value (`ddof=0`) |
| `round_of_max_fired` | zero-based index of the earliest maximal slice, divided by `max(n_slices − 1, 1)` |
| `n_active_rounds`, `frac_active_rounds` | slices with ≥ 1 fired detector; divided by `n_slices` |
| `fired_round_center` | fired-count-weighted mean of the normalised slice index (`0.0` when nothing fires) |
| `fired_round_spread` | fired-count-weighted population standard deviation of the normalised slice index (`0.0` when nothing fires; ≤ 0.5) |
| `n_fired_boundary_adjacent`, `frac_fired_boundary_adjacent` | fired nodes in the boundary-adjacent set; divided by that set's size |
| `n_fired_logical_edge_adjacent`, `frac_fired_logical_edge_adjacent` | fired nodes in the logical-edge-adjacent set; divided by that set's size |
| `n_fired_neighbor_pairs` | unordered non-boundary adjacency pairs with both endpoints fired |
| `n_fired_isolated` | fired nodes with no fired neighbour in the non-boundary graph |
| `pm_weight`, `pm_weight_per_fired` | PyMatching's solution weight; divided by `max(n_fired_total, 1)` |
| `pm_guess` | PyMatching's predicted flip |

followed by the label/metadata columns `truth`, `pm_wrong`, `run_id`, `split`, which are
**never** residual-model inputs. `features.FEATURE_COLUMNS` is the single source of the
input list; `audit_feature_columns` refuses any CSV whose header differs in name or order,
and `FeatureContext` (detector count, time slices, graph summary — nothing else) is the
only input feature extraction receives. `truth` and `pm_wrong` are computed after the
features and PyMatching's outputs exist. A pure-Python reference extractor exists for the
spot checks. The optional matching-solution edge features (`pm_n_edges`, …) are **not**
in version 1; the prototype measurement and the decision are in `PROGRESS.md`.

## 8. Data partitions

Every row gets a deterministic `split` before any model is fitted. Codes:
`calibration 0`, `train 1`, `validation 2`, `test 3` (the CSV carries the names, the HDF5
the codes, and the mapping is an HDF5 attribute).

- Independent simulations: a seeded permutation of `run_id` (`np.random.default_rng(
  SeedSequence(seed))`), boundaries at `round(cumulative fraction × n_rows)` with the last
  boundary forced to `n_rows`.
- Hardware: contiguous blocks in source row order. Willow acquisition chronology is
  unknown, so source row order is a grouping convention and is stated as unverified.
- `calibration` is reserved for a data-fitted matching model; no built dataset uses it
  because no matching weights are estimated here.

Nothing is fitted on validation or test rows: not a matching graph, not a scaler, not a
feature transform, not a threshold, not a residual model.

## 9. Output layout

```
data/residual/                                 (ignored by git)
  MANIFEST.md                                  one row per dataset, blocked/failed rows included
  <name>/
    <name>_features.csv                        section 7
    <name>_raw.h5                              group /residual: detectors uint8 (N, ceil(D/8)),
                                               observables uint8 (N, ceil(O/8)), run_id int64, split int8
                                               root attrs: format, format_version, n_detectors, n_observables,
                                               bit_order="little", source_content_hash, config_hash,
                                               decoder_dem_sha256, feature_schema_version, split_codes, versions
    <name>_note.md                             every field the brief lists, plus the summary digest
    <name>_validation.json                     section 11
    <name>_resolved_config.json                absolute paths + config_hash (self-verifying)
    <name>_decoder.dem                         the exact model text (for Willow, the archive member bytes)
    <name>_decoder_metadata.json               kind, hashes, DEM statistics, graph sizes, slices, provenance
    <name>_summary.json                        the numbers the note and MANIFEST are rendered from
    <name>_sanity.json                         section 12 (or a recorded skip reason)
  .checkpoints/<name>/                         inventory.json, pilot.json, checkpoint.json, *.chk
  sources/zenodo-13273331/                     fetched Willow members + receipts.json
```

The raw arrays live under the `/residual` group rather than at the file root on purpose:
qecgen classifies an HDF5 with a root `detectors` dataset and no manifest as an
interrupted generation run, whereas this layout reads as "not a qecgen dataset".
`run_id` is identical between the CSV and the HDF5 and equals the row index.

## 10. Crash safety and resumability

Nothing is ever appended to a final file. A build runs three stages:

- **Stage A** writes raw checkpoint chunks (`raw_chunk_NNNNN.chk`, npz format written
  through an open file handle then `os.replace`) from the source rows and/or the seeded
  sampling stream; the extension prefix hash is checked when the source rows are reached.
- **Stage B** decodes each raw chunk in `feature_rows` blocks, extracts features, then
  computes `truth` and `pm_wrong`, and writes `feat_chunk_NNNNN.chk`.
- **Stage C** assigns splits, assembles every artifact inside a `qecgen.run.staged()`
  scratch directory (CSV and HDF5 streamed from the chunks; decoder files; resolved
  config; summary; validation run on the scratch copy; sanity model; note), and commits
  the set with qecgen's two-phase move. `MANIFEST.md` is rendered the same way. A crash
  can never leave a partial file with a final name.

`checkpoint.json` records the identity (configuration hash, source hash, decoder DEM
sha256, graph digest, schema version, software versions, seed, chunk size, shots), every
completed chunk's row range, row count and sha256, and partial aggregates. On restart the
identity is compared field by field and any difference refuses the resume; chunk files
are hash-verified before use; only a contiguous, non-overlapping prefix of completed
chunks is reused. The seeded stream is replayed from row 0 (chunk k cannot be sampled
without sampling 0..k−1); only the writes are skipped.

## 11. Validation

`validate` runs fifteen named checks covering the brief's fourteen assertions:
`row_count_csv_equals_raw`, `run_id_unique_contiguous_identical`,
`raw_widths_match_true_widths`, `bit_packing_little_endian`, `binary_columns`,
`pm_wrong_consistent`, `batch_vs_single_decode`, `no_nan_inf`, `schema_exact`,
`fractions_in_range`, `note_matches_data`, `checksums_match`, `source_prefix_matches`,
`feature_inputs_audit`, `no_fit_on_held_out`. Spot checks decode at least five random raw
rows one at a time with `matching.decode`, recompute every feature with the pure-Python
reference extractor, and compare with the CSV; the little-endian check additionally
shows, on a row with an asymmetric byte, that big-endian unpacking does **not** reproduce
the published guess, weight and features. `checksums_match` re-streams the source file
and recomputes its content hash; `note_matches_data` parses the rendered note's labelled
numbers and its `summary_sha256`.

The **pilot alignment investigation** (recorded in `pilot.json` and `PROGRESS.md`):
(1) self-consistency with `qecgen.qa.decode_stored_shots` — labelled as sharing the
construction path, so it cannot detect misalignment; (2) negative controls: a seeded
random permutation of detector columns and big-endian unpacking, whose error rates must
exceed the aligned rate's interval (they approach 50 % for large syndromes; d=3 syndromes
are sparse and land lower); (3) per-detector empirical firing rate vs the DEM-predicted
marginal `(1 − ∏(1 − 2p_j)) / 2` (Pearson r, max absolute difference); (4) a fresh
re-sample from `derive_seeds(seed, 2)[1]` whose interval must overlap the stored rows';
(5) the stored detection-event rate vs a fresh circuit sample. For Willow, (2) and (3) are
run with the shipped DEM, which is the structural ordering evidence beyond byte equality
with the archive; a plausible error rate alone is only a diagnostic.

## 12. Sanity residual model

A validation experiment, not the deliverable. Two scikit-learn baselines (`residual`
optional-dependency group, pinned) predict `pm_wrong` from the 24 input columns:
`StandardScaler` + `LogisticRegression(class_weight="balanced")` and
`HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False)`. Fitting
uses **train** rows only; the decision threshold is chosen on **validation** rows to
minimise the corrected logical error rate, with "flip nothing" (`+inf`) as a candidate
so an uninformative model selects no flips; evaluation happens **once** on test rows.
Reported per model: positive prevalence, always-zero accuracy, balanced accuracy, ROC
AUC, precision–recall AUC, precision and recall for `pm_wrong == 1`, the confusion
matrix, PyMatching's test logical error rate, the corrected decoder's test rate
(`corrected_guess = pm_guess XOR predicted_pm_wrong`) with its 95 % interval, the
absolute and relative change, the numbers of correcting and harmful flips and an exact
paired (McNemar) p-value. The practical criterion is a lower held-out logical error rate
after both corrected failures and newly introduced flips are counted.

## 13. Limitations

- **d=25/r=25 at p=0.005** has almost no PyMatching failures; the dataset is extremely
  imbalanced and residual learning is likely untestable there. This is reported, not
  adjusted.
- **d=9/r=200** sits at the edge of the 6–16 % expectation band; that is consistent with
  200 rounds of error accumulation and is verified by the alignment investigation.
- **Willow chronology** is unknown; splits are contiguous blocks of source rows.
- **The shipped si1000 DEM** is not reproducible from the shipped `circuit_noisy_si1000.stim`
  (its construction is the publisher's) and is a correlated-matching prior decoded here
  with standard matching. **The RL-optimised prior** was tuned by Google for logical
  error rate on same-device 13-cycle data; overlap with the 10-cycle cohort is unknown, so
  that dataset is labelled additional, not an independent baseline.
- For the d=3 datasets every node is boundary-adjacent, so
  `frac_fired_boundary_adjacent` duplicates `frac_fired` there.
- Hardware and dynamic sources carry no exact DEM claim; the archive-level MD5 of the
  Zenodo zip is not re-verified (only the fetched members are, by CRC32 and sha256).
- No Nexus compatibility is claimed; these are residual feature CSVs.
