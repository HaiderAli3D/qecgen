# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## What this is

`qecgen` generates surface-code QEC datasets for decoder benchmarking: Stim builds the
circuit, shots are sampled in chunks, the detector error model is parsed into sparse
matrices, the result is validated and written through a pluggable exporter registry.

**Read `DATA_CONTRACT.md` before touching anything that produces or labels shots.** Every
file this tool writes maps per-shot *detection events* to per-shot *logical observable
flips* (Contract A), optionally plus *which DEM mechanisms fired* (Contract B, behind
`--emit-mechanisms`). Physical Pauli fault labels (Contract C) are not implemented and are
underdetermined as specified — do not add them; establish which of A/B the requester
actually wants. Never describe output as "physical-error-labelled" anywhere.

`correction.py` is **not** Contract C and must not be removed as if it were. Contract C
inverts a many-to-one map (*given a syndrome, which fault?*) and stays refused. Scoring
runs that map forward (*given a correction, what was its logical effect?*): single-valued
for every input, no tie-breaking, no ground-truth fault label — the correction is an
input, not a target. That is the client brief's "apply and measure", done exactly.

`README.md` documents the user-facing surface and the empirical measurements behind the
design decisions below. `GUIDE.md` is the task-oriented walkthrough of that same surface.
`docs/REALISM.md` covers the device-noise, hardware-import and decoder-transfer path,
and `research/realism/EVIDENCE.md` holds the measurements behind it.

## Commands

Python 3.13+, dependencies pinned exactly (dataset reproducibility depends on the Stim
version). Installed globally on this machine — no venv activation needed.

```bash
pip install -e ".[dev]"                 # runtime + pytest, ruff, mypy, httpx
pip install -e ".[ui]"                  # optional: fastapi, uvicorn for `qecgen ui`
pip install -e ".[decoders]"            # optional: mwpf, fusion-blossom for `sweep --decoder`
pip install -e ".[research]"            # optional: torch, for research/realism

ruff check . && ruff format --check .
mypy --strict qecgen tests
pytest -m "not slow"                    # the fast structural suite
pytest -m slow                          # the slow statistical / end-to-end suite
pytest -m "not requires_mwpf"           # skip what needs the `decoders` extra
pytest tests/test_dem.py::TestName::test_name   # single test
pytest -k xz_bias -v                            # by keyword
pytest research/realism/tests           # NOT collected by a bare `pytest`

cd frontend && npm ci && npm run build  # into qecgen/ui/static (gitignored);
                                        # `build` runs `tsc --noEmit` first
cd frontend && npm run typecheck        # that gate alone, no bundle
node --test frontend/tests/configuration.test.mjs   # the frontend unit suite

python docs/make_diagrams.py            # redraw the README SVGs
python docs/make_sweep_plot.py          # re-plot the threshold PNG from docs/evidence/
python docs/make_realism_report.py      # re-plot the realism figures from the same directory
```

Nothing enforces any of this. There is no `.github/`, no CI and no pre-commit hook:
these commands are the only gates the project has, and three of them are narrower than
they look. `testpaths = ["tests"]`, so a bare `pytest` never reaches
`research/realism/tests` — a change under `research/` passes a green suite that ran none
of its own tests. `mypy --strict` is scoped to `qecgen tests`, so `research/` and `docs/`
are ruff-clean (`extend-exclude` is `frontend` alone) but carry no strict-mode guarantee
at all. And `frontend/tests/configuration.test.mjs` has no `npm test` script to run it,
which is why every other doc here calls `npm run typecheck` the frontend gate; name that
file rather than its directory, because `node --test frontend/tests/` resolves the
directory as a module and fails outright on Node 22.

The six-lesson teaching site lives in its own repository,
[qecgen-learn](https://github.com/HaiderAli3D/qecgen-learn). Its lesson copy and glossary
state this repo's traps and conventions, so doc corrections here must be swept there too.

The CLI installs as `qecgen` (also runnable as `python -m qecgen.cli`):
`generate`, `generate-config`, `multi-env`, `drift`, `sweep`, `validate [--qa]`,
`score`, `benchmark`, `inspect`, `formats`, `delete`, `ui`. Every command prints its
fully resolved config before doing work, so a terminal log is a complete record of the
run. `data/`, `out/`, `runs/` and all dataset extensions are gitignored. `*.csv` is
among them, negated by `!docs/evidence/*.csv` for the committed sweep evidence a README
figure is built from.

**Verify a change with `python -m qecgen.cli` from the repo root, not with `qecgen`.** An
editable install resolves to whichever checkout it was made from, which need not be this
one — measured here: the console script loads a different tree while looking identical in
the terminal, config table included. `python -m` puts the working directory first on
`sys.path`, so it serves the code you just edited. `run-ui.cmd` (repo root, tracked)
launches the UI the same way, pinning cwd and `PYTHONPATH` to `%~dp0` so it serves the
checkout it sits in by construction rather than by luck.

`qecgen ui` serves the web UI for **every** command on loopback: generation, sweeps,
scoring, QA, provenance and the registry. The frontend is built on demand — the command
names the build line rather than serving a blank page.

Frontend iteration is two processes, not a rebuild loop: `qecgen ui --dev` serves the API
on 8765 and is the only mode that emits CORS, for exactly the Vite origins; `npm run dev`
serves the pages on 5173 and proxies `/api` across. Without `--dev` the browser request is
refused outright, not merely unstyled.

## Architecture

Dependencies flow one way; there are no cycles.

```
circuits.py    NoiseModel -> ChannelVector -> stim.Circuit.generated (+ apply_xz_bias rewrite)
sampling.py    chunked shot generation; the ONLY place sample() is called
dem.py         stim DEM -> DemStructure (sparse H/L, priors, components, coords)
correction.py  logical operator extraction; scores a SUPPLIED Pauli correction (not Contract C)
decoders.py    sinter decoder-name resolution + backend availability; implements no decoder
dataset.py     canonical model: EnvironmentSpec, DatasetMeta, InMemoryDataset,
               Reader/StreamingWriter protocols, content hashing
environments.py orchestration: build_single/multi/drift, stream_single_environment,
               seed derivation, drift axes, drift_dataset_names
configuration.py normalizes version-1 config JSON for the legacy/device/hardware runs;
               manifest version 2 carries generation_config and generation_audit
noise.py       explicit static channel construction for configured runs
hardware.py    validates supported source identities for hardware imports
exporters/     Exporter protocol + registry (hdf5, npz, parquet, jsonl, csv,
               ml_csv), bit_columns.py holds the one-column-per-bit encoding
               shared by the two CSV formats,
               infer_format; structure_json.py holds the normative structure encoding
               shared byte-for-byte by jsonl and csv
deletion.py    what one artifact IS on disk, and removing it to the OS recycle bin.
               plan_deletion names the whole set (a format's companions, a sweep's triple,
               a drift directory) and refuses reserved paths; execute reports one outcome
               per file. Front-end agnostic: `qecgen delete` and the UI's delete routes
               both call it, so neither re-derives which files travel together
run.py         one job end to end. RunSpec produces a dataset; AnalysisSpec (sweep,
               score, qa, benchmark) reads what exists and reports. `run` and `analyse` dispatch,
               `job_total` says what a progress bar counts, `resolved_config` is the
               record both front ends print. Imports neither typer nor pydantic, and
               imports qecgen.sweep only inside the sweep branch
validate.py    fast deterministic structural checks (default)
qa.py          slow statistical checks with Clopper-Pearson intervals (opt-in);
               also decode_stored_shots/benchmark_dataset, the decoder baseline
               over a file's own shots -- the one place a matcher is built
sweep.py       sinter threshold sweeps -> CSV + plot (independent of the dataset path)
cli.py         typer commands, config printing, progress
ui/            local web UI. protocol.py (wire format) + worker.py (child process)
               depend only on stdlib and run.py; jobs.py supervises the children;
               datasets.py browses and resolves paths under the data root; sweeps.py does
               the same for sweeps, keyed on the .threshold.json sidecar;
               configured.py serves the device and hardware routes; settings.py
               and schemas.py hold config and request models; app.py is the only
               module that imports FastAPI
frontend/      Vite + React source; builds into qecgen/ui/static
docs/          README figures + the scripts that regenerate them; imports qecgen.sweep,
               and nothing imports it
examples/      the committed version-1 configs -- legacy, device-static,
               device-dynamic, willow-import -- that `generate-config` takes as input
research/      a separate tracked package (research/realism) with its own test suite
               and evidence docs; excluded from the wheel by packages.find
               include=["qecgen*"], never imported by qecgen, and never collected by a
               bare pytest
```

`environments.py` is the orchestration layer — most feature work lands there or in
`exporters/`. `run.py` is what a front end calls; `cli.py` and `ui/` hold no domain logic
beyond argument resolution. Anything both front ends would otherwise duplicate belongs in
`run.py`, not in one of them.

## Invariants that produce silently wrong data if broken

Each of these was established empirically and is load-bearing. Breaking one yields a
well-formed file containing wrong data, which passes casual inspection.

- **Little-endian bit packing, everywhere.** NumPy's `packbits`/`unpackbits` default to
  `bitorder='big'`, the opposite of Stim/PyMatching. Use
  `qecgen.sampling.unpack_bits(...)` (and `qecgen.correction.pack_correction(...)` for
  corrections); if you must call NumPy directly, pass `bitorder="little"` explicitly.
  Every manifest records `bit_order`.
- **Packed width never implies true width.** 3 bytes could be 17–24 detectors.
  `n_detectors` / `n_observables` / `n_mechanisms` are explicit manifest fields.
- **One `H` column per `error(...)` instruction, not per graphlike component.** Stim's
  decomposed DEM joins components with `^`; they share one probability. Components are
  preserved separately keyed by `parent_mechanism_id`.
- **The ≤2-detector weight bound holds on components, not mechanisms.** Mechanism weights
  legitimately run 1–4. `validate.py` asserts on components and only *reports* the
  mechanism histogram; asserting the latter fails on correct data.
- **No top-level `p`, `circuit` or `dem`.** They are per-`EnvironmentSpec` properties; a
  single-environment dataset is a list of length one. Adding a dataset-level `p` breaks
  the moment there are two environments.
- **The manifest is decoder-visible; `provenance/` is not.** Circuit and DEM *text* live
  only in the provenance block (written at `--structure full`, stored physically apart).
  Under `FROZEN_PRIOR` that text is exactly what the condition withholds. Never move
  circuit/DEM text into `DatasetMeta.to_json_dict()`.
- **The manifest's `schema` block is derived at serialisation, never a stored field.** It
  names which arrays are features and which are targets, because a consumer could not tell
  and the answer lived only in `DATA_CONTRACT.md`. `jsonl` and `parquet` serialise
  `dataclasses.replace(meta, structure_level=recorded_structure_level(...))` to record a
  downgrade, so a stored block would be built *before* that replace and keep advertising a
  DEM the file no longer carries — an over-claim in the one field a reader cannot check
  against the file. Computing it inside `to_json_dict()` makes that impossible rather than
  merely tested. It names array *roles*, not column names: `csv` writes `det_0` unpadded
  and a dataframe-facing format would pad, and one field cannot honestly name both.
  `primary_target` exists so `targets[-1]` is never how the benchmark target is picked —
  under Contract B that expression selects the mechanism labels. No role names a physical
  fault, not even as `"absent"`, which reads as "coming soon" for a target that is refused.
- **`content_hash`'s array names are the digest's alphabet, not a naming registry.**
  `content_hash` and `StreamingContentHasher` fold the literal strings `detectors`,
  `observables`, `environment_ids`, `mechanisms` into the digest and take no
  `DatasetMeta`. The `schema` block contains names that look like them and one that
  deliberately differs — `environment_id`, singular, because that is the *column* name — so
  a "one source of truth for array names" refactor wiring the digest through the block
  would silently rehash every dataset ever produced. Both sites carry a comment; keep them.
- **`full` means full, or the manifest says otherwise.** `hdf5`, `npz`, `csv` and
  `ml_csv` carry the
  provenance text; `jsonl` and `parquet` decline it and therefore must not record
  `structure_level: full`. JSONL's refusal is deliberate and load-bearing — the idiomatic
  reader is `for line in f: json.loads(line)`, one loop from handing a frozen-prior test
  file's own DEM to a decoder — but it recorded `full` anyway for a while, which is an
  over-claim in the one field a reader cannot check against the file. `carries_provenance`
  plus `recorded_structure_level` make it a registry-wide invariant with one parametrised
  test, not five independent conventions. CSV is safe to carry it because a reader that
  does not filter `#` lines never finds the table at all.
- **Column order for the tabular formats has one source, `bit_columns.COLUMN_ORDER`.** It
  used to live in four places per format — the header builder, the row loop in `write`, the
  read offsets and a copy in the tests — agreeing only by convention, and `write` never
  consulted the header it had just emitted. A partial edit therefore stored detector bits
  under the target's column name *while the round trip stayed green*, because both sides
  shared the header builder and every guard in those modules is a format guard rather than
  an order guard: `bits_from_cells` compares cells literally against `"0"`/`"1"`, so one
  bit is indistinguishable from another. `test_a_rows_cells_land_under_their_own_header`
  is the check that closes it — it reads the table by column *name* and compares against
  the in-memory arrays, never against anything the reader produced. Locate a block by name
  or by `block_slices`, never by a literal index; two `columns[1] == ENVIRONMENT_COLUMN`
  tests survived into the reorder and had to be found by failing tests.
- **A CSV dataset's `shot` column must equal its row index**, in `csv` and `ml_csv`
  alike -- the check is shared in `bit_columns.require_row_in_order`, and it matters
  more in `ml_csv`, whose audience is tools that reorder rows by default. This is the one format users
  open in a spreadsheet, and sorting is the one thing a spreadsheet makes trivial — it
  severs the correspondence between a shot's detectors, its `environment_id` and its
  mechanism labels while leaving a file that still parses. `read` refuses a row out of
  order. For the same reason bit cells are compared literally against `"0"`/`"1"`: Excel
  writes `TRUE`/`FALSE` for a boolean-formatted column, and folding an empty cell to `0`
  would invent a shot with no detection events.
- **`FROZEN_PRIOR` vs `ORACLE_CALIBRATED` is stated, never inferred** from whichever DEM
  happened to be in scope. `structure_source_environment_id` and `structure_dem_sha` make
  it auditable.
- **`--emit-mechanisms` switches all three arrays to the DEM sampler.** Sampling circuit
  and DEM separately gives two RNG streams, so labels would not explain the events beside
  them.
- **`chunk_size` is part of the reproducibility contract** and is recorded in every
  manifest: it changes the sequence of `sample()` calls and therefore the sample stream.
  Determinism is asserted on array contents and `content_hash` (BLAKE2b-256, named in
  `content_hash_algorithm`), never on byte-identical files.
- **Seeds are threaded explicitly** via `numpy.random.SeedSequence` spawning. No global
  RNG state is touched anywhere.
- **`validate.py` and `qa.py` stay separate.** Deterministic structural checks must never
  fail for sampling reasons. QA compares Clopper-Pearson intervals, never point estimates,
  and *reports* the threshold crossing rather than asserting a hardcoded value.
- **The PyMatching oracle is built with `Matching.from_detector_error_model(dem)`** on the
  original DEM object — never reconstructed from our own `H`, which would validate the
  parser against itself.
- **A correction oracle must inject `X_ERROR(1)`, never `X`.**
  `compile_detector_sampler` defines detectors relative to a reference sample, so a
  deterministic Clifford folds into the reference and the observable does not move.
  Measured on d=3 rotated memory-Z: `X 1` gives observable 0 and zero detection events;
  `X_ERROR(1) 1` gives observable 1 and one event. An oracle built on `X` passes every
  test while measuring nothing.
- **No front end writes to the path a user will read.** Every exporter writes in place and
  truncates its target on open, and `StreamingHDF5Writer` opens the destination directly,
  so an interrupted run destroys whatever was already there and leaves a file named like a
  finished dataset. Go through `run.staged()`, which commits with `os.replace` from a
  sibling directory. A `.partial` *suffix* does not work — `NPZExporter` rewrites any path
  whose suffix is not `.npz`. Truncated JSONL and CSV are the nastiest cases: both put the
  manifest in the header, so a cut-short file still reads and only `validate_dataset`
  notices the missing rows.
  The commit itself is two-phase: a bare `os.replace` loop is not all-or-nothing, and a
  destination file held open mid-loop (the Windows failure mode) left a mixed old/new
  drift set while the cleanup destroyed the rest of the staged files. Files about to be
  overwritten are displaced into a backup first and restored on failure; a backup whose
  restore also fails is salvaged to a `.qecgen-displaced-*` sibling, never deleted. Every
  staging directory also holds an advisory lock (`.qecgen-lock`) for the lifetime of its
  run — `sweep_partials` probes it and skips live directories, because the UI sweeps at
  startup while a CLI run may be mid-write into the same data root. Sweep outputs (results table,
  plot, threshold JSON) go through `staged()` too — and note that the results table is a
  `.csv`, which is now **also** a dataset extension. It is not a dataset: it has no
  `#__manifest__` header, the CSV reader refuses it with `NotAQecgenDatasetError`, and
  `ui/datasets.list_datasets` lists it as `not_a_dataset` rather than `unreadable`, so an
  intact results table never wears a corruption flag.
- **A format's files are deleted as a set, or not at all.** `staged()` commits every file
  a format writes in one two-phase move, and that atomicity is what makes `ml_csv`'s manifest
  sidecar a *proof* rather than a heuristic: a table without its sidecar is a state this tool
  cannot publish, so it can only be a foreign file. Deleting one member manufactures exactly
  that — the table then reads as `NotAQecgenDatasetError`, i.e. as somebody else's CSV, and
  what it was is unrecoverable from what is left. The same applies to a sweep's
  `.csv`/`.png`/`.threshold.json` triple, which `ui/sweeps.py` keys on the sidecar (remove the
  results table alone and the sweep is still listed with nothing to draw), and to a drift
  directory, which `generate_drift` commits whole so a mixed set cannot exist.
  `deletion.plan_deletion` resolves the set from the registry and `execute` removes the
  **anchor first**, aborting the rest if it fails — orphaned bytes are untidy, but a manifest
  whose table is gone is *false*. A new multi-file format that does not declare `companions()`
  reintroduces the hole silently, because nothing errors.
- **A `send2trash` success does not mean recoverable.** Windows permanently deletes a file too
  large for the recycle bin under `FOF_ALLOWUNDO|FOF_NOCONFIRMATION` and returns success
  either way, as does any volume with no bin at all; qecgen writes multi-gigabyte datasets, so
  that is the ordinary path and not the edge case. The bin's capacity is not reliably readable
  (absent registry value, undocumented default, group policy), so nothing predicts it. No
  `Outcome` member is named `recycled`, `REMOVED` is verified with `os.path.lexists` *after*
  the call rather than inferred from it returning, and neither front end may use the word
  "recoverable" — `RECYCLE_CAVEAT` states the caveat unconditionally instead of above a size
  threshold, because a caveat that appears only sometimes teaches the reader its absence means
  safe.
- **`git_commit()` must never use pipes.** It is a `default_factory` on every
  `DatasetMeta`, so it runs on the path of every generated file. With
  `capture_output=True`, a timeout kills git but then joins the pipe reader threads, and a
  git helper that inherited the handle keeps them waiting on an EOF that never comes —
  `timeout` stops bounding anything and generation hangs inside its own constructor.
  Observed with py-spy on a run frozen at 1000 of 5000 shots. It writes to a temp file and
  is memoised per working directory; keep both.
- **The final data layer is the trailing run of *non-resetting* measurements.** Data
  qubits are never reset at the end of a memory experiment; ancillas always are. A rule
  that only skips annotations merges the `MR` ancilla layer in and reports 17 data qubits
  for rotated d=3 instead of 9 — and the scoring tests still pass, because ancilla entries
  contribute nothing to the observable.
- **A sweep's progress is denominated in tasks, and the record says so.** `max_errors`
  stops a sweep and `max_shots` is only a ceiling, so its shot total is unknowable before
  it runs. `JobRecord` carries `total_units`/`completed_units` plus `progress_unit`; the
  fields were once `total_shots`/`completed_shots`, and keeping those names while counting
  tasks would be a field lying about its own contents. `load_history` still reads the old
  names off disk, so pre-rename run records survive a restart. Completed tasks are derived
  in `sweep._progress_adapter` by mirroring sinter's own stopping rule (accumulated shots
  ≥ `max_shots` or errors ≥ `max_errors`), because sinter exposes no per-task "done"
  signal to a progress callback; sinter is exact-pinned, and a stale rule costs a slightly
  wrong bar, never a wrong number.
- **The worker moves the control pipe off fd 0 before anything spawns.** A thread parked
  in a blocking `os.read` on **descriptor 0 specifically** wedges a sweep on Windows, and
  it wedges it two ways: `multiprocessing`'s spawn handshake never completes (sinter's
  children reach ~9 MB with one thread and no Python frame while the parent waits forever
  in `_compute_task_ids`), and a native extension load parks in `create_module` for
  `scipy.linalg.blas`. Both were found with `py-spy dump` — identical stacks twenty
  seconds apart, not slow I/O. Measured across four variants of one collection: open pipe
  with no reader finishes in 1.2 s, a reader on fd 0 never finishes, a reader on an
  `os.dup` with fd 0 pointed at devnull finishes in 1.2 s. `_detach_control_channel` does
  the latter, which also stops a sinter worker inheriting — or consuming — the pipe that
  carries `{"cancel": true}`, since `os.dup` is non-inheritable by PEP 446.
  **`_preload` is not a second deadlock fix**, though it was first written as one and its
  docstring said so. With fd 0 detached, importing the sweep stack after the watcher
  thread starts completes normally; reverted to a no-op the same collection still finishes
  in 2.1 s. It is kept because a broken install then fails as an `input` error *before*
  `started` is emitted rather than as an `internal` one a minute into a collection, and
  because it keeps the import conditional so `generate` never pays for matplotlib. The
  dataset path never needed either fix: it spawns nothing, and `qecgen.run` is imported at
  worker module scope.
- **The interactive threshold chart computes no statistics.** `ThresholdChart.tsx` draws
  `logical_error_rate`, `ci_low` and `ci_high` exactly as `sweep.write_csv` wrote them,
  parsed server-side by `ui/sweeps.sweep_detail`. It is a *view* of the artifact; the PNG
  is the artifact of record. The one thing it does mirror is `plot_threshold`'s zero-error
  caret convention — a drawing rule, not a derived number — and both docstrings say so.
  Deriving an interval in TypeScript would be a second implementation of the
  Clopper-Pearson arithmetic and the same silent drift the generated-figure rule prevents.
- **`run.sweep_partials` has nothing to do with `SweepSpec`.** It sweeps (verb) the staging
  directories a dead run left behind. Both senses of the word now live in `run.py`; the
  name stays because `ui/jobs.py`, the tests and these docs all refer to it.
- **A drain thread may only move bytes.** An unread pipe fills — 4 KB for stdout, 64 KB
  for stderr on Windows — and the child then blocks forever on its next write, presenting
  as a run frozen at whatever progress it last reported. `jobs._drain` must never take a
  lock, touch a record or write a file; keeping the readers that dumb is what guarantees
  the child always has somewhere to write. Two more traps live on the same boundary.
  `worker.LineReader` reads its descriptor with raw `os.read` rather than `sys.stdin`,
  because a daemon thread parked in that buffered reader still holds its lock when the
  interpreter finalises, so the process hangs on exit instead of returning a status — and
  because two readers on one `TextIOWrapper` race over the read-ahead buffer and can
  swallow a cancel that arrived in the same packet as the spec. It also strips a trailing
  `\r` explicitly: reading the raw descriptor skips the text layer that would undo Windows
  line endings while the parent writes through a text-mode pipe. End of input is *not* a
  cancellation — treating it as one makes a worker fed a spec from a file cancel itself
  before sampling a shot.
- **A thread blocked reading stdin breaks heavyweight Windows operations, twice over.**
  Both faces of this hazard cost a job that hangs with *no output at all* — no event, no
  error, no exit — which the supervisor can only report as a run that never finished, and
  which the 10-second force-kill does not cover because nothing was ever cancelled.
  *(1) Process creation.* A spawned `multiprocessing` child inherits the standard handles,
  including the pipe the cancel watcher is blocked on, and then never finishes starting: a
  sweep stops at "Starting 2 workers..." forever, and the parent waits in
  `_compute_task_ids` while the children sit at ~9 MB with one thread and no Python frame.
  Measured over four variants of one collection: an open pipe with no reader finishes in
  1.2 s, an open pipe *with* a reader on fd 0 never finishes, and a reader on a duplicate
  with fd 0 pointed at devnull finishes in 1.2 s. `worker._detach_control_channel` does the
  last of those — a **non-inheritable** `os.dup` of fd 0 for the watcher (PEP 446), with
  fd 0 itself pointed at the null device, so no child inherits *or consumes* the pipe that
  carries `{"cancel": true}`. *(2) Imports —* see the next entry.
- **Every heavy import must finish before the cancel watcher parks.** Measured: a process
  with *any* thread blocked reading stdin cannot afterwards complete a large DLL-loading
  import on its main thread. `import scipy.linalg` never returns; `import decimal` and
  `import xml.dom.minidom` are unaffected; raw `os.read` and `sys.stdin.readline` deadlock
  identically; closing stdin lets the same import finish in 1.5 s. Generation never hit
  this because `run.py` imports everything at module scope, but an analysis spec importing
  lazily inside `analyse()` hung with **no output at all** — no `started`, no error, no
  exit — which the supervisor can only report as a run that never finished, and which the
  10-second force-kill does not cover because nothing was ever cancelled. `run.preload()`
  is an exhaustive match over `JobSpec` that `worker.main` calls *before* starting the
  watcher thread; a new analysis kind that skips it reintroduces a hang with no symptom.

## Environment

- **PowerShell mangles quotes inside `python -c @'...'@`.** Write diagnostic scripts to the
  scratchpad and run them as files.
- **Use `python -u` for anything redirected or backgrounded**, or `print()` block-buffers
  and a working script is indistinguishable from a hung one.
- **Diagnose a hang with `py-spy dump --pid <pid>`** (via Bash; there is no `py_spy`
  module). Guessing failed three times on the `git_commit` deadlock; one stack dump found it.
- **Other tooling edits this repo concurrently.** A lint failure in a file you did not touch
  may not be yours — check `LastWriteTime` before "fixing" it.
- **`build/lib/qecgen/` is a stale copy of the whole package**, and `qecgen.egg-info/` is
  a third copy of the metadata. Both are gitignored, so `git status` never mentions them,
  and a repo-root grep returns two hits for every symbol. `build/` is provably behind: no
  `ui/configured.py`, a different `ui/app.py` and `ui/schemas.py`, an older frontend
  bundle. Its mtimes are *copied from the source*, so they read as current and are a
  misleading staleness signal — the content diff is the only reliable one. Scope searches
  to `qecgen/`, `tests/` and `research/`.
- **More than one checkout of this repo exists on this machine.** `git worktree list`
  names `codex/realistic-qec-data` at `C:/Projects/qecgen-realistic-noise`, and this tree
  was itself moved from `C:\Projects\qecgen` — the path that
  `data/realism/.venv/pyvenv.cfg` still records. That is the concrete form of the
  `python -m` trap above: an editable install can be resolving to any of them while
  looking identical in the terminal.

## Frontend

- `npm run build` runs `tsc --noEmit` first — that is the frontend typecheck — and writes
  into `qecgen/ui/static`, which `emptyOutDir` wipes. Nothing may be stored there.
- After a rebuild, reload the browser **ignoring cache**, or you verify the previous bundle.
- Form controls carry `autoComplete="off"`. Chrome restores stale values on reload and
  React's `onChange` takes them as user input, silently changing what a run does.
- `/api/*` sends `Cache-Control: no-store`; without it a cached capabilities GET still names
  the previous `--data-root` after a restart.
- Labels associate by `htmlFor`/`useId`, never by nesting — an interactive element inside a
  `<label>` costs the control its accessible name.
- Setting copy lives in `explainers.ts`, derived from the domain docs. It belongs in the same
  correction sweep as `README.md` and `GUIDE.md`.
- **A `setState` updater must be pure**, and this one bit. `Info`'s single-open-panel singleton
  was claimed *inside* the updater, which StrictMode double-invokes precisely to surface that:
  the first call installed the panel's own closer, the second called it, and the popover opened
  and shut within one click. The double-invocation is development-only, so the built bundle
  `qecgen ui` serves was fine and only `npm run dev` — the mode the frontend is actually worked
  on in — was broken. Module-level state is claimed in an effect keyed on `open`, released in
  its cleanup behind an identity check so a newly-opened panel's closer is never nulled.
  The teaching site's `Term.tsx` (in the qecgen-learn repo) is a deliberate copy of this
  component and carries the same fix; a change here must be mirrored there.

- **A portal inside a modal `<dialog>` is inert.** `Info` renders its panel through a portal
  onto `document.body`. A `<dialog>` opened with `showModal()` sits in the top layer, and
  everything outside it is inert and painted *under* the `::backdrop` — so an `<Info>` inside
  the confirmation dialog opens a panel nobody can see or click. The delete explainer's
  trigger lives in the page's action row, outside the dialog, for that reason. The same fact
  is why `ConfirmDelete` needs no portal at all: the top layer already escapes every
  ancestor's overflow and stacking context, which is the only thing `Info`'s portal was for.
- **`ConfirmDelete` listens for `cancel`, never for `close`.** `close` fires for a
  programmatic `close()` too, and StrictMode mounts an effect, tears it down and mounts it
  again — so a `close` listener reports a user cancellation the user never made.
  Development-only, which is the mode the frontend is worked on in; same family as the `Info`
  singleton bug. `cancel` fires only for Escape, a programmatic `close()` emits none, and
  `close()` on a shut dialog is a spec no-op. The effect cleanup closes the dialog while its
  node is still connected, or the browser cannot restore focus to the trigger.
- **A delete outcome is described from what happened, not from one sentence.**
  `DeletionReport.complete` is vacuously true when nothing was attempted, so a single
  "Everything named above is gone" appears over an empty table for a run deleted with its
  files deliberately *kept* — a well-formed statement of the opposite of what happened. The
  outcome lead branches on `files.length` and `deleted_files` for that reason.

## Extension points

### Configured realism path

`configuration.py` normalizes version-1 JSON for legacy/device/hardware runs;
`noise.py` builds explicit static channels and `sampling.iter_profile_chunks`
owns chunked dynamic sampling. `hardware.py` validates supported source identities.
`ConfiguredSpec` goes through the same staged publishing path as existing jobs.
Manifest version 2 carries `generation_config` and `generation_audit`; legacy
commands preserve their old stream and schema. Unsupported manifest versions fail.

Dynamic profiles and hardware imports provide Contract A with none/coords structure
only; do not invent an independent DEM or mechanism labels for them. Configured
runs are single-environment and do not implement frozen-prior drift. New model kinds
must not reach legacy circuit-rebuilding QA or benchmarking paths. Correction
scoring is supported for generated canonical device circuits; hardware/external
circuit roles require a separate audit and are refused by correction scoring.
Covariates never silently become calibrated error probabilities. See
`docs/REALISM.md` for supported controls, physics sources and measured limitations.
Research transfer evidence lives in `research/realism`; keep raw data ignored and
generate public figures from the committed aggregate evidence, never hand-entered rates.

- **New export format:** one module in `qecgen/exporters/` satisfying the `Exporter`
  protocol, plus one entry in `EXPORTERS`. Parametrised round-trip tests pick it up from
  the registry automatically. `write()`'s `structure_level` must agree with
  `meta.structure_level` (`require_level_agreement`); a format that cannot round-trip
  structure, or that declines to carry provenance at `full`, must *downgrade* the recorded
  level rather than over-claim — `recorded_structure_level` applies both rules so they
  cannot drift apart. Two dispatch tables live in `exporters/__init__.py`, three lines from
  `EXPORTERS`: `_MANIFEST_READERS` (every format needs one — omitting it lists every file
  of the format as `unreadable`) and `_PROVENANCE_READERS` (only if the format carries
  provenance; the "no provenance stored" message is built from `carries_provenance`, not
  from those keys). They were previously private to a front end each, in different states
  of completeness, which is exactly why neither front end could use the other's. A new
  format must also raise `NotAQecgenDatasetError` for a file it did not write — that is
  the distinction `list_datasets` uses to say "not ours" rather than "broken" — and the
  signal must be **provable** rather than heuristic. The enumerated cases are on that
  exception's docstring; HDF5 is the subtle one, needing both no manifest *and* no
  `detectors` dataset, because `StreamingHDF5Writer.abort()` leaves the arrays behind.
  A format that writes more than one file must also declare them in `companions(path)`
  (`ml_csv`'s three sidecars are the existing case) so `qecgen.deletion` can offer the set;
  `test_companions_name_every_file_the_format_writes` compares the declaration against what a
  real write at `full` left in an empty directory, so a format that declares the method and
  returns `()` fails rather than silently orphaning its sidecars.
- **New analysis job kind:** one dataclass in `run.py`, one branch in each of `analyse`,
  `job_total`, `preload` and `resolved_config` — all exhaustive `match` statements, so
  mypy names the ones you missed — plus `protocol.spec_to_json`/`spec_from_json`, a
  pydantic model in `ui/schemas.py`, and an entry in `TestSpecRoundTrip._example`. That
  last one is not optional: `spec_from_json` is an if-chain mypy cannot check, and the
  `get_args(JobSpec)` round trip is what covers it. `preload` is the one that fails
  silently if forgotten — see the stdin invariants above.
  Four more sites have no `match` to name them, and `benchmark` needed every one:
  `protocol.ANALYSIS_MODES` and `mode_of`; a `_preview` arm in `ui/app.py` — an analysis
  spec has a real dataset to describe, so it *answers* rather than refusing the way a
  sweep does; the `AnalysisMode` union in `frontend/src/types.ts` (not `RunMode`); and the
  pages that launch and render it, where an artifact with no `kind` reaches the browser as
  `undefined` and takes the page down.
- **New drift axis:** one builder function plus one entry in `AXIS_BUILDERS`, plus its
  domain check in `_validate_axis_value` and its unbiased point in `unbiased_point` (both
  fail closed on an unknown axis; the registry test probes all three points).
- **New run kind in the UI:** a spec in `run.py`, one branch each in `protocol.MODES` /
  `mode_of` / `spec_to_json` / `spec_from_json`, and a request model in `schemas.py` joined
  to the `JobRequest` union — `RunRequest` is the dataset-producing subset, so an analysis
  kind added there is the wrong union. The worker needs **nothing**: an analysis spec falls
  through `worker.main`'s `isinstance` chain to `analyse`, and the progress denominator
  comes from `run.job_total`, not from a `_progress_denominator` entry. Anything heavy must
  be imported in `run.preload`, not lazily mid-run — see the deadlock invariant above.
  Four sites are easy to miss: the `/api/preview` guard (a non-dataset spec must be
  refused there with a pointer, not estimated), the `RunMode` union in
  `frontend/src/types.ts`, a browse module plus routes if the outputs are not datasets,
  and the artifact `kind` the worker reports — a payload shape the front end branches on,
  so a new one without a `kind` reaches the browser as `undefined` and takes the page down.
- **Decoders:** `decoders.py` resolves *names* against `sinter.BUILT_IN_DECODERS` and
  probes backends by module name via `find_spec`. It must never `import mwpf` or
  `fusion_blossom` — that import would be the first brick of the adapter layer the README
  puts out of scope. An `mwpf.*` entry in the `pyproject.toml` mypy overrides means the
  boundary has been crossed. A bespoke decoder needs no code here:
  `sinter.collect(custom_decoders=...)` is the extension point.
- **Nexus:** no exporter exists and the input format is unknown. Do not claim Nexus
  compatibility in code, docs or commit messages until an exporter passes a fixture
  supplied by the Nexus team.

## Conventions

- mypy `--strict` clean; ruff `E,F,I,N,UP,B,A,C4,SIM,RUF` at line length 100.
- `filterwarnings = ["error::DeprecationWarning"]` — a deprecation warning fails the suite.
- Third-party stubs: `stim`, `pymatching`, `sinter`, `h5py`, `pyarrow` and `scipy` ship no
  `py.typed`, so `pyproject.toml` carries narrowly scoped `ignore_missing_imports`
  overrides. Keep them enumerated; never blanket-ignore. `matplotlib` ships `py.typed`
  and needs no entry — a dead override for it sat in the list once, and dead entries make
  the list read as "whatever mypy complained about".
- Docstrings explain **why a trap exists**, not what the code does — that is the house
  style throughout, and the reason each invariant above survives refactoring. Preserve
  those explanations when editing; if you correct a claim, correct it in the docstring,
  `README.md`, `GUIDE.md` and `DATA_CONTRACT.md` together. `GUIDE.md` is the task-oriented
  walkthrough; its §14 Traps restates the invariants above in user-facing terms and rots
  silently if it is left out of that sweep. So does `frontend/src/explainers.ts`, which
  states them at the point of use, and the qecgen-learn repo's glossary and lesson pages,
  which state them to a reader with no other source to check against.
- **A sweep is rendered by two things and they must not disagree.** `sweep.plot_threshold`
  writes the PNG; `frontend/src/components/ThresholdChart.tsx` draws the same points in the
  browser. Only the PNG is the artifact of record. Change the caret convention, the axis
  treatment or the series labelling in one and change it in the other — the docstrings
  point at each other for exactly this reason.
- **README figures are generated, never hand-drawn, and drift silently.**
  `docs/make_diagrams.py` ports `Lattice.tsx`'s plaquette algorithm and `styles.css`'s
  palette line-for-line; `docs/make_sweep_plot.py` re-plots through `sweep.plot_threshold`
  from the committed evidence run in `docs/evidence/`. Change either source and rerun the
  script, or the README teaches a different code than the tree contains. `.gitignore`
  excludes `*.png` and `*.threshold.json` for run output and readmits those two
  directories by negation, so a figure written anywhere else is dropped from `git add`
  without a word.
- Never name a method after a builtin — mypy resolves `list[...]` *inside that class body*
  to the method, which is how `JobStore.list` broke every annotation in its own file.
  Nothing in the tree shadows a builtin any more, so ruff's `A` rules run with no ignores;
  keep it that way rather than re-adding an `A003` exception.
- `tests/test_review_regressions.py` holds one test per externally-found defect, named for
  the finding and asserting the *old* behaviour is now impossible. Add there when fixing a
  review finding.
- UI job tests inject `JobStore(worker_command=...)` with a scripted child, so event
  sequences and misbehaviours (floods, crashes, silence) are exact and sample nothing.
- Errors are raised, not papered over: conflicting inputs (e.g. `CODE_CAPACITY` with
  `rounds != 1`, unstackable environments, filename collisions in `drift`) raise rather
  than being silently overridden, so a manifest can never disagree with its file.
- **`AGENTS.md` is a deliberate copy of this file** — Codex reads that name, Claude Code
  reads this one, and both are kept self-contained rather than one pointing at the other.
  Every edit here must be made there too. The only lines that may differ are the title and
  the one sentence naming the tool; `diff CLAUDE.md AGENTS.md` should report exactly those
  two hunks and nothing else.

## Workspace layout (this checkout lives inside `C:\Projects\quantum`)

Since 2026-09-16 this repository is **software only**. The generated datasets, the
Willow/device-realism source material and the residual-error dataset project that used to
live under `data/` were moved one level up, into the workspace described by
`C:\Projects\quantum\CLAUDE.md` (and its `AGENTS.md` copy):

```
..\datasets\     generated qecgen datasets, sidecars, run records, sweeps (former data/)
..\willow\       Willow mirror shards, intake receipts, research results, research .venv
                 (former data/realism/; point research tooling at it with --root ..\willow)
..\residual\     the PyMatching residual-error pipeline (package qecgen_residual, tests,
                 configs, PROGRESS.md) and its outputs; imports this checkout via PYTHONPATH
..\run-ui.cmd    serves the web UI over ..\datasets by setting QECGEN_DATA_ROOT
```

Consequences inside this repo: `data/` is only the gitignored default scratch root for a
bare `qecgen generate` and is empty on a fresh checkout; `run-ui.cmd` honours
`QECGEN_DATA_ROOT` (default `data`); `examples/willow-import.json` documents the
hardware-import format but its source paths resolve only when the Willow shards are
placed (or acquired with `research.realism.acquire`) under `data/realism/` — in this
workspace they are under `..\willow\raw\`. Nothing task-specific gets added to the
`qecgen` package: a new pipeline built on qecgen belongs beside the repo, as `residual\` is.
Records written before the move name `C:\Projects\qecgen\…` or `…\qecgen\data\…`; see
the workspace `CLAUDE.md` for the path mapping.
