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

## Commands

Python 3.13+, dependencies pinned exactly (dataset reproducibility depends on the Stim
version). Installed globally on this machine — no venv activation needed.

```bash
pip install -e ".[dev]"                 # runtime + pytest, ruff, mypy, httpx
pip install -e ".[ui]"                  # optional: fastapi, uvicorn for `qecgen ui`
pip install -e ".[decoders]"            # optional: mwpf, fusion-blossom for `sweep --decoder`

ruff check . && ruff format --check .
mypy --strict qecgen tests
pytest -m "not slow"                    # 473 fast structural tests
pytest -m slow                          # 7 statistical / end-to-end tests
pytest tests/test_dem.py::TestName::test_name   # single test
pytest -k xz_bias -v                            # by keyword

cd frontend && npm ci && npm run build  # into qecgen/ui/static (gitignored);
                                        # `build` runs `tsc --noEmit` first
cd frontend && npm run typecheck        # that gate alone, no bundle

python docs/make_diagrams.py            # redraw the README SVGs
python docs/make_sweep_plot.py          # re-plot the threshold PNG from docs/evidence/
```

The six-lesson teaching site lives in its own repository,
[qecgen-learn](https://github.com/HaiderAli3D/qecgen-learn). Its lesson copy and glossary
state this repo's traps and conventions, so doc corrections here must be swept there too.

The CLI installs as `qecgen` (also runnable as `python -m qecgen.cli`):
`generate`, `multi-env`, `drift`, `sweep`, `validate [--qa]`, `score`, `inspect`,
`formats`, `ui`. Every command prints its fully resolved config before doing work, so a
terminal log is a complete record of the run. `data/`, `out/`, `runs/` and all dataset
extensions are gitignored. `*.csv` is among them, negated by `!docs/evidence/*.csv` for
the committed sweep evidence a README figure is built from.

`qecgen ui` serves the web UI for the three dataset-producing commands on loopback. The
frontend is built on demand — the command names the build line rather than serving a
blank page.

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
exporters/     Exporter protocol + registry (hdf5, npz, parquet, jsonl, csv),
               infer_format; structure_json.py holds the normative structure encoding
               shared byte-for-byte by jsonl and csv
run.py         one run end to end: specs, format routing, staged writes, WrittenFile.
               The layer both front ends call; imports neither typer nor pydantic
validate.py    fast deterministic structural checks (default)
qa.py          slow statistical checks with Clopper-Pearson intervals (opt-in)
sweep.py       sinter threshold sweeps -> CSV + plot (independent of the dataset path)
cli.py         typer commands, config printing, progress
ui/            local web UI. protocol.py (wire format) + worker.py (child process)
               depend only on stdlib and run.py; jobs.py supervises the children;
               datasets.py browses and resolves paths under the data root; settings.py
               and schemas.py hold config and request models; app.py is the only
               module that imports FastAPI
frontend/      Vite + React source; builds into qecgen/ui/static
docs/          README figures + the scripts that regenerate them; imports qecgen.sweep,
               and nothing imports it
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
- **`full` means full, or the manifest says otherwise.** `hdf5`, `npz` and `csv` carry the
  provenance text; `jsonl` and `parquet` decline it and therefore must not record
  `structure_level: full`. JSONL's refusal is deliberate and load-bearing — the idiomatic
  reader is `for line in f: json.loads(line)`, one loop from handing a frozen-prior test
  file's own DEM to a decoder — but it recorded `full` anyway for a while, which is an
  over-claim in the one field a reader cannot check against the file. `carries_provenance`
  plus `recorded_structure_level` make it a registry-wide invariant with one parametrised
  test, not five independent conventions. CSV is safe to carry it because a reader that
  does not filter `#` lines never finds the table at all.
- **A CSV dataset's `shot` column must equal its row index.** This is the one format users
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

## Extension points

- **New export format:** one module in `qecgen/exporters/` satisfying the `Exporter`
  protocol, plus one entry in `EXPORTERS`. Parametrised round-trip tests pick it up from
  the registry automatically. `write()`'s `structure_level` must agree with
  `meta.structure_level` (`require_level_agreement`); a format that cannot round-trip
  structure, or that declines to carry provenance at `full`, must *downgrade* the recorded
  level rather than over-claim — `recorded_structure_level` applies both rules so they
  cannot drift apart. Two front-end dispatches are **not** registry-derived and need one
  entry each: `ui/datasets._MANIFEST_READERS` (the cheap listing path — omitting it lists
  every file of the format as `unreadable`) and `cli._PROVENANCE_READERS` (only if the
  format carries provenance; the "no provenance stored" message is built from its keys).
  Both are covered by tests that fail if you forget, and `cli._read_manifest_only` should
  gain a branch if the format can produce a manifest without reading its shots. If the new
  extension can also name a file qecgen writes for another purpose — as `.csv` does, for
  `qecgen sweep` — the reader must raise `NotAQecgenDatasetError`, which is the distinction
  `list_datasets` uses to say "not ours" rather than "broken".
- **New drift axis:** one builder function plus one entry in `AXIS_BUILDERS`, plus its
  domain check in `_validate_axis_value` and its unbiased point in `unbiased_point` (both
  fail closed on an unknown axis; the registry test probes all three points).
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
