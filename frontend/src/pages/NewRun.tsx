import { useEffect, useMemo, useState } from "react";
import { ApiError, api } from "../api";
import {
  Checkbox,
  Field,
  ListField,
  NumberField,
  Select,
  invalidListTokens,
  parseList,
} from "../components/Field";
import { Info } from "../components/Info";
import { Intro } from "../components/Intro";
import { Lattice } from "../components/Lattice";
import { EXPLAINERS } from "../explainers";
import { bytes, count } from "../format";
import type { Capabilities, Mode, Preview } from "../types";

const MODES: { id: Mode; label: string; blurb: string }[] = [
  { id: "generate", label: "Single", blurb: "One environment, one file." },
  {
    id: "multi-env",
    label: "Multi-environment",
    blurb: "Several environments pooled into one file, shots shuffled and labelled.",
  },
  {
    id: "drift",
    label: "Drift",
    blurb: "A training set plus one test set per drifted value, under a stated condition.",
  },
];

interface FormState {
  distance: number | "";
  rounds: number | "";
  basis: string;
  rotated: boolean;
  noise_model: string;
  p: number | "";
  seed: number | "";
  chunk_size: number | "";
  fmt: string;
  structure_level: string;
  emit_mechanisms: boolean;
  out: string;
  shots: number | "";
  axis: string;
  axis_values: string;
  base_p: number | "";
  shots_per_env: number | "";
  shuffle: boolean;
  train_p: number | "";
  test_values: string;
  condition: string;
}

function initial(caps: Capabilities): FormState {
  return {
    distance: 3,
    rounds: "",
    basis: "z",
    rotated: true,
    noise_model: "stim_uniform_circuit_level",
    p: 0.005,
    seed: 0,
    chunk_size: caps.default_chunk_size,
    fmt: "hdf5",
    structure_level: "none",
    emit_mechanisms: false,
    out: "dataset.h5",
    shots: 100000,
    axis: "p",
    axis_values: "0.003, 0.005, 0.008, 0.012",
    base_p: "",
    shots_per_env: 25000,
    shuffle: true,
    train_p: 0.005,
    test_values: "0.007, 0.010, 0.014",
    condition: "frozen_prior",
  };
}

function buildBody(mode: Mode, form: FormState): Record<string, unknown> | null {
  const shared = {
    mode,
    distance: form.distance,
    // Under code_capacity the Rounds field is disabled and shows "1 (fixed)", but the
    // form still holds whatever was typed under the previous model. Sending that stale
    // value would fail the run at sample time; null hands the decision to
    // default_rounds, the same rule the display is quoting.
    rounds: form.noise_model === "code_capacity" || form.rounds === "" ? null : form.rounds,
    basis: form.basis,
    rotated: form.rotated,
    noise_model: form.noise_model,
    seed: form.seed,
    chunk_size: form.chunk_size,
    fmt: form.fmt,
    structure_level: form.structure_level,
    emit_mechanisms: form.emit_mechanisms,
    out: form.out,
  };
  if (form.distance === "" || form.seed === "" || form.chunk_size === "" || !form.out) return null;

  if (mode === "generate") {
    if (form.p === "" || form.shots === "") return null;
    return { ...shared, p: form.p, shots: form.shots };
  }
  if (mode === "multi-env") {
    // A token that fails to parse blocks the body outright. Submitting the parseable
    // remainder would silently produce fewer environments than were typed.
    const values = parseList(form.axis_values);
    if (invalidListTokens(form.axis_values).length > 0) return null;
    if (values.length === 0 || form.shots_per_env === "") return null;
    if (form.axis !== "p" && form.base_p === "") return null;
    return {
      ...shared,
      axis_values: values,
      shots_per_env: form.shots_per_env,
      axis: form.axis,
      base_p: form.base_p === "" ? null : form.base_p,
      shuffle: form.shuffle,
    };
  }
  const tests = parseList(form.test_values);
  if (invalidListTokens(form.test_values).length > 0) return null;
  if (tests.length === 0 || form.train_p === "" || form.shots === "") return null;
  return {
    ...shared,
    train_p: form.train_p,
    test_values: tests,
    shots: form.shots,
    condition: form.condition,
    axis: form.axis,
  };
}

function listFieldError(text: string): string | undefined {
  const bad = invalidListTokens(text);
  if (bad.length === 0) return undefined;
  return `Not numbers: ${bad.join(", ")}`;
}

interface Props {
  caps: Capabilities;
  onSubmitted: (id: string) => void;
}

export function NewRun({ caps, onSubmitted }: Props) {
  const [mode, setMode] = useState<Mode>("generate");
  const [form, setForm] = useState<FormState>(() => initial(caps));
  const [preview, setPreview] = useState<Preview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const set = <K extends keyof FormState>(key: K, value: FormState[K]) =>
    setForm((current) => ({ ...current, [key]: value }));

  const body = useMemo(() => buildBody(mode, form), [mode, form]);
  const bodyKey = JSON.stringify(body);

  // Structure levels other than "none" have to match between build and export --
  // require_level_agreement refuses a mismatch -- so there is one control, not two.
  const codeCapacity = form.noise_model === "code_capacity";
  const format = caps.formats.find((entry) => entry.name === form.fmt);
  const structureLost = form.structure_level !== "none" && format && !format.structure_round_trip;

  useEffect(() => {
    if (!body) {
      // The error goes with the preview: an incomplete form is a blank slate, and a
      // complaint about a body that no longer exists reads as a complaint about this one.
      setPreview(null);
      setPreviewError(null);
      return;
    }
    let live = true;
    const timer = window.setTimeout(() => {
      api
        .preview(body)
        .then((result) => {
          if (!live) return;
          setPreview(result);
          setPreviewError(null);
        })
        .catch((error: unknown) => {
          if (!live) return;
          setPreview(null);
          setPreviewError(error instanceof ApiError ? error.message : String(error));
        });
    }, 220);
    return () => {
      live = false;
      window.clearTimeout(timer);
    };
  }, [bodyKey, body]);

  useEffect(() => {
    // Keep the filename honest when the format changes; the exporter would rename it
    // anyway and then the reported path would not be the one that was typed.
    if (!format) return;
    setForm((current) => {
      if (mode === "drift") return current;
      const stem = current.out.replace(/\.[^./\\]+$/, "");
      return { ...current, out: `${stem}${format.extension}` };
    });
  }, [form.fmt, mode, format]);

  async function submit() {
    if (!body) return;
    setBusy(true);
    setSubmitError(null);
    try {
      const record = await api.submit(body);
      onSubmitted(record.id);
    } catch (error) {
      setSubmitError(error instanceof ApiError ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  const active = MODES.find((entry) => entry.id === mode);

  return (
    <>
      <Intro />
      <div className="run-layout">
      <div className="panel form-card">
        <fieldset style={{ borderTop: "none" }}>
          <legend>
            <span className="field-head">
              Run kind
              <Info topic={EXPLAINERS.mode} />
            </span>
          </legend>
          <div className="segmented">
            {MODES.map((entry) => (
              <button
                key={entry.id}
                type="button"
                aria-pressed={mode === entry.id}
                onClick={() => {
                  // Re-clicking the active mode is a no-op, not a reset: the handler
                  // below overwrites the output name, which must not happen to a
                  // filename the user already typed.
                  if (entry.id === mode) return;
                  setMode(entry.id);
                  setForm((current) => ({
                    ...current,
                    out: entry.id === "drift" ? "drift" : `dataset${format?.extension ?? ""}`,
                    // A drift study exists to ship a DEM with its test files, so it
                    // defaults to structure. Matches `qecgen drift --structure dem`.
                    structure_level:
                      entry.id === "drift" && current.structure_level === "none"
                        ? "dem"
                        : current.structure_level,
                  }));
                }}
              >
                {entry.label}
              </button>
            ))}
          </div>
          <p className="note" style={{ marginTop: "0.6rem" }}>
            {active?.blurb}
          </p>
        </fieldset>

        <fieldset>
          <legend>Code</legend>
          <div className="grid">
            <NumberField
              label="Distance"
              topic={EXPLAINERS.distance}
              min={2}
              value={form.distance}
              onChange={(v) => set("distance", v)}
            />
            <NumberField
              label="Rounds"
              topic={EXPLAINERS.rounds}
              min={1}
              value={codeCapacity ? "" : form.rounds}
              disabled={codeCapacity}
              placeholder={codeCapacity ? "1 (fixed)" : "= distance"}
              hint={
                codeCapacity
                  ? "Code capacity measures perfectly, so it is always one round."
                  : "Blank uses the distance."
              }
              onChange={(v) => set("rounds", v)}
            />
            <Select
              label="Memory basis"
              topic={EXPLAINERS.basis}
              value={form.basis}
              options={caps.bases}
              onChange={(v) => set("basis", v)}
            />
            <Checkbox
              label="Rotated layout"
              topic={EXPLAINERS.rotated}
              checked={form.rotated}
              onChange={(v) => set("rotated", v)}
            />
          </div>
        </fieldset>

        <fieldset>
          <legend>Noise</legend>
          <div className="grid">
            <Select
              label="Model"
              topic={EXPLAINERS.noise_model}
              value={form.noise_model}
              options={caps.noise_models}
              onChange={(v) => set("noise_model", v)}
            />
            {mode === "generate" && (
              <NumberField
                label="Error rate p"
                topic={EXPLAINERS.p}
                step="0.001"
                value={form.p}
                onChange={(v) => set("p", v)}
              />
            )}
            {mode !== "generate" && (
              <Select
                label="Drift axis"
                topic={EXPLAINERS.axis}
                value={form.axis}
                options={caps.drift_axes}
                onChange={(v) => set("axis", v)}
                hint={form.axis === "p" ? undefined : "Varies something other than p."}
              />
            )}
            {mode === "multi-env" && (
              <NumberField
                label="Base p"
                topic={EXPLAINERS.base_p}
                step="0.001"
                value={form.base_p}
                disabled={form.axis === "p"}
                placeholder={form.axis === "p" ? "from axis values" : "required"}
                hint={form.axis === "p" ? undefined : "Required: the axis is not p."}
                onChange={(v) => set("base_p", v)}
              />
            )}
            {mode === "drift" && (
              <NumberField
                label="Training p"
                topic={EXPLAINERS.train_p}
                step="0.001"
                value={form.train_p}
                onChange={(v) => set("train_p", v)}
              />
            )}
          </div>
        </fieldset>

        {mode === "multi-env" && (
          <fieldset>
            <legend>Environments</legend>
            <div className="grid">
              <ListField
                label={form.axis === "p" ? "Error rates" : "Axis values"}
                topic={EXPLAINERS.axis_values}
                value={form.axis_values}
                onChange={(v) => set("axis_values", v)}
                hint="Comma separated. One environment each."
                error={listFieldError(form.axis_values)}
              />
              <NumberField
                label="Shots per environment"
                topic={EXPLAINERS.shots_per_env}
                min={1}
                value={form.shots_per_env}
                onChange={(v) => set("shots_per_env", v)}
              />
              <Checkbox
                label="Shuffle shots"
                topic={EXPLAINERS.shuffle}
                checked={form.shuffle}
                onChange={(v) => set("shuffle", v)}
              />
            </div>
            {!form.shuffle && (
              <span className="flag">
                Unshuffled, the row index tells a model which environment a shot came from.
                Leave this on unless you are debugging.
              </span>
            )}
          </fieldset>
        )}

        {mode === "drift" && (
          <fieldset>
            <legend>Drift</legend>
            <div className="grid">
              <ListField
                label="Test values"
                topic={EXPLAINERS.test_values}
                value={form.test_values}
                onChange={(v) => set("test_values", v)}
                hint="Comma separated. One test file each."
                error={listFieldError(form.test_values)}
              />
              <Select
                label="Condition"
                topic={EXPLAINERS.condition}
                value={form.condition}
                options={caps.drift_conditions}
                onChange={(v) => set("condition", v)}
              />
            </div>
            <span className="flag flag--calm">
              {form.condition === "frozen_prior"
                ? "Test files ship the training environment's structure, so the decoder has to generalise."
                : "Each test file ships its own environment's structure. That measures a ceiling, not generalisation."}
            </span>
          </fieldset>
        )}

        <fieldset>
          <legend>Sampling</legend>
          <div className="grid">
            {mode !== "multi-env" && (
              <NumberField
                label={mode === "drift" ? "Shots per file" : "Shots"}
                topic={EXPLAINERS.shots}
                min={1}
                value={form.shots}
                onChange={(v) => set("shots", v)}
              />
            )}
            <NumberField
              label="Seed"
              topic={EXPLAINERS.seed}
              min={0}
              value={form.seed}
              onChange={(v) => set("seed", v)}
            />
            <NumberField
              label="Chunk size"
              topic={EXPLAINERS.chunk_size}
              min={1}
              value={form.chunk_size}
              onChange={(v) => set("chunk_size", v)}
              hint="Part of the reproducibility contract."
            />
          </div>
        </fieldset>

        <fieldset>
          <legend>Output</legend>
          <div className="grid">
            <Select
              label="Format"
              topic={EXPLAINERS.fmt}
              value={form.fmt}
              options={caps.formats.map((entry) => entry.name)}
              onChange={(v) => set("fmt", v)}
            />
            <Select
              label="Structure"
              topic={EXPLAINERS.structure_level}
              value={form.structure_level}
              options={caps.structure_levels}
              onChange={(v) => set("structure_level", v)}
            />
            <Checkbox
              label="Record mechanisms"
              topic={EXPLAINERS.emit_mechanisms}
              checked={form.emit_mechanisms}
              onChange={(v) => set("emit_mechanisms", v)}
            />
            <Field
              label={mode === "drift" ? "Output directory" : "Output file"}
              topic={EXPLAINERS.out}
              hint={`Relative to ${caps.data_root}`}
            >
              {(id) => (
                <input
                  id={id}
                  value={form.out}
                  autoComplete="off"
                  onChange={(event) => set("out", event.target.value)}
                />
              )}
            </Field>
          </div>
          {structureLost && format && (
            // Both names are derived from the registry: hardcoding "Parquet" here was
            // wrong the moment the same warning fired for JSONL.
            <span className="flag">
              The {format.name} format cannot round-trip structure, so the file records{" "}
              <code>structure_level: none</code>. Use{" "}
              {caps.formats
                .filter((entry) => entry.structure_round_trip)
                .map((entry) => entry.name)
                .join(" or ")}{" "}
              if the structure has to survive a read.
            </span>
          )}
          {form.emit_mechanisms && (
            <span className="flag flag--calm">
              Mechanism labels say which DEM mechanisms fired, not which physical Pauli
              faults occurred. All three arrays come from the DEM sampler so they explain
              each other.
            </span>
          )}
        </fieldset>
      </div>

      <aside className="aside">
        <div className="panel">
          <Lattice
            distance={typeof form.distance === "number" ? form.distance : 3}
            basis={form.basis}
          />
          {!form.rotated && (
            <p className="note">
              This run uses the unrotated layout; the figure shows the rotated one.
            </p>
          )}
          <hr className="rule" />
          {preview ? (
            <>
              <dl className="readout">
                <dt>Shots</dt>
                <dd className="big">{count(preview.total_shots)}</dd>
                <dt>Detectors</dt>
                <dd>{count(preview.n_detectors)}</dd>
                <dt>Observables</dt>
                <dd>{preview.n_observables}</dd>
                <dt>DEM mechanisms</dt>
                <dd>{count(preview.n_mechanisms)}</dd>
                <dt>Rounds</dt>
                <dd>{preview.rounds}</dd>
                <dt>Row</dt>
                <dd>{preview.row_bytes} B</dd>
                <dt>Uncompressed</dt>
                <dd>{bytes(preview.estimated_bytes)}</dd>
                <dt>Files</dt>
                <dd>{preview.n_files}</dd>
                <dt>Chunks per file</dt>
                <dd>{count(preview.chunks)}</dd>
              </dl>
              {preview.will_stream ? (
                <span className="flag flag--calm">
                  Streams to disk. Memory stays at one chunk however many shots you ask for.
                </span>
              ) : (
                <span className="flag">
                  Holds every shot in memory before writing — about{" "}
                  {bytes(preview.estimated_bytes)} plus overhead.
                </span>
              )}
            </>
          ) : (
            <p className="note">{previewError ?? "Fill in the run to see what it will cost."}</p>
          )}
          <hr className="rule" />
          <button className="primary" onClick={submit} disabled={!body || busy}>
            {busy ? "Starting…" : "Start run"}
          </button>
          {submitError && <span className="flag flag--bad">{submitError}</span>}
        </div>

        <div className="panel">
          <h3>Resolved configuration</h3>
          <pre className="mono-block">{JSON.stringify(body ?? {}, null, 2)}</pre>
          <p className="note" style={{ marginTop: "0.6rem" }}>
            Stored with the run, so the record says what was asked for including the
            values you left at their defaults.
          </p>
        </div>
      </aside>
      </div>
    </>
  );
}
