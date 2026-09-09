import { useEffect, useId, useRef, useState } from "react";
import { api } from "../api";
import { ConfigurationControls, object, TextSetting, type ConfigObject } from "../components/ConfigurationControls";
import { Field, Select } from "../components/Field";
import { NoiseControls } from "../components/NoiseControls";
import { assertBrowserNumbers, CONFIG_STARTERS, parseConfiguration, saveConfiguration, sweepValues } from "../configuration";
import { count } from "../format";
import type { Capabilities, ConfiguredLayout, ConfiguredPreview, ConfiguredSweepPreview } from "../types";

const message = (error: unknown) => error instanceof Error ? error.message : String(error);
const clone = (value: ConfigObject): ConfigObject => structuredClone(value);

export function ConfiguredRun({ caps, onSubmitted }: {
  caps: Capabilities; onSubmitted: (id: string) => void;
}) {
  const jsonId = useId();
  const hintId = useId();
  const [config, setConfig] = useState<ConfigObject>(() => clone(CONFIG_STARTERS.device));
  const modes = useRef<Record<string, ConfigObject>>({});
  const [jsonDraft, setJsonDraft] = useState(() => JSON.stringify(CONFIG_STARTERS.device, null, 2));
  const [jsonDirty, setJsonDirty] = useState(false);
  const [preview, setPreview] = useState<ConfiguredPreview | null>(null);
  const [layout, setLayout] = useState<ConfiguredLayout | null>(null);
  const [sweep, setSweep] = useState<ConfiguredSweepPreview | null>(null);
  const [sweepField, setSweepField] = useState("noise.probabilities.measurement");
  const [sweepText, setSweepText] = useState("0.003, 0.006, 0.01");
  const [sweepBody, setSweepBody] = useState<{ config: ConfigObject; field: string; values: (number | boolean)[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const busyRef = useRef(false);
  const [submitted, setSubmitted] = useState<string[]>([]);
  const revision = useRef(0);
  const mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; revision.current++; }; }, []);

  function invalidate() {
    revision.current++;
    setPreview(null); setLayout(null); setSweep(null); setSweepBody(null); setError(null); setSubmitted([]);
  }
  function edit(next: ConfigObject) {
    invalidate(); setConfig(next); setJsonDraft(JSON.stringify(next, null, 2)); setJsonDirty(false);
  }
  function selectMode(mode: string) {
    if (mode !== "legacy" && mode !== "device" && mode !== "hardware") return;
    modes.current[String(config.mode)] = clone(config);
    edit(clone(modes.current[mode] ?? CONFIG_STARTERS[mode]));
    setSweepField(mode === "legacy" ? "legacy.p" : "noise.probabilities.measurement");
  }
  function currentConfig(): ConfigObject {
    if (jsonDirty) throw new Error("Apply or discard the JSON edits before validating or running.");
    assertBrowserNumbers(config);
    return clone(config);
  }
  async function operation(name: string, work: (current: number) => Promise<void>) {
    if (busyRef.current) return;
    busyRef.current = true;
    const current = revision.current;
    setBusy(name); setError(null);
    try { await work(current); }
    catch (err) { if (mounted.current && current === revision.current) setError(message(err)); }
    finally { busyRef.current = false; if (mounted.current) setBusy(null); }
  }
  const isCurrent = (current: number) => mounted.current && current === revision.current;
  const disabled = busy !== null || jsonDirty;

  async function validate() {
    setPreview(null);
    await operation("Validating", async (current) => {
      const result = await api.configuredPreview(currentConfig());
      if (isCurrent(current)) setPreview(result);
    });
  }
  async function showLayout() {
    await operation("Reading circuit", async (current) => {
      const result = await api.configuredLayout(currentConfig());
      if (isCurrent(current)) setLayout(result);
    });
  }
  async function previewSweep() {
    setSweep(null); setSweepBody(null);
    await operation("Validating sweep", async (current) => {
      const body = { config: currentConfig(), field: sweepField.trim(), values: sweepValues(sweepText) };
      const result = await api.configuredSweepPreview(body.config, body.field, body.values);
      if (isCurrent(current)) { setSweep(result); setSweepBody(body); }
    });
  }
  async function submitSweep() {
    if (!sweepBody || !sweep) return;
    await operation("Queuing sweep", async (current) => {
      const result = await api.configuredSweep(sweepBody.config, sweepBody.field, sweepBody.values);
      if (isCurrent(current)) { setSubmitted(result.runs.map(run => run.id)); setSweep(null); setSweepBody(null); }
    });
  }
  function applyJson() {
    try { edit(parseConfiguration(jsonDraft)); }
    catch (err) { setError(message(err)); }
  }
  async function importFile(file: File) {
    await operation("Reading JSON", async (current) => {
      const parsed = parseConfiguration(await file.text());
      if (isCurrent(current)) edit(parsed);
    });
  }

  return <main className="configured-run">
    <div className="config-heading"><div>
      <h2>Device and hardware runs</h2>
      <p>Choose how errors are generated, record the assumptions, then validate and run. Every dataset keeps its resolved configuration and seed.</p>
    </div><span className="tag">Config version {String(config.version ?? "missing")}</span></div>

    <section className="panel config-section">
      <div className="field-grid">
        <Select label="Generation mode" value={String(config.mode)} options={["legacy", "device", "hardware"]}
          disabled={disabled} onChange={selectMode} hint="Each mode keeps its edits while this page remains open." />
        <Field label="Import a JSON configuration" hint="Invalid files leave the form unchanged; oversized integer seeds require the CLI.">{(id) =>
          <input id={id} type="file" accept=".json,application/json" disabled={busy !== null}
            onChange={(event) => { const file = event.target.files?.[0]; event.target.value = ""; if (file) void importFile(file); }} />}
        </Field>
      </div>
      <div className="config-actions">
        {([ ["legacy", "Legacy example"], ["device", "Static device example"], ["dynamic", "Experimental dynamic example"], ["hardware", "Willow hardware example"] ] as const).map(([key, title]) =>
          <button key={key} disabled={busy !== null} onClick={() => { edit(clone(CONFIG_STARTERS[key])); setSweepField(key === "legacy" ? "legacy.p" : "noise.probabilities.measurement"); }}>{title}</button>)}
        <button disabled={busy !== null || jsonDirty} onClick={() => {
          try { saveConfiguration(JSON.stringify(currentConfig(), null, 2), "qecgen-config.json"); }
          catch (err) { setError(message(err)); }
        }}>Save JSON</button>
      </div>
      <p className="note">Loading an example replaces the current form. Device example parameters are illustrative assumptions, not measured calibration. The Willow example uses downloaded third-party-derived data with pinned checksums.</p>
      <p className="note">Input and output paths are below <code>{caps.data_root}</code>. Hardware example paths assume the usual <code>data</code> root; adjust them when using another root.</p>
    </section>

    {jsonDirty && <p className="flag flag--warn">JSON edits are pending. Apply them below, or discard them to continue using the form.</p>}
    <ConfigurationControls value={config} onChange={edit} caps={caps} disabled={disabled} />

    <section className="panel config-section">
      <h3>Circuit reference</h3>
      <p className="note">Read the actual circuit IDs and interaction pairs before entering qubit-specific noise. This checks circuit identity, not the quality of a calibration.</p>
      <button disabled={disabled} onClick={() => void showLayout()}>Show circuit qubits and layers</button>
      {layout && <div aria-live="polite" className="config-reference">
        <p>{count(layout.qubits.length)} qubits · {count(layout.edges.length)} interaction pairs · {count(layout.layer_count)} layers · {count(layout.n_detectors)} detectors · {count(layout.n_observables)} observables</p>
        <p>{layout.note}</p>
        <details><summary>Qubit IDs and interaction pairs</summary>
          <p>Qubits: <code>{layout.qubits.join(", ")}</code></p>
          <p>Pairs: <code>{layout.edges.map(pair => pair.join(" ↔ ")).join("; ") || "none"}</code></p>
          <p>Circuit SHA-256: <code>{layout.circuit_sha256}</code></p>
        </details>
      </div>}
    </section>

    {config.mode === "device" && <NoiseControls value={object(config.noise)}
      onChange={(noise) => edit({ ...config, noise })} disabled={disabled} />}

    <section className="panel config-section">
      <h3>Validate and generate</h3>
      <p className="note">Outputs are detection events and logical observable flips. Mechanism targets require a supported static detector error model.</p>
      <div className="config-actions">
        <button disabled={disabled} onClick={() => void validate()}>Validate configuration</button>
        <button className="primary" disabled={disabled || !preview} onClick={() => void operation("Starting run", async (current) => {
          if (!preview) return;
          const result = await api.submit({ mode: "configured", config: preview.config });
          if (isCurrent(current)) onSubmitted(result.id);
        })}>Start run</button>
      </div>
      {preview && <div aria-live="polite" className="config-reference">
        <p>{count(preview.total_shots)} shots → <code>{preview.output_path}</code> ({preview.format})</p>
        {preview.n_detectors !== undefined && preview.n_observables !== undefined &&
          <p>{count(preview.n_detectors)} detectors · {count(preview.n_observables)} observables</p>}
        <p>{preview.note}</p>
        {preview.timing && <details>
          <summary>Configured physical timing</summary>
          <p>Total circuit duration: {preview.timing.physical_duration_s === null
            ? "not specified" : `${preview.timing.physical_duration_s} seconds`}.</p>
          <p className="note">These values are reported by the generator from the configured timing. They are not measured hardware round rates.</p>
          {preview.timing.round_durations_s.length > 0 && <div className="config-table-scroll">
            <table><thead><tr><th>Round</th><th>Duration (seconds)</th><th>Round rate (Hz)</th></tr></thead>
              <tbody>{preview.timing.round_durations_s.map((duration, index) =>
                <tr key={index}><td>{index + 1}</td><td>{duration}</td>
                  <td>{preview.timing?.round_rates_hz[index] ?? "not specified"}</td></tr>)}</tbody>
            </table>
          </div>}
        </details>}
        <details><summary>Resolved configuration</summary><pre className="mono-block">{JSON.stringify(preview.config, null, 2)}</pre></details>
      </div>}
    </section>

    {config.mode !== "hardware" ? <section className="panel config-section">
      <h3>Sweep one parameter</h3>
      <p className="note">Each value creates a separately reproducible run. A server-derived seed and output path are shown before submission. Sweep values are numbers or true/false.</p>
      <div className="field-grid">
        <TextSetting label="Sweep field" value={sweepField} disabled={disabled}
          onChange={(v) => { invalidate(); setSweepField(v); }} hint="Examples: legacy.p, noise.probabilities.measurement, noise.drift.enabled, noise.qubit_overrides.1.measurement" />
        <TextSetting label="Sweep values" value={sweepText} disabled={disabled}
          onChange={(v) => { invalidate(); setSweepText(v); }} hint="Comma-separated, for example 0.001, 0.003, 0.01 or false, true." />
      </div>
      <div className="config-actions">
        <button disabled={disabled} onClick={() => void previewSweep()}>Preview sweep</button>
        <button disabled={disabled || !sweep} onClick={() => void submitSweep()}>Queue sweep runs</button>
      </div>
      {sweep && <div aria-live="polite" className="config-reference">
        <p>{sweep.runs.length} runs · {count(sweep.total_shots)} total shots. {sweep.note}</p>
        <div className="config-table-scroll"><table><thead><tr><th>Value</th><th>Output path</th><th>Exact seed</th><th>Config</th></tr></thead>
          <tbody>{sweep.runs.map((run, index) => <tr key={run.output_path}><td>{String(sweep.values[index])}</td><td><code>{run.output_path}</code></td><td><code>{run.seed}</code></td>
            <td><button onClick={() => saveConfiguration(run.config_json, `qecgen-sweep-${index}.json`)}>Save JSON</button></td></tr>)}</tbody></table></div>
      </div>}
      {submitted.length > 0 && <p role="status">Queued {submitted.length} runs: {submitted.map((id, index) => <span key={id}>{index > 0 && ", "}<a href={`#/runs/${id}`}>{id}</a></span>)}</p>}
    </section> : <p className="note">Parameter sweeps apply to synthetic legacy/device runs; hardware import preserves measured rows.</p>}

    <details className="panel config-section">
      <summary>Advanced JSON editor</summary>
      <p id={hintId}>The form preserves unrecognised fields for server validation. Apply JSON explicitly to update the form. Invalid edits stay here without replacing the last usable configuration.</p>
      <label htmlFor={jsonId}>Versioned JSON configuration</label>
      <textarea id={jsonId} className="config-editor" rows={22} aria-describedby={hintId}
        autoComplete="off" spellCheck={false} disabled={busy !== null} value={jsonDraft}
        onChange={(event) => { invalidate(); setJsonDraft(event.target.value); setJsonDirty(true); }} />
      <div className="config-actions">
        <button disabled={busy !== null || !jsonDirty} onClick={applyJson}>Apply JSON to form</button>
        <button disabled={busy !== null || !jsonDirty} onClick={() => { setJsonDraft(JSON.stringify(config, null, 2)); setJsonDirty(false); setError(null); }}>Discard JSON edits</button>
      </div>
    </details>
    {busy && <p className="config-status" role="status">{busy}…</p>}
    {error && <p className="config-status flag flag--bad" role="alert">{error}</p>}
  </main>;
}
