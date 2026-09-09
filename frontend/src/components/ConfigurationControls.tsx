import { Checkbox, Field, Select } from "./Field";
import type { Capabilities } from "../types";

export type ConfigObject = Record<string, unknown>;
export function object(value: unknown): ConfigObject {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as ConfigObject : {};
}

export function TextSetting({ label, value, onChange, hint, disabled = false }: {
  label: string; value: unknown; onChange: (value: string) => void;
  hint?: string; disabled?: boolean;
}) {
  return <Field label={label} hint={hint}>{(id) => <input id={id}
    value={typeof value === "string" || typeof value === "number" ? String(value) : ""}
    disabled={disabled} autoComplete="off" onChange={(event) => onChange(event.target.value)}
  />}</Field>;
}

export function IntegerSetting({ label, value, onChange, hint, disabled = false, min = 0 }: {
  label: string; value: unknown; onChange: (value: number | string) => void;
  hint?: string; disabled?: boolean; min?: number;
}) {
  const raw = typeof value === "string" || typeof value === "number" ? String(value) : "";
  const valid = /^\d+$/.test(raw) && Number.isSafeInteger(Number(raw)) && Number(raw) >= min;
  return <Field label={label} hint={hint} error={!valid ? `Enter an integer from ${min} to ${Number.MAX_SAFE_INTEGER}.` : undefined}>
    {(id) => <input id={id} value={raw} inputMode="numeric" disabled={disabled}
      autoComplete="off" aria-invalid={!valid || undefined} onChange={(event) => {
        const text = event.target.value;
        // Keep invalid or oversized input verbatim. Rounding a uint64 seed would
        // silently select a different experiment before it reaches Python.
        const parsed = Number(text);
        onChange(/^\d+$/.test(text) && Number.isSafeInteger(parsed) && parsed >= min ? parsed : text);
      }} />}
  </Field>;
}

export function ConfigurationControls({ value, onChange, caps, disabled = false }: {
  value: ConfigObject; onChange: (value: ConfigObject) => void;
  caps: Capabilities; disabled?: boolean;
}) {
  const mode = String(value.mode);
  const output = object(value.output);
  const sampling = object(value.sampling);
  const circuit = object(value.circuit);
  const legacy = object(value.legacy);
  const provenance = object(value.parameter_provenance);
  const hardware = object(value.hardware);
  const expected = object(hardware.expected);
  const change = (section: string, key: string, next: unknown) => {
    const fields = { ...object(value[section]) };
    if (next === undefined) delete fields[key]; else fields[key] = next;
    onChange({ ...value, [section]: fields });
  };
  const expectedChange = (key: string, next: unknown) =>
    change("hardware", "expected", { ...expected, [key]: next });
  const custom = "stim_file" in circuit;

  return <div className="configuration-controls">
    <section className="panel config-section">
      <h3>Output and sampling</h3>
      <div className="field-grid">
        <TextSetting label="Output path" value={output.path} onChange={(v) => change("output", "path", v)}
          disabled={disabled} hint="Relative to the UI data root. Existing output is replaced atomically." />
        <Select label="Output format" value={String(output.format ?? "hdf5")}
          options={caps.formats.map((f) => f.name)} onChange={(v) => change("output", "format", v)}
          disabled={disabled} hint={`Required suffix: ${caps.formats.find(f => f.name === output.format)?.extension ?? "choose a format"}`} />
        <Select label="Stored structure" value={String(output.structure ?? "none")}
          options={caps.structure_levels} onChange={(v) => change("output", "structure", v)} disabled={disabled}
          hint="Hardware and dynamic profiles support none or coords, not an exact static DEM." />
        <IntegerSetting label={mode === "hardware" ? "Rows to import" : "Shots"} min={1} value={sampling.shots}
          onChange={(v) => change("sampling", "shots", v)} disabled={disabled} />
        <IntegerSetting label="Seed" value={sampling.seed} onChange={(v) => change("sampling", "seed", v)}
          disabled={disabled} hint="This browser supports exact integers up to 9007199254740991. Use the CLI for larger uint64 seeds." />
        <IntegerSetting label="Chunk size" min={1} value={sampling.chunk_size}
          onChange={(v) => change("sampling", "chunk_size", v)} disabled={disabled}
          hint="Part of reproducibility: changing chunks changes the sample stream." />
        <Checkbox label="Emit DEM mechanisms (Contract B)" checked={sampling.emit_mechanisms === true}
          onChange={(v) => change("sampling", "emit_mechanisms", v)} disabled={disabled} />
      </div>
    </section>

    <section className="panel config-section">
      <h3>Circuit and layout</h3>
      <p className="note">Distance, rounds and basis must agree with an imported circuit. Qubit IDs are circuit IDs, not row numbers.</p>
      <div className="field-grid">
        <IntegerSetting label="Code distance" value={circuit.distance} min={2}
          onChange={(v) => change("circuit", "distance", v)} disabled={disabled} />
        <IntegerSetting label="Syndrome rounds" value={circuit.rounds} min={1}
          onChange={(v) => change("circuit", "rounds", v)} disabled={disabled}
          hint="Legacy code capacity requires one round." />
        <Select label="Memory basis" value={String(circuit.basis ?? "z")} options={caps.bases}
          onChange={(v) => change("circuit", "basis", v)} disabled={disabled} />
        <Checkbox label="Rotated layout" checked={circuit.rotated !== false}
          onChange={(v) => change("circuit", "rotated", v)} disabled={disabled} />
        {mode === "device" && <Checkbox label="Use a custom Stim circuit" checked={custom} disabled={disabled}
          onChange={(enabled) => {
            const next = { ...circuit };
            if (enabled) { next.stim_file = ""; next.sha256 = ""; }
            else { delete next.stim_file; delete next.sha256; }
            onChange({ ...value, circuit: next });
          }} />}
        {mode === "device" && custom && <>
          <TextSetting label="Custom Stim path" value={circuit.stim_file} disabled={disabled}
            onChange={(v) => change("circuit", "stim_file", v)} />
          <TextSetting label="Custom circuit SHA-256" value={circuit.sha256} disabled={disabled}
            onChange={(v) => change("circuit", "sha256", v)} hint="64 lowercase hexadecimal characters; circuit identity is checked by the server." />
        </>}
      </div>
    </section>

    {mode === "legacy" && <section className="panel config-section">
      <h3>Legacy noise convention</h3>
      <div className="field-grid">
        <Select label="Legacy noise model" value={String(legacy.noise_model ?? "stim_uniform_circuit_level")}
          options={caps.noise_models} onChange={(v) => change("legacy", "noise_model", v)} disabled={disabled} />
        <Field label="Uniform probability p" hint="A synthetic probability, not a hardware calibration.">{(id) =>
          <input id={id} type="number" min={0} max={1} step="any" value={String(legacy.p ?? "")}
            disabled={disabled} autoComplete="off" onChange={(event) => change("legacy", "p", event.target.value === "" ? "" : Number(event.target.value))} />}
        </Field>
      </div>
    </section>}

    {mode === "device" && <section className="panel config-section">
      <h3>Where the parameters came from</h3>
      <p className="note">A measured or fitted label requires an attributable source. Fitting uses training data only.</p>
      <div className="field-grid">
        <Select label="Parameter provenance" value={String(provenance.kind ?? "scenario")}
          options={["scenario", "measured", "fitted"]} onChange={(v) => change("parameter_provenance", "kind", v)} disabled={disabled} />
        <TextSetting label="Parameter description" value={provenance.description} disabled={disabled}
          onChange={(v) => change("parameter_provenance", "description", v)} />
        <TextSetting label="Parameter source" value={provenance.source} disabled={disabled}
          onChange={(v) => change("parameter_provenance", "source", v || undefined)} hint="Required for measured and fitted profiles; use a source URL or identifier." />
        <Select label="Fit partition" value={String(provenance.fit_partition ?? "not specified")}
          options={["not specified", "train"]} disabled={disabled}
          onChange={(v) => change("parameter_provenance", "fit_partition", v === "not specified" ? undefined : v)}
          hint="Choose train for a fitted profile. Validation and test data must not be used to fit it." />
      </div>
    </section>}

    {mode === "hardware" && <section className="panel config-section">
      <h3>Published hardware source</h3>
      <p className="note">Imports check the source files and cohort identity. Outcomes remain observed hardware targets; no DEM mechanisms are invented.</p>
      <div className="field-grid">
        <TextSetting label="Hardware table path" value={hardware.table} disabled={disabled}
          onChange={(v) => change("hardware", "table", v)} />
        <TextSetting label="Hardware Stim path" value={hardware.circuit} disabled={disabled}
          onChange={(v) => change("hardware", "circuit", v)} />
        <IntegerSetting label="First source row (offset)" value={hardware.offset ?? 0} disabled={disabled}
          onChange={(v) => change("hardware", "offset", v)} />
        <TextSetting label="Expected table SHA-256" value={expected.table_sha256} disabled={disabled}
          onChange={(v) => expectedChange("table_sha256", v)} />
        <TextSetting label="Expected circuit SHA-256" value={expected.circuit_sha256} disabled={disabled}
          onChange={(v) => expectedChange("circuit_sha256", v)} />
        <IntegerSetting label="Expected source distance" value={expected.distance} min={2} disabled={disabled}
          onChange={(v) => expectedChange("distance", v)} />
        <IntegerSetting label="Expected source rounds" value={expected.rounds} min={1} disabled={disabled}
          onChange={(v) => expectedChange("rounds", v)} />
        <Select label="Expected source basis" value={String(expected.basis ?? "Z")} options={["X", "Z"]} disabled={disabled}
          onChange={(v) => expectedChange("basis", v)} />
        <TextSetting label="Expected orientation" value={expected.orientation} disabled={disabled}
          onChange={(v) => expectedChange("orientation", v)} />
      </div>
    </section>}
  </div>;
}
