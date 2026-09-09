import { useState, type ReactNode } from "react";
import { Checkbox, Field, NumberField, Select } from "./Field";

type ObjectValue = Record<string, unknown>;
interface Props {
  value: ObjectValue;
  onChange: (value: ObjectValue) => void;
  disabled?: boolean;
}

function object(value: unknown): ObjectValue {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as ObjectValue : {};
}

function list(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function text(value: unknown): string {
  return value === undefined || value === null ? "" : String(value);
}

function number(value: unknown): number | "" {
  return typeof value === "number" && Number.isFinite(value) ? value : "";
}

function setField(value: ObjectValue, key: string, next: unknown, omitBlank = false): ObjectValue {
  const copy = { ...value };
  if (omitBlank && next === "") delete copy[key];
  else copy[key] = next;
  return copy;
}

function TextControl({ label, value, onChange, hint, error, placeholder }: {
  label: string; value: unknown; onChange: (value: string) => void;
  hint?: string; error?: string; placeholder?: string;
}) {
  return <Field label={label} hint={hint} error={error}>{id => (
    <input id={id} value={text(value)} autoComplete="off" placeholder={placeholder}
      aria-invalid={error ? true : undefined} onChange={event => onChange(event.target.value)} />
  )}</Field>;
}

function NumericList({ label, value, onChange, integers = false, hint }: {
  label: string; value: unknown; onChange: (value: unknown[]) => void;
  integers?: boolean; hint?: string;
}) {
  const values = list(value);
  const invalid = values.some(item => typeof item !== "number" || !Number.isFinite(item)
    || (integers && !Number.isSafeInteger(item)));
  return <TextControl label={label} value={values.map(text).join(", ")}
    hint={hint ?? (integers ? "Comma-separated whole numbers." : "Comma-separated numbers.")}
    error={invalid ? "Complete every number; empty or invalid entries cannot be submitted." : undefined}
    onChange={raw => onChange(raw.trim() === "" ? [] : raw.split(",").map(part => {
      const token = part.trim();
      // Keep unfinished tokens in the configuration so validation cannot silently
      // run with the last valid list or turn a blank comma entry into a zero.
      if (token === "") return "";
      const parsed = Number(token);
      return Number.isFinite(parsed) && (!integers || Number.isSafeInteger(parsed))
        ? parsed : token;
    }))} />;
}

function NumericControl({ label, value, onChange, hint, min = 0, step = "any", fallback }: {
  label: string; value: unknown; onChange: (value: number | "") => void;
  hint?: string; min?: number; step?: string | number; fallback?: number;
}) {
  return <NumberField label={label}
    value={value === undefined && fallback !== undefined ? fallback : number(value)}
    onChange={onChange} hint={hint} min={min} step={step} />;
}

function Choice({ label, value, options, onChange, hint }: {
  label: string; value: unknown; options: readonly string[];
  onChange: (value: string) => void; hint?: string;
}) {
  const selected = text(value);
  return <Select label={label} value={selected} onChange={onChange} hint={hint}
    options={selected && !options.includes(selected) ? [selected, ...options] : options} />;
}

function Section({ title, children, hint }: { title: string; children: ReactNode; hint?: string }) {
  return <details className="noise-section">
    <summary>{title}</summary>
    {hint ? <p className="hint">{hint}</p> : null}
    {children}
  </details>;
}

const operations = [
  ["one_qubit_gate", "One-qubit gate probability"],
  ["two_qubit_gate", "Two-qubit gate probability"],
  ["measurement", "Readout probability"],
  ["reset", "Reset probability"],
  ["idle", "Idle probability"],
] as const;

function ProbabilityFields({ value, onChange, overrides = false }: {
  value: ObjectValue; onChange: (value: ObjectValue) => void; overrides?: boolean;
}) {
  return <div className="field-grid">{operations
    .filter(([key]) => !overrides || key !== "two_qubit_gate")
    .map(([key, label]) => <NumericControl key={key} label={label} value={value[key]}
      fallback={overrides ? undefined : 0}
      hint={overrides ? "0–1. Blank inherits the base probability." : "0–1; zero disables this channel."}
      onChange={next => onChange(setField(value, key, next, overrides))} />)}</div>;
}

function MapRows({ title, value, onChange, create, children, edge = false }: {
  title: string; value: ObjectValue; onChange: (value: ObjectValue) => void;
  create: () => unknown; children: (value: unknown, onChange: (value: unknown) => void) => ReactNode;
  edge?: boolean;
}) {
  const [draftKey, setDraftKey] = useState("");
  const parts = draftKey.split(",");
  const canonical = (part: string) => /^(0|[1-9]\d*)$/.test(part) && Number.isSafeInteger(Number(part));
  const valid = edge
    ? parts.length === 2 && parts.every(canonical) && Number(parts[0]) < Number(parts[1])
    : canonical(draftKey);
  const duplicate = Object.hasOwn(value, draftKey);
  const error = draftKey && (!valid || duplicate)
    ? duplicate ? "This entry already exists." : edge
      ? "Use two distinct indices, smaller first: 0,1." : "Use one nonnegative whole qubit index."
    : undefined;
  return <div className="noise-rows">
    {Object.entries(value).map(([key, item]) => <fieldset className="noise-row" key={key}>
      <legend>{edge ? "Edge" : "Qubit"} {key}</legend>
      {children(item, next => onChange({ ...value, [key]: next }))}
      <div className="noise-row-actions"><button type="button" onClick={() => {
        const next = { ...value }; delete next[key]; onChange(next);
      }}>Remove {edge ? "edge" : "qubit"} {key}</button></div>
    </fieldset>)}
    <div className="noise-row-actions">
      <TextControl label={`New ${title}`} value={draftKey} onChange={setDraftKey}
        placeholder={edge ? "0,1" : "0"} error={error}
        hint={edge ? "A physical gate pair, written smaller,larger." : "Use indices from this circuit."} />
      <button type="button" disabled={!valid || duplicate} onClick={() => {
        onChange({ ...value, [draftKey]: create() }); setDraftKey("");
      }}>Add {title}</button>
    </div>
  </div>;
}

function PairRows({ value, onChange }: { value: unknown; onChange: (value: unknown[]) => void }) {
  const rows = list(value);
  return <div className="noise-rows">{rows.map((raw, index) => {
    const pair = list(raw);
    const update = (position: number, next: number | "") => {
      const changed = [...pair]; changed[position] = next;
      onChange(rows.map((row, at) => at === index ? changed : row));
    };
    return <fieldset key={index} className="noise-row"><legend>Neighbor pair {index + 1}</legend>
      <div className="field-grid">
        <NumericControl label="Source qubit" value={pair[0]} step={1} onChange={next => update(0, next)} />
        <NumericControl label="Affected neighbor" value={pair[1]} step={1} onChange={next => update(1, next)} />
      </div>
      <button type="button" onClick={() => onChange(rows.filter((_, at) => at !== index))}>
        Remove neighbor pair {index + 1}
      </button>
    </fieldset>;
  })}<button type="button" onClick={() => onChange([...rows, ["", ""]])}>Add neighbor pair</button></div>;
}

function CoherenceControls({ value, onChange }: { value: ObjectValue; onChange: (v: ObjectValue) => void }) {
  const set = (key: string, next: unknown) => onChange({ ...value, [key]: next });
  return <>
    <Checkbox label="Enable T1/T2 approximation" checked={value.enabled === true}
      onChange={enabled => onChange({ t2_protocol: "exponential_ramsey", ...value, enabled })} />
    <p className="hint">Requires calibration for every used qubit and a duration for every circuit layer.
      This Pauli approximation loses relaxation's preference for the ground state.</p>
    <div className="field-grid">
      <Choice label="T2 measurement protocol" value={value.t2_protocol ?? "exponential_ramsey"}
        options={["exponential_ramsey"]} onChange={next => set("t2_protocol", next)} />
      <Checkbox label="Gate and idle errors exclude decoherence" checked={value.residual_gate_errors_exclude_decoherence === true}
        onChange={next => set("residual_gate_errors_exclude_decoherence", next)} />
      <NumericList label="Layer durations (seconds)" value={value.layer_durations_s}
        onChange={next => set("layer_durations_s", next)}
        hint="One nonnegative duration per TICK-separated layer, including the final layer. These durations affect errors." />
      <NumericList label="Round end layers" value={value.round_end_layers} integers
        onChange={next => set("round_end_layers", next)}
        hint="Optional increasing positive layer counts. They identify round durations and rates; no independent round-rate multiplier is applied." />
    </div>
    <MapRows title="coherence qubit" value={object(value.qubits)}
      onChange={next => set("qubits", next)} create={() => ({ t1_s: "", t2_s: "" })}>
      {(raw, update) => {
        const calibration = object(raw);
        return <div className="field-grid">
          <NumericControl label="T1 (seconds)" value={calibration.t1_s}
            onChange={next => update({ ...calibration, t1_s: next })} hint="Positive measured relaxation time." />
          <NumericControl label="T2 (seconds)" value={calibration.t2_s}
            onChange={next => update({ ...calibration, t2_s: next })} hint="Positive exponential Ramsey time; this approximation requires T2 ≤ 2 × T1." />
        </div>;
      }}
    </MapRows>
  </>;
}

function SpatialControls({ value, onChange }: { value: unknown; onChange: (v: unknown[]) => void }) {
  const rows = list(value);
  return <div className="noise-rows">{rows.map((raw, index) => {
    const entry = object(raw);
    const set = (key: string, next: unknown) => onChange(rows.map((row, at) => at === index ? { ...entry, [key]: next } : row));
    return <fieldset key={index} className="noise-row"><legend>Shared event {index + 1}</legend>
      <div className="field-grid">
        <Checkbox label="Enable shared event" checked={entry.enabled !== false} onChange={next => set("enabled", next)} />
        <NumericList label="Affected qubits" value={entry.qubits} integers onChange={next => set("qubits", next)} />
        <TextControl label="Pauli effects" value={entry.paulis} onChange={next => set("paulis", next)}
          hint="One X, Y or Z per qubit, in the same order; for example XX." />
        <NumericControl label="Shared event probability" value={entry.probability}
          onChange={next => set("probability", next)} hint="0–1 per active circuit layer. One shared draw affects the listed qubits." />
        <Choice label="Evidence for this shared event" value={entry.basis ?? "scenario_assumption"}
          options={["scenario_assumption", "measured"]} onChange={next => set("basis", next)} />
      </div>
      <button type="button" onClick={() => onChange(rows.filter((_, at) => at !== index))}>Remove shared event {index + 1}</button>
    </fieldset>;
  })}<button type="button" onClick={() => onChange([...rows, {
    enabled: true, qubits: [], paulis: "", probability: "", basis: "scenario_assumption",
  }])}>Add shared event</button></div>;
}

function DynamicControls({ kind, value, onChange }: {
  kind: "drift" | "bursts" | "leakage"; value: ObjectValue; onChange: (v: ObjectValue) => void;
}) {
  const set = (key: string, next: unknown) => onChange({ ...value, [key]: next });
  const numbers: { key: string; label: string; hint: string; min?: number }[] = kind === "drift" ? [
    { key: "baseline_probability", label: "Baseline probability", hint: "Strictly between 0 and 1 before drift." },
    { key: "rho", label: "Shot-to-shot persistence (rho)", min: -1, hint: "−1 to 1. State advances once per acquisition shot." },
    { key: "sigma_logit", label: "Drift innovation scale", hint: "Nonnegative standard deviation in log-odds; not a measured time constant." },
  ] : kind === "bursts" ? [
    { key: "onset_probability", label: "Burst onset probability", hint: "0–1 per inactive acquisition shot." },
    { key: "recovery_probability", label: "Burst recovery probability", hint: "0–1 per active acquisition shot." },
    { key: "effect_probability", label: "Burst effect probability", hint: "0–1 per listed qubit and layer while the burst is active." },
  ] : [
    { key: "entry_probability", label: "Leakage entry probability", hint: "0–1 per layer; proxy occupancy starts empty each shot." },
    { key: "recovery_probability", label: "Leakage recovery probability", hint: "0–1 per layer." },
    { key: "reset_removal_probability", label: "Removal by reset probability", hint: "0–1 when a selected qubit is reset." },
    { key: "effect_probability", label: "Occupied-qubit effect probability", hint: "0–1 for each independent X and Z effect." },
    { key: "neighbor_effect_probability", label: "Neighbor effect probability", hint: "0–1 for each X and Z effect on a listed neighbor of an occupied source." },
  ];
  return <>
    <Checkbox label={`Enable ${kind === "leakage" ? "leakage-effect proxy" : kind}`}
      checked={value.enabled === true} onChange={enabled => onChange({
        ...(kind === "leakage" ? {} : { pauli: "X" }), ...value, enabled,
      })} />
    <div className="field-grid">
      <NumericList label={`${kind === "leakage" ? "Leakage" : kind === "bursts" ? "Burst" : "Drift"} qubits`}
        value={value.qubits} integers onChange={next => set("qubits", next)}
        hint="Distinct qubit indices from this circuit; required when enabled." />
      {kind !== "leakage" ? <Choice label="Pauli effect" value={value.pauli ?? "X"}
        options={["X", "Y", "Z"]} onChange={next => set("pauli", next)} /> : null}
      {numbers.map(({ key, label, hint, min }) => <NumericControl key={key} label={label}
        value={value[key]} onChange={next => set(key, next)} hint={hint} min={min} />)}
    </div>
    {kind === "leakage" ? <>
      <p className="hint">Directed pairs below apply effects from a leakage qubit to its neighbor.
        This is a classical effect proxy; it does not simulate a third quantum level.</p>
      <PairRows value={value.neighbor_edges} onChange={next => set("neighbor_edges", next)} />
    </> : <p className="hint">State persists across output chunks and stays constant inside each shot.
      Shot index is not a measured wall-clock time.</p>}
    <p className="hint">Enabled dynamic mechanisms support Contract A with none/coordinates structure;
      they cannot provide an exact static detector error model or mechanism labels.</p>
    <button type="button" onClick={() => {
      const cleared = { ...value };
      const fields = ["enabled", "qubits", ...(kind === "leakage" ? ["neighbor_edges"] : ["pauli"]),
        ...numbers.map(item => item.key)];
      for (const key of fields) {
        delete cleared[key];
      }
      onChange({ ...cleared, enabled: false });
    }}>Clear {kind === "leakage" ? "leakage proxy" : kind} settings</button>
  </>;
}

/** Every edit changes only its field; imported extension keys remain for backend validation. */
export function NoiseControls({ value, onChange, disabled = false }: Props) {
  const set = (key: string, next: unknown) => onChange({ version: 1, ...value, [key]: next });
  const covariates = object(value.covariates);
  return <fieldset className="noise-controls" disabled={disabled}>
    <legend>Device noise profile</legend>
    <div className="field-grid">
      <TextControl label="Profile label" value={value.label} onChange={next => set("label", next)}
        hint="A name for this calibration or scenario." />
      <Choice label="Noise profile version" value={value.version ?? 1} options={["1"]}
        onChange={next => set("version", Number(next))} hint="Version 1 is the supported noise schema." />
      <Choice label="Classical control policy" value={value.classical_control_policy ?? "reject"}
        options={["reject", "ideal_pauli_frame"]} onChange={next => set("classical_control_policy", next)}
        hint="Ideal Pauli frame accepts supported sweep-controlled gates with zero synthetic sweep bits; conditional control-pulse errors are not modeled." />
    </div>
    <ProbabilityFields value={object(value.probabilities)} onChange={next => set("probabilities", next)} />
    <Section title="Per-qubit probabilities" hint="Overrides replace the corresponding base rate. Remove an entry to inherit every base rate for that qubit.">
      <MapRows title="probability qubit" value={object(value.qubit_overrides)}
        onChange={next => set("qubit_overrides", next)} create={() => ({})}>
        {(raw, update) => <ProbabilityFields value={object(raw)} onChange={update} overrides />}
      </MapRows>
    </Section>
    <Section title="Two-qubit gate overrides" hint="Overrides apply to an actual physical gate pair, independent of the pair's listed order in the circuit.">
      <MapRows title="gate edge" edge value={object(value.edge_overrides)}
        onChange={next => set("edge_overrides", next)} create={() => ""}>
        {(raw, update) => <NumericControl label="Two-qubit gate override probability" value={raw}
          onChange={update} hint="0–1. Remove the edge to use the base two-qubit rate." />}
      </MapRows>
    </Section>
    <Section title="T1/T2 and circuit timing" hint="Use compatible measured times. Gate error estimates must exclude decoherence before adding it separately.">
      <CoherenceControls value={object(value.coherence)} onChange={next => set("coherence", next)} />
    </Section>
    <Section title="Spatially shared events" hint="Explicit qubit groups control proximity effects. Distance alone does not determine an error probability.">
      <SpatialControls value={value.spatial} onChange={next => set("spatial", next)} />
    </Section>
    <Section title="Drift across acquisition shots">
      <DynamicControls kind="drift" value={object(value.drift)} onChange={next => set("drift", next)} />
    </Section>
    <Section title="Bursts across acquisition shots">
      <DynamicControls kind="bursts" value={object(value.bursts)} onChange={next => set("bursts", next)} />
    </Section>
    <Section title="Leakage-effect proxy">
      <DynamicControls kind="leakage" value={object(value.leakage)} onChange={next => set("leakage", next)} />
    </Section>
    <Section title="Frequency, temperature and humidity" hint="These are recorded covariates with zero automatic effect. Timing above affects errors through the chosen decoherence model. Humidity has no isolated, calibrated response in this generator.">
      <div className="field-grid">
        {([
          ["cryostat_temperature_k", "Cryostat temperature (K)"],
          ["effective_qubit_temperature_k", "Effective qubit temperature (K)"],
          ["humidity_relative_fraction", "Relative humidity (fraction)"],
        ] as const).map(([key, label]) => <NumericControl key={key} label={label}
          value={covariates[key]} hint={key === "humidity_relative_fraction"
            ? "Optional 0–1 fraction; blank leaves it unrecorded."
            : "Optional positive temperature; blank leaves it unrecorded."}
          onChange={next => set("covariates", setField(covariates, key, next, true))} />)}
      </div>
      <MapRows title="frequency qubit" value={object(covariates.transition_frequency_hz)}
        onChange={next => set("covariates", { ...covariates, transition_frequency_hz: next })}
        create={() => ""}>
        {(raw, update) => <NumericControl label="Transition frequency (Hz)" value={raw} onChange={update}
          hint="Positive qubit transition frequency, not syndrome round rate. Recorded without an automatic error multiplier." />}
      </MapRows>
    </Section>
  </fieldset>;
}
