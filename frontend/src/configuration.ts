import type { ConfigObject } from "./components/ConfigurationControls";

const common = {
  version: 1,
  output: { path: "device-static.h5", format: "hdf5", structure: "coords" },
  sampling: { shots: 100000, seed: 12345, chunk_size: 10000, emit_mechanisms: false },
  circuit: { distance: 3, rounds: 3, basis: "z", rotated: true },
};
const staticNoise = {
  version: 1, label: "Illustrative scenario; not device calibration",
  probabilities: { one_qubit_gate: 0.001, two_qubit_gate: 0.003, measurement: 0.01, reset: 0.002, idle: 0.0001 },
  qubit_overrides: { "1": { measurement: 0.02 } }, edge_overrides: {},
  coherence: { enabled: false }, spatial: [], drift: { enabled: false },
  bursts: { enabled: false }, leakage: { enabled: false }, covariates: {},
};

export const CONFIG_STARTERS: Record<"legacy" | "device" | "dynamic" | "hardware", ConfigObject> = {
  legacy: { ...common, mode: "legacy", output: { ...common.output, path: "legacy.h5" },
    legacy: { noise_model: "stim_uniform_circuit_level", p: 0.005 } },
  device: { ...common, mode: "device", noise: staticNoise,
    parameter_provenance: { kind: "scenario", description: "Illustrative assumed parameters, not device calibration." } },
  dynamic: { ...common, mode: "device", output: { ...common.output, path: "device-dynamic.h5" },
    parameter_provenance: { kind: "scenario", description: "Experimental Pauli-effect proxies with illustrative parameters; not calibrated physical time constants." },
    noise: { ...staticNoise,
      drift: { enabled: true, qubits: [1, 3], pauli: "Z", baseline_probability: 0.002, rho: 0.99, sigma_logit: 0.2 },
      bursts: { enabled: true, qubits: [1, 3, 5], pauli: "X", onset_probability: 0.001, recovery_probability: 0.1, effect_probability: 0.05 },
      leakage: { enabled: true, qubits: [1], entry_probability: 0.001, recovery_probability: 0.1,
        reset_removal_probability: 1, effect_probability: 0.5, neighbor_edges: [[1, 2]], neighbor_effect_probability: 0.1 },
    } },
  hardware: { ...common, mode: "hardware", output: { ...common.output, path: "willow-z10.h5" },
    sampling: { ...common.sampling, shots: 50000 }, circuit: { ...common.circuit, rounds: 10 },
    hardware: {
      table: "realism/raw/willow-derived-z-r010/d3_at_q10_7__Z__r010.parquet",
      circuit: "realism/raw/willow-derived-z-r010-circuit/d3_at_q10_7__Z__r010.stim", offset: 0,
      expected: { table_sha256: "2324ecb77a859d005850341a442396d6c3e9f8a2f12a4928364f27933b8ac696",
        circuit_sha256: "fba4d5575c0afa11ce2126acbbe7d3a2546609ecac66195ea2ba696c45ef085e",
        distance: 3, rounds: 10, basis: "Z", orientation: "q10_7" },
    } },
};

export function assertBrowserNumbers(value: unknown, path = "configuration"): void {
  if (typeof value === "number" && (!Number.isFinite(value) ||
    (Number.isInteger(value) && !Number.isSafeInteger(value)))) {
    throw new Error(`${path} is outside the browser's exact numeric range. Use the CLI for uint64 seeds above 9007199254740991.`);
  }
  if (value && typeof value === "object") {
    for (const [key, item] of Object.entries(value)) assertBrowserNumbers(item, `${path}.${key}`);
  }
}

/**
 * JSON.parse validates syntax but discards earlier duplicate properties. Scan the
 * original tokens as well, so an imported seed cannot silently change meaning.
 * This scanner runs only after syntax validation; numbers are left to JSON.parse.
 */
function requireUniqueKeys(text: string): void {
  let cursor = 0;
  function whitespace() {
    while (cursor < text.length && " \t\r\n".includes(text[cursor]!)) cursor++;
  }
  function stringToken(): string {
    const start = cursor++;
    while (cursor < text.length) {
      const character = text[cursor++];
      if (character === "\\") cursor++;
      else if (character === '"') return JSON.parse(text.slice(start, cursor)) as string;
    }
    throw new Error("Unterminated JSON string");
  }
  function value(path: string): void {
    whitespace();
    if (text[cursor] === '"') {
      stringToken();
    } else if (text[cursor] === "{") {
      cursor++;
      whitespace();
      const keys = new Set<string>();
      if (text[cursor] !== "}") {
        while (cursor < text.length) {
          const key = stringToken();
          if (keys.has(key)) throw new Error(`Duplicate configuration field at ${path}: ${JSON.stringify(key)}`);
          keys.add(key);
          whitespace();
          cursor++; // Colon; JSON.parse already validated it.
          value(`${path}[${JSON.stringify(key)}]`);
          whitespace();
          if (text[cursor] === "}") break;
          cursor++; // Comma.
          whitespace();
        }
      }
      cursor++;
    } else if (text[cursor] === "[") {
      cursor++;
      whitespace();
      let index = 0;
      if (text[cursor] !== "]") {
        while (cursor < text.length) {
          value(`${path}[${index++}]`);
          whitespace();
          if (text[cursor] === "]") break;
          cursor++; // Comma.
        }
      }
      cursor++;
    } else {
      // Number, boolean or null. Keeping the lexical spelling out of Number()
      // supports every valid exponent/fraction form without another numeric parser.
      while (cursor < text.length && !" \t\r\n,]}".includes(text[cursor]!)) cursor++;
    }
  }
  value("configuration");
}

export function parseConfiguration(text: string): ConfigObject {
  const value: unknown = JSON.parse(text);
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("The configuration must be a JSON object.");
  }
  requireUniqueKeys(text);
  assertBrowserNumbers(value);
  return value as ConfigObject;
}

export function sweepValues(text: string): (number | boolean)[] {
  const tokens = text.split(",").map(s => s.trim());
  if (tokens.some(s => s === "")) throw new Error("Enter a comma-separated list without empty sweep values.");
  return tokens.map(token => {
    if (token === "true" || token === "false") return token === "true";
    const number = Number(token);
    if (!Number.isFinite(number)) throw new Error(`Invalid sweep value: ${token}`);
    assertBrowserNumbers(number, "Sweep value");
    return number;
  });
}

export function saveConfiguration(text: string, filename: string): void {
  const blob = new Blob([text.endsWith("\n") ? text : text + "\n"], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  // Revoking in a later task lets browsers start the download first.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
