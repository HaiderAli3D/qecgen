/** Payload shapes returned by the qecgen API. Mirrors qecgen/ui/*.py. */

export type Mode = "generate" | "multi-env" | "drift";

export type RunStatus =
  | "queued"
  | "running"
  | "cancelling"
  | "succeeded"
  | "failed"
  | "cancelled";

export const TERMINAL: readonly RunStatus[] = ["succeeded", "failed", "cancelled"];

export interface FormatInfo {
  name: string;
  extension: string;
  streaming: boolean;
  structure_round_trip: boolean;
}

export interface Capabilities {
  version: string;
  noise_models: string[];
  bases: string[];
  structure_levels: string[];
  drift_axes: string[];
  drift_conditions: string[];
  default_chunk_size: number;
  formats: FormatInfo[];
  data_root: string;
  runs_dir: string;
  max_concurrent_jobs: number;
  static_built: boolean;
}

export interface Preview {
  total_shots: number;
  n_detectors: number;
  n_observables: number;
  n_mechanisms: number;
  row_bytes: number;
  estimated_bytes: number;
  n_files: number;
  chunks: number;
  will_stream: boolean;
  materialises: boolean;
  rounds: number;
  channels: Record<string, number>;
}

export interface WrittenFile {
  path: string;
  shots: number;
  content_hash: string | null;
  drift_condition: string;
  structure_source_environment_id: number | null;
}

/**
 * A file a job produced that is *not* a dataset — a sweep's plot or threshold sidecar.
 *
 * Deliberately not a `WrittenFile`. That type means one specific thing (a dataset, with a
 * shot count, a content hash and a drift condition) and the Runs table renders those
 * columns; a PNG has none of them, so reusing the shape would print invented values.
 */
export interface Artifact {
  path: string;
  kind: string;
  size_bytes: number;
}

export interface RunRecord {
  id: string;
  mode: Mode;
  spec: Record<string, unknown>;
  status: RunStatus;
  total_shots: number;
  completed_shots: number;
  /**
   * What `total_shots` and `completed_shots` count. Empty when the total is not knowable
   * in advance, which the progress bar renders as indeterminate. The field names still
   * say "shots" because renaming them would break every run record already on disk.
   */
  progress_unit: string;
  phase: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  files: WrittenFile[];
  artifacts: Artifact[];
  warnings: string[];
  error: string | null;
  error_kind: string | null;
  result: Record<string, unknown> | null;
}

export interface ManifestSummary {
  distance: number | null;
  rounds: number | null;
  basis: string | null;
  shots: number | null;
  n_detectors: number | null;
  n_observables: number | null;
  contract: string | null;
  structure_level: string | null;
  drift_condition: string | null;
  drift_axis: string | null;
  n_environments: number;
  content_hash: string | null;
  generated_at: string | null;
}

export interface DatasetEntry {
  path: string;
  name: string;
  format: string;
  size_bytes: number;
  modified_at: number;
  manifest: ManifestSummary | null;
  unreadable: string | null;
  /**
   * Set when the file carries a dataset extension but qecgen did not write it — a
   * `qecgen sweep` results table, most likely. Distinct from `unreadable`: the file is
   * intact, so flagging it as corruption would devalue the flag that means a worker
   * died mid-write.
   */
  not_a_dataset: string | null;
}

export interface Check {
  name: string;
  passed: boolean;
  detail: string;
  requirement: string;
}

export interface ValidationReport {
  ok: boolean;
  checks: Check[];
}

/** A field-attributable validation failure, as pydantic reports it. */
export interface FieldError {
  loc: (string | number)[];
  msg: string;
}
