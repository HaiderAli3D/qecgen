/** Payload shapes returned by the qecgen API. Mirrors qecgen/ui/*.py. */

/** Job kinds that produce a dataset. The New run form offers exactly these. */
export type RunMode = "generate" | "multi-env" | "drift";

/** Job kinds that read what already exists and report on it. */
export type AnalysisMode = "sweep" | "score" | "qa";

export type Mode = RunMode | AnalysisMode;

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
  /** Whether this format stores circuit and DEM text at `--structure full`. */
  carries_provenance: boolean;
}

/** A sinter decoder name, and whether its backend is actually installed here. */
export interface DecoderInfo {
  name: string;
  installed: boolean;
  usable: boolean;
  backing_package: string | null;
  /** Actionable text when it cannot be used: a typo, or an absent package to install. */
  problem: string | null;
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
  decoders: DecoderInfo[];
  default_sweep_decoders: string[];
}

/** What a dataset's correction arrays must look like, derived from its own circuit. */
export interface CorrectionSchema {
  n_data_qubits: number;
  packed_width: number;
  n_observables: number;
  shots: number;
  schema_digest: string;
  bit_order: string;
  content_hash: string | null;
  drift_condition: string;
}

/** A correction file on disk, read from its .npy headers rather than its arrays. */
export interface CorrectionEntry {
  path: string;
  name: string;
  shots: number;
  width: number;
  dtype: string;
  unpacked: boolean;
  size_bytes: number;
}

/** `/api/preview` for a score: the compatibility check, before either file is read. */
export interface ScorePreview extends CorrectionSchema {
  correction_shots: number;
  correction_width: number;
  correction_dtype: string;
  unpacked: boolean;
  compatible: boolean;
  problems: string[];
}

/** `/api/preview` for a sweep: the grid, and whether the decoders can actually run. */
export interface SweepPreview {
  distances: number[];
  rates: number[];
  decoders: {
    name: string;
    usable: boolean;
    problem: string | null;
    backing_package: string | null;
  }[];
  n_tasks: number;
  max_errors: number;
  max_shots_per_task: number;
  workers: number;
  out_csv: string;
  out_plot: string;
  out_summary: string;
  usable: boolean;
  note: string;
}

/** `/api/preview` for QA. `max_total_shots` is a ceiling, not an estimate. */
export interface QaPreview {
  n_environments: number;
  max_shots_per_environment: number;
  target_errors: number;
  max_total_shots: number;
  shots_in_file: number | null;
  resamples: boolean;
  note: string;
}

/** One environment's measured logical error rate, with its interval. */
export interface QaEnvironment {
  environment_id: number;
  axis: string;
  axis_value: number;
  p: number;
  logical_error_rate: number;
  ci_low: number;
  ci_high: number;
  failures: number;
  shots: number;
  detection_event_rate: number;
}

/** One exponential-suppression fit. `lambda` below 1 means distance is not helping. */
export interface SuppressionRow {
  p: number;
  lambda: number;
  lambda_low: number;
  lambda_high: number;
  prefactor: number;
  distances_used: number[];
  excluded_zero_error: number[];
  residual_dof: number;
  reduced_chi_square: number | null;
  suppressing: boolean;
}

export interface SweepDecoderSummary {
  crossing_p: number | null;
  crossing_method: string;
  suppression: SuppressionRow[];
}

/** The `.threshold.json` payload, byte for byte what the sidecar on disk holds. */
export interface ThresholdSummary {
  qecgen_version: string;
  alpha: number;
  reported_not_asserted: string;
  dem_seen_by_decoders: string;
  noise_model: string;
  basis: string;
  max_errors: number;
  max_shots: number;
  censored_points: {
    decoder: string;
    distance: number;
    p: number;
    shots: number;
    errors: number;
  }[];
  decoders: Record<string, SweepDecoderSummary>;
}

/** A sweep result set on disk, indexed by the one file that names itself a sweep. */
export interface SweepEntry {
  path: string;
  name: string;
  csv: string | null;
  plot: string | null;
  modified_at: number;
  summary: ThresholdSummary & { unreadable?: string };
}

/**
 * Circuit and DEM text, fetched only when explicitly asked for.
 *
 * Never rendered beside a manifest by default. Under `FROZEN_PRIOR` this text is exactly
 * what the condition withholds from a decoder, and the separation is defeated by a page
 * that shows both because both happened to be loaded.
 */
export interface Provenance {
  path: string;
  structure_level: string;
  drift_condition: string;
  structure_source_environment_id: number | null;
  environments: { environment_id: number; circuit: string; dem: string }[];
  stored: boolean;
  formats_that_store_it: string[];
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
