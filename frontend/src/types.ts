/** Payload shapes returned by the qecgen API. Mirrors qecgen/ui/*.py. */

/** Job kinds that produce a dataset. The New run form offers exactly these. */
export type RunMode = "generate" | "multi-env" | "drift";

/** Job kinds that read what already exists and report on it. */
export type AnalysisMode = "sweep" | "score" | "qa";

export type Mode = RunMode | AnalysisMode;

/**
 * What a run's progress counts.
 *
 * A dataset run counts shots. A sweep counts sinter tasks, because `max_errors` stops a
 * sweep and `max_shots` is only a ceiling, so its shot total is not knowable in advance.
 * Empty when the total is not knowable at all, which the bar renders as indeterminate.
 * The unit travels with the number so a bar never labels one thing while counting another.
 */
export type ProgressUnit = "shots" | "tasks" | "";

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


/**
 * One decoder name and whether it can run here.
 *
 * Unusable decoders are reported rather than filtered out of the list: `problem` names the
 * exact pip install that is missing, and a form that silently omitted the entry would
 * leave a user who came looking for it with nothing to act on.
 */
export interface DecoderInfo {
  name: string;
  usable: boolean;
  problem: string | null;
  backing_package: string | null;
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
  decoders: DecoderInfo[];
  /** Named `_sweep_` because it is the sweep form's default, not a global preference. */
  default_sweep_decoders: string[];
  /** The host's core count. The sweep form defaults `workers` to two fewer than this. */
  cpu_count: number;
  data_root: string;
  runs_dir: string;
  max_concurrent_jobs: number;
  static_built: boolean;
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

/** A point that hit the shot ceiling before the error target, so its interval is widest. */
export interface CensoredPoint {
  decoder: string;
  distance: number;
  p: number;
  shots: number;
  errors: number;
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
  kind: "dataset";
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
 *
 * `kind` is what `worker._ARTIFACT_KINDS` assigned from the extension — "plot", "results
 * table", "summary" — not a closed union, because a new analysis kind adds one without
 * the browser needing to know it in advance.
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
  total_units: number;
  completed_units: number;
  /**
   * What `total_units` and `completed_units` count. Named "units" rather than "shots"
   * because a sweep counts tasks, and a field that says shots while holding tasks is a
   * field lying about its own contents. `JobStore.load_history` still reads the older
   * `total_shots`/`completed_shots` keys, so run records written before the rename load.
   */
  progress_unit: ProgressUnit;
  phase: string | null;
  /** Free-form line about what is happening now. For a sweep, sinter's own status line. */
  detail: string | null;
  /** Sweep only. A dataset run's shot count is `completed_units` itself. */
  shots_collected: number | null;
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

/* ---------- sweeps ---------- */

export interface SweepPreview {
  distances: number[];
  error_rates: number[];
  decoders: DecoderInfo[];
  /** False when any named decoder is unknown or its backend is not installed. */
  usable: boolean;
  n_tasks: number;
  max_errors: number;
  max_shots_per_task: number;
  /** Worst case if every task runs to its ceiling, not a forecast. See `note`. */
  max_shots_total: number;
  workers: number;
  results_path: string;
  plot_path: string;
  summary_path: string;
  /** A rerun onto an existing stem replaces all three files together. */
  overwrites: boolean;
  note: string;
}

export interface SweepEntry {
  stem: string;
  summary_path: string;
  results_path: string | null;
  plot_path: string | null;
  modified_at: number;
  size_bytes: number;
  decoders: string[];
  crossings: Record<string, number | null>;
  unreadable: string | null;
}

/**
 * One collected point, exactly as `qecgen.sweep.write_csv` wrote it.
 *
 * Every number here is read from a column of the results table. Nothing in the browser
 * derives a rate or an interval — the chart is a *view* of the artifact, and `sweep.png`
 * remains the artifact of record.
 */
export interface SweepPoint {
  decoder: string;
  distance: number;
  p: number;
  shots: number;
  errors: number;
  discards: number;
  rate: number;
  ci_low: number;
  ci_high: number;
}

export interface SweepSeries {
  decoder: string;
  distance: number;
  points: SweepPoint[];
}

export interface ThresholdSummary {
  qecgen_version: string;
  alpha: number;
  reported_not_asserted: string;
  dem_seen_by_decoders: string;
  noise_model: string;
  basis: string;
  max_errors: number;
  max_shots: number;
  censored_points: CensoredPoint[];
  decoders: Record<string, SweepDecoderSummary>;
}

export interface SweepDetail {
  stem: string;
  results_path: string;
  plot_path: string | null;
  /** The plot's mtime, for cache-busting: a re-run writes the same path. */
  plot_modified_at: number | null;
  summary_path: string;
  series: SweepSeries[];
  summary: ThresholdSummary;
}
