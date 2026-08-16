/** Payload shapes returned by the qecgen API. Mirrors qecgen/ui/*.py. */

export type Mode = "generate" | "multi-env" | "drift" | "sweep";

/**
 * What a run's progress counts.
 *
 * A dataset run counts shots. A sweep counts sinter tasks, because `max_errors` stops a
 * sweep and `max_shots` is only a ceiling, so its shot total is not knowable in advance.
 * The unit travels with the number so a bar never labels one thing while counting another.
 */
export type ProgressUnit = "shots" | "tasks";

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
  default_decoders: string[];
  cpu_count: number;
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
  kind: "dataset";
  path: string;
  shots: number;
  content_hash: string | null;
  drift_condition: string;
  structure_source_environment_id: number | null;
}

/**
 * One of a sweep's three outputs.
 *
 * Shares only `path` with a dataset: a sweep has no shot count, no content hash and no
 * drift condition, so the two are a tagged union rather than one record with half its
 * fields nulled out.
 */
export interface SweepArtifact {
  kind: "sweep_results" | "sweep_plot" | "sweep_summary";
  path: string;
}

export type RunArtifact = WrittenFile | SweepArtifact;

export interface RunRecord {
  id: string;
  mode: Mode;
  spec: Record<string, unknown>;
  status: RunStatus;
  total_units: number;
  completed_units: number;
  progress_unit: ProgressUnit;
  phase: string | null;
  /** Free-form line about what is happening now. For a sweep, sinter's own status line. */
  detail: string | null;
  /** Sweep only. A dataset run's shot count is `completed_units` itself. */
  shots_collected: number | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  files: RunArtifact[];
  warnings: string[];
  error: string | null;
  error_kind: string | null;
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
  error_rates: number[];
  distances: number[];
  decoders: string[];
  n_tasks: number;
  max_shots_total: number;
  results_path: string;
  plot_path: string;
  summary_path: string;
  overwrites: boolean;
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

export interface SuppressionFit {
  p: number;
  lambda: number | null;
  lambda_low: number | null;
  lambda_high: number | null;
  prefactor: number | null;
  distances_used: number[];
  excluded_zero_error: number[];
  residual_dof: number;
  reduced_chi_square: number | null;
  suppressing: boolean;
}

export interface DecoderSummary {
  crossing_p: number | null;
  crossing_method: string;
  suppression: SuppressionFit[];
}

export interface CensoredPoint {
  decoder: string;
  distance: number;
  p: number;
  shots: number;
  errors: number;
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
  decoders: Record<string, DecoderSummary>;
}

export interface SweepDetail {
  stem: string;
  results_path: string;
  plot_path: string | null;
  summary_path: string;
  series: SweepSeries[];
  summary: ThresholdSummary;
}
