import type {
  Capabilities,
  CorrectionEntry,
  CorrectionSchema,
  DatasetEntry,
  FieldError,
  Preview,
  Provenance,
  RunRecord,
  RunStatus,
  SweepDetail,
  SweepEntry,
  SweepPreview,
  ValidationReport,
} from "./types";
import { TERMINAL } from "./types";

/**
 * The server answers a bad request two ways, and they mean different things.
 *
 * 422 is pydantic: the body broke a field constraint, and `fields` says which. The form
 * can point at the offending input.
 * 400 is everything else — a path that escapes the data root, an unknown format — and
 * carries one message for the whole form.
 *
 * Domain rules that need a built circuit (code_capacity with rounds != 1, environments
 * that will not stack) cannot be checked at submit time at all. Those arrive later, as a
 * failed run whose error is the domain's own sentence.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly fields: FieldError[];

  constructor(status: number, message: string, fields: FieldError[] = []) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.fields = fields;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    let detail: unknown = await response.text();
    try {
      detail = JSON.parse(detail as string);
    } catch {
      // A non-JSON body is the 503 from an unbuilt frontend, or a proxy error page.
    }
    const payload = (detail as { detail?: unknown })?.detail ?? detail;
    if (Array.isArray(payload)) {
      const fields = payload as FieldError[];
      const first = fields[0];
      const label = first ? String(first.loc[first.loc.length - 1]) : "input";
      throw new ApiError(
        response.status,
        `${label}: ${first?.msg ?? "invalid"}`,
        fields,
      );
    }
    throw new ApiError(response.status, String(payload));
  }
  return (await response.json()) as T;
}

export const api = {
  capabilities: () => request<Capabilities>("/api/capabilities"),

  preview: (body: unknown) =>
    request<Preview>("/api/preview", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  submit: (body: unknown) =>
    request<RunRecord>("/api/runs", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  runs: () => request<RunRecord[]>("/api/runs"),

  run: (id: string) => request<RunRecord>(`/api/runs/${id}`),

  cancel: (id: string) =>
    request<RunRecord>(`/api/runs/${id}/cancel`, { method: "POST" }),

  datasets: () => request<DatasetEntry[]>("/api/datasets"),

  manifest: (path: string) =>
    request<Record<string, unknown>>(
      `/api/datasets/manifest?path=${encodeURIComponent(path)}`,
    ),

  validate: (path: string) =>
    request<ValidationReport>("/api/datasets/validate", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),

  downloadUrl: (path: string) =>
    `/api/datasets/download?path=${encodeURIComponent(path)}`,

  /**
   * Circuit and DEM text. Called only from an explicit control, never alongside the
   * manifest — under a frozen prior that text is exactly what the condition withholds,
   * and a page that loads both because both were available has defeated the separation
   * the server went to the trouble of building.
   */
  provenance: (path: string) =>
    request<Provenance>(
      `/api/datasets/provenance?path=${encodeURIComponent(path)}`,
    ),

  correctionSchema: (path: string) =>
    request<CorrectionSchema>(
      `/api/datasets/correction-schema?path=${encodeURIComponent(path)}`,
    ),

  corrections: () => request<CorrectionEntry[]>("/api/corrections"),

  sweepPreview: (body: unknown) =>
    request<SweepPreview>("/api/sweeps/preview", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  sweeps: () => request<SweepEntry[]>("/api/sweeps"),

  sweep: (path: string) =>
    request<SweepDetail>(`/api/sweeps/detail?path=${encodeURIComponent(path)}`),

  /**
   * A sweep plot, as an `<img>` source.
   *
   * Its own route rather than `downloadUrl`, which sets `Content-Disposition: attachment`
   * because a dataset is a file you save. The `v` parameter is the file's mtime: a re-run
   * sweep writes the *same path*, so without it a cached image would show the previous
   * run's plot.
   */
  plotUrl: (path: string, version: number) =>
    `/api/sweeps/plot?path=${encodeURIComponent(path)}&v=${Math.round(version)}`,
};

const REFRESH_COALESCE_MS = 200;
const RETRY_BASE_MS = 500;
const RETRY_MAX_MS = 15_000;

/**
 * Follow one run's events until it reaches a terminal state.
 *
 * The server keeps a replayable per-run event log, which is why reconnection is owned
 * here rather than left to `EventSource`: a fresh subscription can always start from
 * nothing and catch up, whereas the built-in retry parks permanently in CLOSED after a
 * hard failure (non-200, server briefly down) and cannot be told to stop once the run
 * is finished. The terminal `status` event is the only reliable stop signal — the
 * server ends the stream when a run finishes, and `EventSource` reports any end of
 * stream as an error, indistinguishable from a dropped connection.
 */
export function followRun(id: string, onEvent: () => void): () => void {
  let source: EventSource | null = null;
  let stopped = false;
  let retryMs = RETRY_BASE_MS;
  let refreshTimer: number | null = null;
  let retryTimer: number | null = null;

  // A subscription replays the whole log, and a long run's log holds thousands of
  // progress events; refreshing per event would turn one replay into that many
  // GET /api/runs. Trailing-edge coalescing makes a burst of any size cost one request.
  const refresh = () => {
    if (refreshTimer !== null) return;
    refreshTimer = window.setTimeout(() => {
      refreshTimer = null;
      onEvent();
    }, REFRESH_COALESCE_MS);
  };

  const connect = () => {
    retryTimer = null;
    const stream = new EventSource(`/api/runs/${id}/events`);
    source = stream;
    stream.onopen = () => {
      retryMs = RETRY_BASE_MS;
    };
    const handler = () => refresh();
    for (const kind of ["started", "progress", "phase", "warning"]) {
      stream.addEventListener(kind, handler);
    }
    stream.addEventListener("status", (event) => {
      refresh();
      try {
        const payload = JSON.parse(
          String((event as MessageEvent<string>).data),
        ) as {
          status?: RunStatus;
        };
        if (payload.status && TERMINAL.includes(payload.status)) {
          stopped = true;
          stream.close();
        }
      } catch {
        // Unparseable data. Keep following; the refresh above still converges the page.
      }
    });
    stream.onerror = () => {
      // Closed without a terminal status means transient — the server restarted, the
      // laptop slept — so resubscribe with backoff rather than going silent mid-run.
      stream.close();
      if (stopped) return;
      refresh();
      retryTimer = window.setTimeout(connect, retryMs);
      retryMs = Math.min(retryMs * 2, RETRY_MAX_MS);
    };
  };

  connect();
  return () => {
    stopped = true;
    if (refreshTimer !== null) window.clearTimeout(refreshTimer);
    if (retryTimer !== null) window.clearTimeout(retryTimer);
    source?.close();
  };
}
