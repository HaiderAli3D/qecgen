import { useCallback, useEffect, useMemo, useState } from "react";
import { ApiError, api, followRun } from "../api";
import { Checkbox, Field, NumberField, Select, invalidListTokens, parseList } from "../components/Field";
import { Info } from "../components/Info";
import { ThresholdChart } from "../components/ThresholdChart";
import { EXPLAINERS } from "../explainers";
import { count, elapsed, when } from "../format";
import type { Capabilities, RunRecord, SweepDetail, SweepEntry, SweepPreview } from "../types";
import { TERMINAL } from "../types";

interface FormState {
  distances: string;
  p_low: number | "";
  p_high: number | "";
  p_count: number | "";
  max_errors: number | "";
  max_shots: number | "";
  workers: number | "";
  decoders: string[];
  noise_model: string;
  basis: string;
  rotated: boolean;
  out: string;
}

function initial(caps: Capabilities): FormState {
  return {
    distances: "3, 5",
    p_low: 0.001,
    p_high: 0.02,
    p_count: 8,
    max_errors: 500,
    max_shots: 100000000,
    // A sweep forks a pool of its own on top of the worker process the job already owns,
    // so it does not default to every core.
    workers: Math.max(1, caps.cpu_count - 2),
    decoders: [...caps.default_sweep_decoders],
    noise_model: "stim_uniform_circuit_level",
    basis: "z",
    rotated: true,
    out: "sweeps/sweep.csv",
  };
}

function buildBody(form: FormState): Record<string, unknown> | null {
  // A token that fails to parse blocks the body outright rather than narrowing the sweep
  // silently -- the same rule the dataset form applies to its lists.
  if (invalidListTokens(form.distances).length > 0) return null;
  const distances = parseList(form.distances);
  if (distances.length === 0) return null;
  if (form.p_low === "" || form.p_high === "" || form.p_count === "") return null;
  if (form.max_errors === "" || form.max_shots === "" || form.workers === "") return null;
  if (form.decoders.length === 0 || !form.out) return null;
  return {
    mode: "sweep",
    distances,
    p_low: form.p_low,
    p_high: form.p_high,
    p_count: form.p_count,
    max_errors: form.max_errors,
    max_shots: form.max_shots,
    workers: form.workers,
    decoders: form.decoders,
    noise_model: form.noise_model,
    basis: form.basis,
    rotated: form.rotated,
    out: form.out,
  };
}

function Summary({ detail }: { detail: SweepDetail }) {
  const summary = detail.summary;
  const decoders = Object.entries(summary.decoders);
  return (
    <>
      {decoders.map(([name, entry]) => (
        <div key={name} className="panel" style={{ padding: "1.25rem" }}>
          <div className="section-head">
            <h3>{name}</h3>
            <span className="tag">
              {/* `== null` catches undefined as well: a sidecar written by another
                  version, missing the field, would otherwise call toPrecision on
                  undefined and blank the page during render. */}
              {entry.crossing_p == null
                ? "no crossing in range"
                : `crossing p ≈ ${entry.crossing_p.toPrecision(3)}`}
            </span>
          </div>
          <p className="note">{entry.crossing_method}</p>
          {entry.suppression.length === 0 ? (
            <p className="note" style={{ marginTop: "0.6rem" }}>
              No suppression fit: fewer than two distances with errors.
            </p>
          ) : (
            <table style={{ marginTop: "0.6rem" }}>
              <thead>
                <tr>
                  <th className="num">p</th>
                  <th className="num">Λ</th>
                  {/* Derived, because these are the FIT intervals and the sidecar records
                      the alpha they were computed at. Hardcoding 95 would start lying the
                      day alpha becomes settable. */}
                  <th>{Math.round((1 - summary.alpha) * 100)}% interval</th>
                  <th>Suppressing</th>
                </tr>
              </thead>
              <tbody>
                {entry.suppression.map((fit) => (
                  <tr key={fit.p}>
                    <td className="num">{fit.p.toPrecision(3)}</td>
                    <td className="num">{fit.lambda === null ? "—" : fit.lambda.toPrecision(3)}</td>
                    <td>
                      {fit.lambda_low === null || fit.lambda_high === null
                        ? "—"
                        : `[${fit.lambda_low.toPrecision(3)}, ${fit.lambda_high.toPrecision(3)}]`}
                    </td>
                    <td>{fit.suppressing ? "yes" : "no"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      ))}

      {summary.censored_points.length > 0 && (
        <div className="panel" style={{ padding: "1.25rem" }}>
          <h3>Censored points</h3>
          <span className="flag">
            {summary.censored_points.length} point(s) hit the shot ceiling before reaching the
            error target. Their intervals are the widest in the file, and they are the ones
            most likely to be over-trusted.
          </span>
          <table style={{ marginTop: "0.6rem" }}>
            <thead>
              <tr>
                <th>Decoder</th>
                <th className="num">d</th>
                <th className="num">p</th>
                <th className="num">Shots</th>
                <th className="num">Errors</th>
              </tr>
            </thead>
            <tbody>
              {summary.censored_points.map((point, index) => (
                <tr key={index}>
                  <td>{point.decoder}</td>
                  <td className="num">{point.distance}</td>
                  <td className="num">{point.p.toPrecision(3)}</td>
                  <td className="num">{count(point.shots)}</td>
                  <td className="num">{count(point.errors)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="panel" style={{ padding: "1.25rem" }}>
        <h3>Conditions</h3>
        <dl className="readout">
          <dt>Noise model</dt>
          <dd>{summary.noise_model}</dd>
          <dt>Basis</dt>
          <dd>{summary.basis}</dd>
          <dt>Max errors</dt>
          <dd>{count(summary.max_errors)}</dd>
          <dt>Max shots</dt>
          <dd>{count(summary.max_shots)}</dd>
          <dt>Confidence</dt>
          <dd>{`${Math.round((1 - summary.alpha) * 100)}%`}</dd>
        </dl>
        <span className="flag flag--calm">{summary.reported_not_asserted}</span>
        <p className="note" style={{ marginTop: "0.6rem" }}>
          Decoders saw: {summary.dem_seen_by_decoders}
        </p>
      </div>
    </>
  );
}

function Result({ detail }: { detail: SweepDetail }) {
  const [view, setView] = useState<"chart" | "png">("chart");
  return (
    <div className="stack">
      <div className="panel" style={{ padding: "1.25rem" }}>
        <div className="section-head">
          <h2 className="truncate" title={detail.stem}>
            {detail.stem}
          </h2>
          <div className="segmented">
            <button type="button" aria-pressed={view === "chart"} onClick={() => setView("chart")}>
              Interactive
            </button>
            <button
              type="button"
              aria-pressed={view === "png"}
              onClick={() => setView("png")}
              disabled={detail.plot_path === null}
            >
              Plot
            </button>
          </div>
        </div>

        {view === "chart" ? (
          <ThresholdChart series={detail.series} />
        ) : detail.plot_path ? (
          <img
            className="sweep-plot"
            src={api.plotUrl(detail.plot_path, detail.plot_modified_at ?? 0)}
            alt="Threshold plot"
          />
        ) : (
          <p className="empty">This sweep has no plot beside its results table.</p>
        )}

        <p className="note" style={{ marginTop: "0.6rem" }}>
          {view === "chart" ? (
            <>
              An interactive view of <code>{detail.results_path}</code>. Every value drawn is
              read from that table — nothing here recomputes a rate or an interval.{" "}
              {detail.plot_path && (
                <>
                  <code>{detail.plot_path}</code> is the artifact of record.
                </>
              )}
            </>
          ) : (
            <>
              <code>{detail.plot_path}</code>, exactly as written to disk. This is the artifact
              of record and the one to put in a paper.
            </>
          )}
        </p>

        <div className="row" style={{ marginTop: "0.85rem" }}>
          <a className="button" href={api.downloadUrl(detail.results_path)}>
            Download results CSV
          </a>
          <a className="button" href={api.downloadUrl(detail.summary_path)}>
            Download summary JSON
          </a>
        </div>
      </div>

      <Summary detail={detail} />
    </div>
  );
}

interface Props {
  caps: Capabilities;
}

export function Sweeps({ caps }: Props) {
  const [form, setForm] = useState<FormState>(() => initial(caps));
  const [preview, setPreview] = useState<SweepPreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [entries, setEntries] = useState<SweepEntry[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<SweepDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [live, setLive] = useState<RunRecord | null>(null);
  /**
   * Bumped when the selected sweep's FILE changes without its path changing.
   *
   * Re-running onto the same stem is the normal workflow — the default `out` is a
   * constant and the preview says all three files are replaced — so the new summary path
   * is string-identical to the old one, React bails out of the `setSelected`, and an
   * effect keyed only on the path never refetches. The panel would then show the previous
   * sweep's chart, crossing and Λ table under the new sweep's name, with a `succeeded`
   * badge and a fresh timestamp beside it and nothing to indicate the mismatch.
   */
  const [reload, setReload] = useState(0);

  const set = <K extends keyof FormState>(key: K, value: FormState[K]) =>
    setForm((current) => ({ ...current, [key]: value }));

  const body = useMemo(() => buildBody(form), [form]);
  const bodyKey = JSON.stringify(body);

  const refreshSweeps = useCallback(() => {
    api
      .sweeps()
      .then((found) => {
        setEntries(found);
        setListError(null);
        // Land on the newest sweep so the page has something to show on arrival.
        setSelected((current) => current ?? found[0]?.summary_path ?? null);
      })
      // Report the failure; do NOT clear `entries`. Setting it to [] renders "No sweeps
      // yet", so a server restart or a dropped connection would claim the data root is
      // empty — and because this also runs when a sweep finishes, a blip at that moment
      // would say "no sweeps" immediately after one succeeded.
      .catch((error: unknown) =>
        setListError(error instanceof ApiError ? error.message : String(error)),
      );
  }, []);

  useEffect(refreshSweeps, [refreshSweeps]);

  useEffect(() => {
    if (!body) {
      setPreview(null);
      setPreviewError(null);
      return;
    }
    let alive = true;
    const timer = window.setTimeout(() => {
      api
        .sweepPreview(body)
        .then((result) => {
          if (!alive) return;
          setPreview(result);
          setPreviewError(null);
        })
        .catch((error: unknown) => {
          if (!alive) return;
          setPreview(null);
          setPreviewError(error instanceof ApiError ? error.message : String(error));
        });
    }, 220);
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [bodyKey, body]);

  useEffect(() => {
    if (!selected) {
      setDetail(null);
      return;
    }
    let alive = true;
    api
      .sweep(selected)
      .then((found) => {
        if (!alive) return;
        setDetail(found);
        setDetailError(null);
      })
      .catch((error: unknown) => {
        if (!alive) return;
        setDetail(null);
        setDetailError(error instanceof ApiError ? error.message : String(error));
      });
    return () => {
      alive = false;
    };
  }, [selected, reload]);

  // Adopt a sweep that is already in flight. `live` is component state, so without this a
  // trip to another tab and back would lose the running sweep, re-arm Start, and let a
  // second sweep queue up and commit its staged triple over the first one's results. Also
  // picks up a sweep started from the CLI or a second browser tab.
  useEffect(() => {
    let alive = true;
    api
      .runs()
      .then((records) => {
        if (!alive) return;
        const inflight = records.find(
          (record) => record.mode === "sweep" && !TERMINAL.includes(record.status),
        );
        if (inflight) setLive((current) => current ?? inflight);
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, []);

  // Follow the running sweep. When it finishes, re-read the listing so its result appears
  // without a manual refresh, and select it.
  //
  // Keyed on the id STRING, never on the record. Depending on `live` made this effect its
  // own trigger: the callback replaces the record with a freshly parsed object, the
  // reference changes, the effect tears down and resubscribes -- and because a new
  // EventSource replays the whole event log from zero, the cycle sustained itself at one
  // reconnect per progress tick, re-streaming a log that grows with the run. That is
  // exactly the cost `followRun`'s coalescing exists to avoid. `Runs.tsx` depends on an id
  // string for the same reason.
  const liveId = live && !TERMINAL.includes(live.status) ? live.id : null;

  useEffect(() => {
    if (!liveId) return;
    return followRun(liveId, () => {
      api
        .run(liveId)
        .then((record) => {
          setLive(record);
          setListError(null);
          if (!TERMINAL.includes(record.status)) return;
          const summary = record.artifacts.find(
            (artifact) => artifact.kind === "summary",
          );
          refreshSweeps();
          if (!summary) return;
          // The record stores absolute paths; the browse routes are root-relative.
          const name = summary.path.split(/[\\/]/).pop();
          api
            .sweeps()
            .then((found) => {
              const match = found.find((entry) => entry.summary_path.endsWith(name ?? ""));
              if (match) setSelected(match.summary_path);
              // Unconditional, and the whole point: on a re-run onto the same stem the
              // path is unchanged, so only this forces the panel to re-read the file that
              // was just replaced underneath it.
              setReload((n) => n + 1);
            })
            .catch((error: unknown) =>
              setListError(error instanceof ApiError ? error.message : String(error)),
            );
        })
        // Losing contact with a run in flight is not benign: nothing else writes `live`,
        // so a silent swallow leaves the page showing "running" forever with Start
        // disabled, indistinguishable from a slow d=7 task.
        .catch((error: unknown) =>
          setListError(error instanceof ApiError ? error.message : String(error)),
        );
    });
  }, [liveId, refreshSweeps]);

  async function submit() {
    if (!body) return;
    setBusy(true);
    setSubmitError(null);
    try {
      setLive(await api.submit(body));
    } catch (error) {
      setSubmitError(error instanceof ApiError ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function cancel() {
    if (!live) return;
    try {
      setLive(await api.cancel(live.id));
    } catch (error) {
      // Reported, not swallowed: nothing else here re-reads the record, so a refused
      // cancel (409 already finished, 404, network) would otherwise leave the button
      // unchanged and the user unable to tell a rejection from a slow acknowledgement.
      setSubmitError(error instanceof ApiError ? error.message : String(error));
    }
  }

  const running = liveId !== null;
  const fraction = live && live.total_units > 0 ? live.completed_units / live.total_units : 0;

  return (
    <div className="run-layout">
      <div className="stack">
        <div className="panel form-card">
          <fieldset style={{ borderTop: "none" }}>
            <legend>
              <span className="field-head">
                Grid
                <Info topic={EXPLAINERS.distances} />
              </span>
            </legend>
            <div className="grid">
              <Field
                label="Distances"
                topic={EXPLAINERS.distances}
                hint="Comma separated. At least two to see a crossing."
                error={
                  invalidListTokens(form.distances).length > 0
                    ? `Not numbers: ${invalidListTokens(form.distances).join(", ")}`
                    : undefined
                }
              >
                {(id) => (
                  <input
                    id={id}
                    value={form.distances}
                    autoComplete="off"
                    onChange={(event) => set("distances", event.target.value)}
                  />
                )}
              </Field>
              <NumberField
                label="p from"
                topic={EXPLAINERS.p_range}
                step="0.001"
                value={form.p_low}
                onChange={(v) => set("p_low", v)}
              />
              <NumberField
                label="p to"
                topic={EXPLAINERS.p_range}
                step="0.001"
                value={form.p_high}
                onChange={(v) => set("p_high", v)}
              />
              <NumberField
                label="Steps"
                topic={EXPLAINERS.p_range}
                min={1}
                value={form.p_count}
                onChange={(v) => set("p_count", v)}
              />
            </div>
            {preview && (
              <p className="note" style={{ marginTop: "0.6rem" }}>
                Rates: {preview.error_rates.map((rate) => rate.toPrecision(3)).join(", ")}
              </p>
            )}
          </fieldset>

          <fieldset>
            <legend>
              <span className="field-head">
                Decoders
                <Info topic={EXPLAINERS.decoders} />
              </span>
            </legend>
            <div className="grid decoder-list">
              {caps.decoders.map((decoder) => (
                <Checkbox
                  key={decoder.name}
                  label={decoder.name}
                  checked={form.decoders.includes(decoder.name)}
                  disabled={!decoder.usable}
                  onChange={(on) =>
                    set(
                      "decoders",
                      on
                        ? [...form.decoders, decoder.name]
                        : form.decoders.filter((name) => name !== decoder.name),
                    )
                  }
                />
              ))}
            </div>
            {caps.decoders
              .filter((decoder) => !decoder.usable)
              .map((decoder) => (
                <span className="flag" key={decoder.name}>
                  {decoder.problem}
                </span>
              ))}
            {form.decoders.length === 0 && (
              <span className="flag flag--bad">Select at least one decoder.</span>
            )}
          </fieldset>

          <fieldset>
            <legend>Stopping</legend>
            <div className="grid">
              <NumberField
                label="Max errors"
                topic={EXPLAINERS.max_errors}
                min={1}
                value={form.max_errors}
                onChange={(v) => set("max_errors", v)}
                hint="Per task. The real stopping condition."
              />
              <NumberField
                label="Max shots"
                topic={EXPLAINERS.max_shots}
                min={1}
                value={form.max_shots}
                onChange={(v) => set("max_shots", v)}
                hint="Per task. A ceiling, not a target."
              />
              <NumberField
                label="Workers"
                topic={EXPLAINERS.workers}
                min={1}
                value={form.workers}
                onChange={(v) => set("workers", v)}
                hint={`${caps.cpu_count} cores available.`}
              />
            </div>
          </fieldset>

          <fieldset>
            <legend>Circuit</legend>
            <div className="grid">
              <Select
                label="Noise model"
                topic={EXPLAINERS.noise_model}
                value={form.noise_model}
                options={caps.noise_models}
                onChange={(v) => set("noise_model", v)}
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
              <Field
                label="Results file"
                topic={EXPLAINERS.sweep_out}
                hint={`Relative to ${caps.data_root}. Plot and summary land beside it.`}
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
            {/* Not unconditional: code capacity measures perfectly, so `default_rounds`
                returns 1 for it rather than the distance. Stating the memory-experiment
                default here regardless would contradict the noise-model explainer on this
                same fieldset. */}
            <p className="note" style={{ marginTop: "0.6rem" }}>
              {form.noise_model === "code_capacity"
                ? "Code capacity measures perfectly, so every task runs a single round."
                : "Rounds are the distance for every task, the memory-experiment default."}
            </p>
          </fieldset>
        </div>

        {detailError && <span className="flag flag--bad">{detailError}</span>}
        {detail && <Result detail={detail} />}

        <div className="panel">
          <h3 style={{ padding: "0 1.25rem" }}>Sweeps on disk</h3>
          {listError && (
            <span className="flag flag--bad" style={{ margin: "0 1.25rem" }}>
              Could not read the sweep listing: {listError}
            </span>
          )}
          {entries === null && !listError && <p className="empty">Loading…</p>}
          {entries !== null && entries.length === 0 && !listError && (
            <p className="empty">No sweeps yet. Configure one above.</p>
          )}
          {entries !== null && entries.length > 0 && (
            <table>
              <thead>
                <tr>
                  <th>Sweep</th>
                  <th>Decoders</th>
                  <th>Crossing</th>
                  <th>Plot</th>
                  <th>Modified</th>
                </tr>
              </thead>
              <tbody>
                {entries.map((entry) => (
                  <tr
                    key={entry.summary_path}
                    className="clickable"
                    role="button"
                    tabIndex={0}
                    aria-pressed={entry.summary_path === selected}
                    onClick={() => setSelected(entry.summary_path)}
                    onKeyDown={(event) => {
                      if (event.key !== "Enter" && event.key !== " ") return;
                      event.preventDefault();
                      setSelected(entry.summary_path);
                    }}
                  >
                    <td className="truncate" title={entry.stem}>
                      {entry.stem}
                    </td>
                    {/* A corrupt sidecar must not read as a valid sweep that found no
                        crossing — both would otherwise render as an em dash. The server
                        computes the reason; showing it is what makes the flag mean
                        something. Mirrors the dataset browser. */}
                    {entry.unreadable ? (
                      <td colSpan={4}>
                        <span className="flag flag--bad" style={{ marginTop: 0 }}>
                          Unreadable: {entry.unreadable}
                        </span>
                      </td>
                    ) : (
                      <>
                        <td>{entry.decoders.join(", ") || "—"}</td>
                        <td>
                          {Object.entries(entry.crossings)
                            // `== null` catches undefined too: a sidecar missing the field
                            // would otherwise reach toPrecision on undefined and take the
                            // whole page down with it.
                            .map(([name, p]) => `${name}: ${p == null ? "none" : p.toPrecision(3)}`)
                            .join("; ") || "—"}
                        </td>
                        <td>{entry.plot_path ? "yes" : "missing"}</td>
                        <td>{when(new Date(entry.modified_at * 1000).toISOString())}</td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <aside className="aside">
        <div className="panel">
          <h3>This sweep</h3>
          {preview ? (
            <dl className="readout">
              <dt>Tasks</dt>
              <dd className="big">{count(preview.n_tasks)}</dd>
              <dt>Distances</dt>
              <dd>{preview.distances.join(", ")}</dd>
              <dt>Rates</dt>
              <dd>{preview.error_rates.length}</dd>
              <dt>Decoders</dt>
              {/* Objects, not strings: the preview reports each decoder's backend
                  availability beside its name, so an absent package is named here
                  rather than discovered inside a worker. */}
              <dd>{preview.decoders.map((entry) => entry.name).join(", ")}</dd>
              <dt>Shot ceiling</dt>
              <dd>{count(preview.max_shots_total)}</dd>
            </dl>
          ) : (
            <p className="note">{previewError ?? "Fill in the grid to see what it will run."}</p>
          )}
          <p className="note" style={{ marginTop: "0.6rem" }}>
            No time estimate: a sweep stops on errors, so how long it takes depends on the
            logical error rate it exists to measure.
          </p>
          {preview?.overwrites && (
            <span className="flag">
              A sweep already exists at this stem. All three files are replaced together.
            </span>
          )}
          <hr className="rule" />
          <button className="primary" onClick={submit} disabled={!body || busy || running}>
            {busy ? "Starting…" : running ? "Sweep running…" : "Start sweep"}
          </button>
          {submitError && <span className="flag flag--bad">{submitError}</span>}
        </div>

        {live && (
          <div className="panel">
            <div className="section-head">
              <h3>Progress</h3>
              <span className={`status status--${live.status}`}>{live.status}</span>
            </div>
            {/* The status half of this condition is load-bearing: `phase` is never
                cleared on a terminal event, so keying on it alone left an animated
                "still writing" bar above a red Failed flag, and above every success. */}
            <span className={`bar${running && live.phase === "writing" ? " bar--indeterminate" : ""}`}>
              <span
                style={
                  running && live.phase === "writing" ? undefined : { width: `${fraction * 100}%` }
                }
              />
            </span>
            <dl className="readout" style={{ marginTop: "0.85rem" }}>
              <dt>Tasks</dt>
              <dd>
                {count(live.completed_units)} / {count(live.total_units)}
              </dd>
              <dt>Shots collected</dt>
              <dd>{live.shots_collected === null ? "—" : count(live.shots_collected)}</dd>
              <dt>Phase</dt>
              <dd>{live.phase ?? "—"}</dd>
              <dt>Elapsed</dt>
              <dd>{elapsed(live.started_at, live.finished_at)}</dd>
            </dl>
            {live.detail && <pre className="mono-block sinter-status">{live.detail}</pre>}
            {live.error && (
              <span className="flag flag--bad">
                {live.error_kind === "input" ? "Rejected: " : "Failed: "}
                {live.error}
              </span>
            )}
            {running && (
              <div className="row" style={{ marginTop: "0.85rem" }}>
                <button className="danger" onClick={cancel}>
                  {live.status === "cancelling" ? "Cancelling…" : "Cancel sweep"}
                </button>
              </div>
            )}
            <p className="note" style={{ marginTop: "0.6rem" }}>
              Progress counts sinter tasks, not shots: a sweep stops on errors, so its shot
              total is not knowable in advance.
            </p>
          </div>
        )}
      </aside>
    </div>
  );
}
