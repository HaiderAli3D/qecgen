import { useEffect, useMemo, useState } from "react";
import { ApiError, api } from "../api";
import { Checkbox, ListField, NumberField, Select } from "../components/Field";
import { EXPLAINERS } from "../explainers";
import { invalidListTokens, parseList } from "../components/Field";
import { when } from "../format";
import type {
  Capabilities,
  SweepEntry,
  SweepPreview,
  ThresholdSummary,
} from "../types";

interface Props {
  caps: Capabilities;
  onSubmitted: (id: string) => void;
}

interface FormState {
  distances: string;
  p_low: number | "";
  p_high: number | "";
  p_count: number | "";
  max_errors: number | "";
  max_shots: number | "";
  workers: number | "";
  noise_model: string;
  basis: string;
  decoders: string[];
  out: string;
}

function initial(caps: Capabilities): FormState {
  return {
    distances: "3, 5, 7",
    p_low: 0.001,
    p_high: 0.02,
    p_count: 8,
    max_errors: 500,
    max_shots: 100000000,
    workers: 4,
    noise_model: "stim_uniform_circuit_level",
    basis: "z",
    decoders: [...caps.default_sweep_decoders],
    out: "sweeps/sweep.csv",
  };
}

function buildBody(form: FormState): Record<string, unknown> | null {
  const distances = parseList(form.distances);
  if (invalidListTokens(form.distances).length > 0 || distances.length === 0)
    return null;
  if (form.p_low === "" || form.p_high === "" || form.p_count === "")
    return null;
  if (form.max_errors === "" || form.max_shots === "" || form.workers === "")
    return null;
  if (!form.out || form.decoders.length === 0) return null;
  return {
    mode: "sweep",
    distances: distances.map((value) => Math.round(value)),
    p_low: form.p_low,
    p_high: form.p_high,
    p_count: form.p_count,
    max_errors: form.max_errors,
    max_shots: form.max_shots,
    workers: form.workers,
    noise_model: form.noise_model,
    basis: form.basis,
    decoders: form.decoders,
    out: form.out,
  };
}

/**
 * The crossing and suppression table.
 *
 * Reads the same payload that goes into the `.threshold.json` sidecar rather than
 * recomputing anything from the CSV — a second implementation of the crossing rule is
 * exactly what the domain keeps singular.
 */
export function ThresholdReport({ summary }: { summary: ThresholdSummary }) {
  return (
    <div className="stack">
      {Object.entries(summary.decoders).map(([decoder, entry]) => (
        <div key={decoder}>
          <div className="section-head">
            <h3>{decoder}</h3>
            <span className="tag">
              {entry.crossing_p === null
                ? "no crossing in range"
                : `crossing p ≈ ${entry.crossing_p.toPrecision(3)}`}
            </span>
          </div>
          {entry.suppression.length === 0 ? (
            <p className="note">
              No suppression fit: fewer than two distances produced any
              failures.
            </p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th className="num">p</th>
                  <th className="num">Λ</th>
                  <th>Interval</th>
                  <th>Distances</th>
                  <th>Suppressing</th>
                </tr>
              </thead>
              <tbody>
                {entry.suppression.map((row) => {
                  // Λ only means anything below threshold. Saying which side each row is
                  // on stops a super-threshold number reading like a suppression result.
                  const above =
                    entry.crossing_p !== null && row.p >= entry.crossing_p;
                  return (
                    <tr key={row.p}>
                      <td className="num">{row.p.toPrecision(3)}</td>
                      <td className="num">{row.lambda.toPrecision(3)}</td>
                      <td>
                        [{row.lambda_low.toPrecision(3)},{" "}
                        {row.lambda_high.toPrecision(3)}]
                        {row.reduced_chi_square !== null
                          ? ` χ²/dof=${row.reduced_chi_square.toPrecision(3)}`
                          : ""}
                      </td>
                      <td>
                        {row.distances_used.join(", ")}
                        {row.excluded_zero_error.length > 0
                          ? ` (excluded k=0: ${row.excluded_zero_error.join(", ")})`
                          : ""}
                      </td>
                      <td>
                        {above ? (
                          <span className="tag">at or above crossing</span>
                        ) : row.suppressing ? (
                          "yes"
                        ) : (
                          "not established"
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      ))}

      {summary.censored_points.length > 0 && (
        <span className="flag">
          {summary.censored_points.length} point(s) hit the shot ceiling before
          the error target, so their intervals are the widest in the file:{" "}
          {summary.censored_points
            .map((point) => `${point.decoder} d=${point.distance} p=${point.p}`)
            .join("; ")}
          .
        </span>
      )}

      <p className="note">{summary.reported_not_asserted}</p>
      <p className="note">{summary.dem_seen_by_decoders}</p>
    </div>
  );
}

/** One sweep on disk: its plot, and the numbers behind it. */
function SweepDetail({ entry }: { entry: SweepEntry }) {
  if (entry.summary.unreadable) {
    return <span className="flag flag--bad">{entry.summary.unreadable}</span>;
  }
  return (
    <div className="panel" style={{ padding: "1.25rem", marginTop: "0.75rem" }}>
      <div className="section-head">
        <h2 className="truncate">{entry.name}</h2>
        {entry.csv && (
          <a className="button" href={api.downloadUrl(entry.csv)} download>
            Download CSV
          </a>
        )}
      </div>
      {entry.plot && (
        <figure className="plot">
          {/* The mtime in the query is not decoration: a re-run sweep writes the same
              path, so a cached image would show the previous run's plot. */}
          <img
            src={api.plotUrl(entry.plot, entry.modified_at)}
            alt={`Threshold plot for ${entry.name}`}
          />
        </figure>
      )}
      <ThresholdReport summary={entry.summary} />
    </div>
  );
}

export function Sweep({ caps, onSubmitted }: Props) {
  const [form, setForm] = useState<FormState>(() => initial(caps));
  const [preview, setPreview] = useState<SweepPreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [entries, setEntries] = useState<SweepEntry[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  const set = <K extends keyof FormState>(key: K, value: FormState[K]) =>
    setForm((current) => ({ ...current, [key]: value }));

  const body = useMemo(() => buildBody(form), [form]);
  const bodyKey = JSON.stringify(body);

  useEffect(() => {
    if (!body) {
      setPreview(null);
      setPreviewError(null);
      return;
    }
    let live = true;
    const timer = window.setTimeout(() => {
      api
        .preview(body)
        .then((result) => {
          if (!live) return;
          setPreview(result as unknown as SweepPreview);
          setPreviewError(null);
        })
        .catch((error: unknown) => {
          if (!live) return;
          setPreview(null);
          setPreviewError(
            error instanceof ApiError ? error.message : String(error),
          );
        });
    }, 220);
    return () => {
      live = false;
      window.clearTimeout(timer);
    };
  }, [bodyKey, body]);

  useEffect(() => {
    api
      .sweeps()
      .then(setEntries)
      .catch(() => setEntries([]));
  }, []);

  async function submit() {
    if (!body) return;
    setBusy(true);
    setSubmitError(null);
    try {
      const record = await api.submit(body);
      onSubmitted(record.id);
    } catch (error) {
      setSubmitError(error instanceof ApiError ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  const current = entries?.find((entry) => entry.path === selected);
  const oversubscribed =
    typeof form.workers === "number" &&
    form.workers * caps.max_concurrent_jobs > 32;

  return (
    <>
      <div className="run-layout">
        <div className="panel form-card">
          <fieldset style={{ borderTop: "none" }}>
            <legend>Grid</legend>
            <div className="grid">
              <ListField
                label="Distances"
                topic={EXPLAINERS.distances}
                value={form.distances}
                onChange={(value) => set("distances", value)}
                hint="Comma separated."
                error={
                  invalidListTokens(form.distances).length > 0
                    ? `Not numbers: ${invalidListTokens(form.distances).join(", ")}`
                    : undefined
                }
              />
              <NumberField
                label="Lowest p"
                topic={EXPLAINERS.p_range}
                step="0.001"
                min={0}
                value={form.p_low}
                onChange={(value) => set("p_low", value)}
              />
              <NumberField
                label="Highest p"
                topic={EXPLAINERS.p_range}
                step="0.001"
                min={0}
                value={form.p_high}
                onChange={(value) => set("p_high", value)}
              />
              <NumberField
                label="Points"
                topic={EXPLAINERS.p_range}
                min={1}
                value={form.p_count}
                onChange={(value) => set("p_count", value)}
                hint="Evenly spaced, linearly."
              />
            </div>
          </fieldset>

          <fieldset>
            <legend>Code</legend>
            <div className="grid">
              <Select
                label="Model"
                topic={EXPLAINERS.noise_model}
                value={form.noise_model}
                options={caps.noise_models}
                onChange={(value) => set("noise_model", value)}
              />
              <Select
                label="Memory basis"
                topic={EXPLAINERS.basis}
                value={form.basis}
                options={caps.bases}
                onChange={(value) => set("basis", value)}
              />
            </div>
            <p className="note">
              Rounds are the memory-experiment default of one per unit of
              distance, per task. Every task uses the rotated layout.
            </p>
          </fieldset>

          <fieldset>
            <legend>Decoders</legend>
            <div className="grid">
              {caps.decoders.map((decoder) => (
                <Checkbox
                  key={decoder.name}
                  label={decoder.name}
                  checked={form.decoders.includes(decoder.name)}
                  disabled={!decoder.usable}
                  onChange={(checked) =>
                    set(
                      "decoders",
                      checked
                        ? [...form.decoders, decoder.name]
                        : form.decoders.filter((name) => name !== decoder.name),
                    )
                  }
                />
              ))}
            </div>
            {caps.decoders
              .filter((decoder) => !decoder.usable && decoder.problem)
              .map((decoder) => (
                <span className="flag" key={decoder.name}>
                  {decoder.problem}
                </span>
              ))}
          </fieldset>

          <fieldset>
            <legend>Stopping</legend>
            <div className="grid">
              <NumberField
                label="Target errors"
                topic={EXPLAINERS.max_errors}
                min={1}
                value={form.max_errors}
                onChange={(value) => set("max_errors", value)}
              />
              <NumberField
                label="Shot ceiling"
                topic={EXPLAINERS.max_shots}
                min={1}
                value={form.max_shots}
                onChange={(value) => set("max_shots", value)}
              />
              <NumberField
                label="Workers"
                topic={EXPLAINERS.workers}
                min={1}
                value={form.workers}
                onChange={(value) => set("workers", value)}
              />
            </div>
            {oversubscribed && (
              <span className="flag">
                {form.workers} workers × {caps.max_concurrent_jobs} concurrent
                jobs is more processes than most machines have cores. They will
                contend rather than go faster.
              </span>
            )}
          </fieldset>

          <fieldset>
            <legend>Output</legend>
            <div className="grid">
              <ListField
                label="Output CSV"
                topic={EXPLAINERS.out}
                value={form.out}
                onChange={(value) => set("out", value)}
                hint={`Relative to ${caps.data_root}. The plot and summary are written beside it.`}
              />
            </div>
          </fieldset>
        </div>

        <aside className="aside">
          <div className="panel">
            <h3>Before you run</h3>
            {preview ? (
              <>
                <dl className="readout">
                  <dt>Tasks</dt>
                  <dd className="big">{preview.n_tasks}</dd>
                  <dt>Distances</dt>
                  <dd>{preview.distances.join(", ")}</dd>
                  <dt>Rates</dt>
                  <dd className="wrap">
                    {preview.rates
                      .map((rate) => rate.toPrecision(3))
                      .join(", ")}
                  </dd>
                  <dt>Target errors</dt>
                  <dd>{preview.max_errors}</dd>
                  <dt>Workers</dt>
                  <dd>{preview.workers}</dd>
                </dl>
                {!preview.usable && (
                  <span className="flag flag--bad">
                    {preview.decoders.find((entry) => !entry.usable)?.problem}
                  </span>
                )}
                <span className="flag flag--calm">{preview.note}</span>
              </>
            ) : (
              <p className="note">
                {previewError ?? "Fill in the grid to see its size."}
              </p>
            )}
            <hr className="rule" />
            <button
              className="primary"
              onClick={submit}
              disabled={!body || busy}
            >
              {busy ? "Starting…" : "Start sweep"}
            </button>
            {submitError && (
              <span className="flag flag--bad">{submitError}</span>
            )}
          </div>
          <div className="panel">
            <h3>Resolved configuration</h3>
            <pre className="mono-block">
              {JSON.stringify(body ?? {}, null, 2)}
            </pre>
          </div>
        </aside>
      </div>

      <div className="panel" style={{ marginTop: "1.5rem" }}>
        <div className="section-head" style={{ padding: "1rem 1.25rem 0" }}>
          <h2>Sweeps on disk</h2>
        </div>
        {entries === null ? (
          <p className="empty">Reading sweeps…</p>
        ) : entries.length === 0 ? (
          <p className="empty">
            No sweeps yet. Anything <code>qecgen sweep</code> wrote under the
            data root appears here too.
          </p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Sweep</th>
                <th>Decoders</th>
                <th>Crossing</th>
                <th>Written</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry) => {
                const decoders = entry.summary.decoders ?? {};
                const crossings = Object.entries(decoders)
                  .map(([name, value]) =>
                    value.crossing_p === null
                      ? `${name}: none`
                      : `${name}: ${value.crossing_p}`,
                  )
                  .join("; ");
                return (
                  <tr
                    key={entry.path}
                    className="clickable"
                    role="button"
                    tabIndex={0}
                    aria-pressed={entry.path === selected}
                    onClick={() =>
                      setSelected(entry.path === selected ? null : entry.path)
                    }
                    onKeyDown={(event) => {
                      if (event.key !== "Enter" && event.key !== " ") return;
                      event.preventDefault();
                      setSelected(entry.path === selected ? null : entry.path);
                    }}
                  >
                    <td className="truncate" title={entry.path}>
                      {entry.name}
                    </td>
                    <td>{Object.keys(decoders).join(", ") || "—"}</td>
                    <td>{crossings || "—"}</td>
                    <td>{when(entry.modified_at)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
      {current && <SweepDetail entry={current} />}
    </>
  );
}
