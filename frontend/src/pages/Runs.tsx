import { useEffect, useState } from "react";
import { ApiError, api, followRun } from "../api";
import { Lattice } from "../components/Lattice";
import { ThresholdReport } from "./Sweep";
import { bytes, count, elapsed, shortHash, when } from "../format";
import type { QaEnvironment, RunRecord, ThresholdSummary } from "../types";
import { TERMINAL } from "../types";

function Status({ record }: { record: RunRecord }) {
  return (
    <span className={`status status--${record.status}`}>{record.status}</span>
  );
}

function fraction(record: RunRecord): number {
  if (record.total_shots <= 0) return 0;
  return Math.min(1, record.completed_shots / record.total_shots);
}

function Bar({ record }: { record: RunRecord }) {
  // Two reasons a bar has no meaningful fill, and both must render as motion rather than
  // as a number.
  //
  // The writing phase: nothing reports progress through concatenation, hashing and gzip,
  // so a full bar sitting still reads as a hang.
  //
  // A total of zero: some jobs genuinely cannot know their denominator in advance -- a
  // score reads a dataset it has not opened yet -- and `Math.min(1, n / 0)` is NaN, which
  // this used to floor to 0 and paint as a bar permanently at the left edge. That is
  // indistinguishable from a job that has not started.
  const writing = record.status === "running" && record.phase === "writing";
  const unknown = record.total_shots <= 0;
  const indeterminate = writing || (unknown && record.status === "running");
  return (
    <span className={`bar${indeterminate ? " bar--indeterminate" : ""}`}>
      <span
        style={
          indeterminate ? undefined : { width: `${fraction(record) * 100}%` }
        }
      />
    </span>
  );
}

/**
 * What a run has completed, with its unit.
 *
 * The unit is not decoration. A sweep counts tasks, QA counts shots, and a column headed
 * "Shots" showing a sweep's 6 is a well-formed wrong number -- the reader has no way to
 * know it is not six shots.
 */
function completedText(record: RunRecord): string {
  if (!record.progress_unit) return "—";
  return `${count(record.completed_shots)} ${record.progress_unit}`;
}

/** "12,000 / 48,000 shots", or "—" when the job never had a denominator. */
function progressText(record: RunRecord): string {
  if (record.total_shots <= 0 || !record.progress_unit) return "—";
  return `${count(record.completed_shots)} / ${count(record.total_shots)} ${record.progress_unit}`;
}

const DATASET_MODES: readonly string[] = ["generate", "multi-env", "drift"];

/**
 * What an analysis job produced, rendered per kind.
 *
 * The payload is whatever the domain reported — for a sweep it is the same object that
 * goes into the `.threshold.json` sidecar — so nothing here recomputes a crossing, a
 * Lambda or a rate. Dumping it as JSON would be honest but unreadable; recomputing any of
 * it would be readable but a second source of truth.
 */
function ResultPanel({ result }: { result: Record<string, unknown> }) {
  const kind = String(result.kind ?? "");

  if (kind === "score") {
    const rate = Number(result.logical_error_rate);
    return (
      <div className="panel" style={{ padding: "1.25rem" }}>
        <h3>Logical error rate under the supplied correction</h3>
        <dl className="readout">
          <dt>Rate</dt>
          <dd className="big">{rate.toFixed(5)}</dd>
          <dt>Interval</dt>
          <dd>
            [{Number(result.ci_low).toFixed(5)},{" "}
            {Number(result.ci_high).toFixed(5)}]
          </dd>
          <dt>Failures</dt>
          <dd>
            {count(Number(result.failures))} / {count(Number(result.shots))}
          </dd>
          <dt>Data qubits</dt>
          <dd>{String(result.n_data_qubits)}</dd>
          <dt>Schema</dt>
          <dd>{shortHash(String(result.schema_digest))}</dd>
          <dt>Content hash</dt>
          <dd>{shortHash(String(result.content_hash))}</dd>
        </dl>
        <p className="note">
          A shot succeeds when the correction's induced observable flip equals
          the true flip on every observable. The schema digest identifies the
          qubit ordering this was computed under — the same arrays under a
          different ordering give a different, equally plausible number — and
          the content hash identifies the shots it was computed against.
        </p>
        <span className="flag flag--calm">{String(result.contract)}</span>
      </div>
    );
  }

  if (kind === "qa") {
    const environments = (result.environments ?? []) as QaEnvironment[];
    return (
      <div className="panel" style={{ padding: "1.25rem" }}>
        <h3>Statistical QA</h3>
        {result.ok === false ? (
          <span className="flag flag--bad">{String(result.skipped)}</span>
        ) : (
          <>
            <table>
              <thead>
                <tr>
                  <th>Environment</th>
                  <th className="num">Logical rate</th>
                  <th>Interval</th>
                  <th className="num">Failures</th>
                  <th className="num">Detection rate</th>
                </tr>
              </thead>
              <tbody>
                {environments.map((environment) => (
                  <tr key={environment.environment_id}>
                    <td>
                      {environment.axis}={environment.axis_value}
                    </td>
                    <td className="num">
                      {environment.logical_error_rate.toFixed(5)}
                    </td>
                    <td>
                      [{environment.ci_low.toFixed(5)},{" "}
                      {environment.ci_high.toFixed(5)}]
                    </td>
                    <td className="num">
                      {count(environment.failures)} / {count(environment.shots)}
                    </td>
                    <td className="num">
                      {environment.detection_event_rate.toFixed(5)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="note">{String(result.reported_not_asserted ?? "")}</p>
          </>
        )}
      </div>
    );
  }

  if (kind === "sweep") {
    return (
      <div className="panel" style={{ padding: "1.25rem" }}>
        <h3>Threshold</h3>
        <ThresholdReport summary={result as unknown as ThresholdSummary} />
        <p className="note">
          The full result set, including the plot, is on the{" "}
          <a href="#/sweep">Sweep</a> page.
        </p>
      </div>
    );
  }

  return (
    <div className="panel" style={{ padding: "1.25rem" }}>
      <h3>Result</h3>
      <pre className="mono-block">{JSON.stringify(result, null, 2)}</pre>
    </div>
  );
}

function RunDetail({
  record,
  onChanged,
}: {
  record: RunRecord;
  onChanged: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const spec = record.spec as Record<string, unknown>;
  const live = !TERMINAL.includes(record.status);

  async function cancel() {
    setBusy(true);
    try {
      await api.cancel(record.id);
      onChanged();
    } catch {
      onChanged();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="run-layout" style={{ marginTop: "1rem" }}>
      <div className="stack">
        <div className="panel" style={{ padding: "1.25rem" }}>
          <div className="section-head">
            <h2>{record.mode}</h2>
            <Status record={record} />
          </div>
          <Bar record={record} />
          <dl className="readout" style={{ marginTop: "0.85rem" }}>
            <dt>Progress</dt>
            <dd>{progressText(record)}</dd>
            <dt>Phase</dt>
            <dd>{record.phase ?? "—"}</dd>
            <dt>Elapsed</dt>
            <dd>{elapsed(record.started_at, record.finished_at)}</dd>
            <dt>Started</dt>
            <dd>{when(record.started_at)}</dd>
          </dl>
          {record.error && (
            <span className="flag flag--bad">
              {record.error_kind === "input" ? "Rejected: " : "Failed: "}
              {record.error}
            </span>
          )}
          {record.warnings.map((warning, index) => (
            <span className="flag" key={index}>
              {warning}
            </span>
          ))}
          {live && (
            <div className="row" style={{ marginTop: "0.85rem" }}>
              <button className="danger" onClick={cancel} disabled={busy}>
                {record.status === "cancelling" ? "Cancelling…" : "Cancel run"}
              </button>
              <span className="note">
                Stops at the next chunk boundary. Nothing is left at the output
                path.
              </span>
            </div>
          )}
        </div>

        {record.files.length > 0 && (
          <div className="panel" style={{ padding: "1.25rem" }}>
            <h3>Files written</h3>
            <table>
              <thead>
                <tr>
                  <th>Path</th>
                  <th className="num">Shots</th>
                  <th>Condition</th>
                  <th>Content hash</th>
                </tr>
              </thead>
              <tbody>
                {record.files.map((file) => (
                  <tr key={file.path}>
                    <td className="truncate" title={file.path}>
                      {file.path.split(/[\\/]/).pop()}
                    </td>
                    <td className="num">{count(file.shots)}</td>
                    <td>{file.drift_condition}</td>
                    <td>{shortHash(file.content_hash)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {record.result && <ResultPanel result={record.result} />}

        {/* Its own table, not extra rows in "Files written". Those columns are Shots,
            Condition and Content hash -- a dataset's properties. A plot has none of
            them, and printing an invented shot count beside one is exactly the kind of
            well-formed wrong record this tool exists to avoid producing. */}
        {record.artifacts.length > 0 && (
          <div className="panel" style={{ padding: "1.25rem" }}>
            <h3>Other output</h3>
            <table>
              <thead>
                <tr>
                  <th>Path</th>
                  <th>Kind</th>
                  <th className="num">Size</th>
                </tr>
              </thead>
              <tbody>
                {record.artifacts.map((artifact) => (
                  <tr key={artifact.path}>
                    <td className="truncate" title={artifact.path}>
                      {artifact.path.split(/[\\/]/).pop()}
                    </td>
                    <td>
                      <span className="tag">{artifact.kind}</span>
                    </td>
                    <td className="num">{bytes(artifact.size_bytes)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <aside className="aside">
        {/* Only a dataset run has a single distance to draw. A sweep has several and a
            score has none, so the figure is conditioned on the mode rather than falling
            back to a default -- `Number(spec.distance ?? 3)` would have silently drawn a
            d=3 patch for a d=[3,5,7] sweep, which is a confident picture of the wrong
            thing. */}
        {DATASET_MODES.includes(record.mode) && (
          <div className="panel">
            <Lattice
              distance={Number(spec.distance ?? 3)}
              basis={String(spec.basis ?? "z")}
              progress={fraction(record)}
              label={`${Math.round(fraction(record) * 100)}% sampled`}
            />
            {spec.rotated === false && (
              <p className="note">
                This run uses the unrotated layout; the figure shows the rotated
                one.
              </p>
            )}
          </div>
        )}
        <div className="panel">
          <h3>Resolved configuration</h3>
          <pre className="mono-block">{JSON.stringify(spec, null, 2)}</pre>
          <p className="note">
            The request this run was resolved from, kept with the record so
            history survives a restart.
          </p>
        </div>
      </aside>
    </div>
  );
}

interface Props {
  selected: string | null;
  onSelect: (id: string | null) => void;
}

export function Runs({ selected, onSelect }: Props) {
  const [records, setRecords] = useState<RunRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = () => {
    api
      .runs()
      .then((result) => {
        setRecords(result);
        setError(null);
      })
      .catch((err: unknown) =>
        setError(err instanceof ApiError ? err.message : String(err)),
      );
  };

  useEffect(refresh, []);

  // Records arrive newest first, and "newest non-terminal" is the wrong pick the moment
  // a second run is submitted: the queued record would win, the stream would follow a
  // run where nothing happens, and the running bar would freeze. The run that is moving
  // outranks the run that is waiting; among equals, newest wins.
  const live = (records ?? []).filter(
    (record) => !TERMINAL.includes(record.status),
  );
  const activeId =
    (live.find((record) => record.status !== "queued") ?? live[0])?.id ?? null;

  useEffect(() => {
    // One stream, for whichever run is actually moving. Subscribing per row would open a
    // connection for every finished job in the list to watch nothing happen.
    if (!activeId) return;
    return followRun(activeId, refresh);
  }, [activeId]);

  if (error) return <span className="flag flag--bad">{error}</span>;
  if (!records) return <p className="empty">Loading runs…</p>;
  if (records.length === 0) {
    return (
      <div className="panel empty">
        No runs yet. Configure one on <a href="#/new">New run</a>.
      </div>
    );
  }

  const current = selected
    ? records.find((record) => record.id === selected)
    : undefined;

  return (
    <div className="stack">
      <div className="panel">
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th>Kind</th>
              <th>Status</th>
              <th style={{ width: "22%" }}>Progress</th>
              <th className="num">Completed</th>
              <th className="num">Elapsed</th>
              <th>Started</th>
            </tr>
          </thead>
          <tbody>
            {records.map((record) => (
              // A clickable row is a button in all but markup, so it takes a button's
              // keys too -- otherwise the detail view is mouse-only. Space is
              // preventDefaulted or it scrolls the page instead of selecting.
              <tr
                key={record.id}
                className="clickable"
                role="button"
                tabIndex={0}
                aria-pressed={record.id === selected}
                onClick={() =>
                  onSelect(record.id === selected ? null : record.id)
                }
                onKeyDown={(event) => {
                  if (event.key !== "Enter" && event.key !== " ") return;
                  event.preventDefault();
                  onSelect(record.id === selected ? null : record.id);
                }}
              >
                <td>{record.id}</td>
                <td>{record.mode}</td>
                <td>
                  <Status record={record} />
                </td>
                <td>
                  <Bar record={record} />
                </td>
                <td className="num">{completedText(record)}</td>
                <td className="num">
                  {elapsed(record.started_at, record.finished_at)}
                </td>
                <td>{when(record.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {current && <RunDetail record={current} onChanged={refresh} />}
    </div>
  );
}
