import { useEffect, useState } from "react";
import { ApiError, api, followRun } from "../api";
import { Lattice } from "../components/Lattice";
import { count, elapsed, shortHash, when } from "../format";
import type { RunRecord } from "../types";
import { TERMINAL } from "../types";

function Status({ record }: { record: RunRecord }) {
  return <span className={`status status--${record.status}`}>{record.status}</span>;
}

function fraction(record: RunRecord): number {
  if (record.total_shots <= 0) return 0;
  return Math.min(1, record.completed_shots / record.total_shots);
}

function Bar({ record }: { record: RunRecord }) {
  // The bar goes indeterminate during the writing phase because nothing reports progress
  // through concatenation, hashing and gzip -- and a full bar sitting still reads as a
  // hang. Saying "writing" is more honest than implying the run is done.
  const writing = record.status === "running" && record.phase === "writing";
  return (
    <span className={`bar${writing ? " bar--indeterminate" : ""}`}>
      <span style={writing ? undefined : { width: `${fraction(record) * 100}%` }} />
    </span>
  );
}

function RunDetail({ record, onChanged }: { record: RunRecord; onChanged: () => void }) {
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
            <dt>Sampled</dt>
            <dd>
              {count(record.completed_shots)} / {count(record.total_shots)}
            </dd>
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
                Stops at the next chunk boundary. Nothing is left at the output path.
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
      </div>

      <aside className="aside">
        <div className="panel">
          <Lattice
            distance={Number(spec.distance ?? 3)}
            basis={String(spec.basis ?? "z")}
            progress={fraction(record)}
            label={`${Math.round(fraction(record) * 100)}% sampled`}
          />
          {spec.rotated === false && (
            <p className="note">
              This run uses the unrotated layout; the figure shows the rotated one.
            </p>
          )}
        </div>
        <div className="panel">
          <h3>Resolved configuration</h3>
          <pre className="mono-block">{JSON.stringify(spec, null, 2)}</pre>
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
      .catch((err: unknown) => setError(err instanceof ApiError ? err.message : String(err)));
  };

  useEffect(refresh, []);

  // Records arrive newest first, and "newest non-terminal" is the wrong pick the moment
  // a second run is submitted: the queued record would win, the stream would follow a
  // run where nothing happens, and the running bar would freeze. The run that is moving
  // outranks the run that is waiting; among equals, newest wins.
  const live = (records ?? []).filter((record) => !TERMINAL.includes(record.status));
  const activeId = (live.find((record) => record.status !== "queued") ?? live[0])?.id ?? null;

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

  const current = selected ? records.find((record) => record.id === selected) : undefined;

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
              <th className="num">Shots</th>
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
                onClick={() => onSelect(record.id === selected ? null : record.id)}
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
                <td className="num">{count(record.completed_shots)}</td>
                <td className="num">{elapsed(record.started_at, record.finished_at)}</td>
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
