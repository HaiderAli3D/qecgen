import { useEffect, useState } from "react";
import { ApiError, api, followRun } from "../api";
import { Lattice } from "../components/Lattice";
import { count, elapsed, shortHash, when } from "../format";
import type { RunArtifact, RunRecord, WrittenFile } from "../types";
import { TERMINAL } from "../types";

function Status({ record }: { record: RunRecord }) {
  return <span className={`status status--${record.status}`}>{record.status}</span>;
}

function fraction(record: RunRecord): number {
  if (record.total_units <= 0) return 0;
  return Math.min(1, record.completed_units / record.total_units);
}

const SWEEP_KINDS = ["sweep_results", "sweep_plot", "sweep_summary"];

/**
 * Whether an artifact should be rendered with a dataset's columns.
 *
 * Positive test for the *sweep* kinds rather than equality with `"dataset"`, so anything
 * unrecognised renders as a dataset instead of falling through. Records persisted before
 * artifacts carried a `kind` have no discriminant at all: `JobStore.load_history`
 * backfills those on read, but a file dict that reaches here without one must not take a
 * branch that then calls a string method on `undefined` — an uncaught render error
 * unmounts the whole app, and the run id lives in the hash, so reloading would blank the
 * page again.
 */
function isDatasetFile(file: RunArtifact): file is WrittenFile {
  return !SWEEP_KINDS.includes((file as { kind?: string }).kind ?? "");
}

/**
 * What a run wrote.
 *
 * Datasets and sweep artifacts share only a path, so rather than one table with half its
 * cells empty for a sweep, each shape gets the columns it actually has. The choice is made
 * once for the whole table and drives both the header and every row — deciding it
 * per-row while the header decided it some other way is how a four-column header ends up
 * over a two-cell row.
 */
function FilesTable({ files }: { files: RunArtifact[] }) {
  const asDatasets = files.every(isDatasetFile);
  return (
    <table>
      <thead>
        <tr>
          <th>Path</th>
          {asDatasets ? (
            <>
              <th className="num">Shots</th>
              <th>Condition</th>
              <th>Content hash</th>
            </>
          ) : (
            <th>Kind</th>
          )}
        </tr>
      </thead>
      <tbody>
        {files.map((file) => (
          <tr key={file.path}>
            <td className="truncate" title={file.path}>
              {file.path.split(/[\\/]/).pop()}
            </td>
            {asDatasets && isDatasetFile(file) ? (
              <>
                <td className="num">{count(file.shots)}</td>
                <td>{file.drift_condition}</td>
                <td>{shortHash(file.content_hash)}</td>
              </>
            ) : (
              <td>{((file as { kind?: string }).kind ?? "dataset").replace("sweep_", "")}</td>
            )}
          </tr>
        ))}
      </tbody>
    </table>
  );
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
            {/* A sweep counts sinter tasks, not shots -- max_errors stops it and max_shots
                is only a ceiling. The record names its own unit so this never labels one
                thing while counting another. */}
            <dt>{record.progress_unit === "tasks" ? "Tasks" : "Sampled"}</dt>
            <dd>
              {count(record.completed_units)} / {count(record.total_units)}
            </dd>
            {record.shots_collected !== null && (
              <>
                <dt>Shots collected</dt>
                <dd>{count(record.shots_collected)}</dd>
              </>
            )}
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
            {/* Datasets and sweep artifacts share only a path. Rather than one table with
                half its cells empty for a sweep, each kind gets the columns it has.
                Header and body branch on the SAME predicate: two discriminants for one
                decision is how a 4-column header ends up over a 2-cell row. */}
            <FilesTable files={record.files} />
            {record.mode === "sweep" && (
              <p className="note" style={{ marginTop: "0.6rem" }}>
                Open this on the <a href="#/sweeps">Sweeps</a> tab to see the curves.
              </p>
            )}
          </div>
        )}
      </div>

      <aside className="aside">
        {/* No lattice for a sweep. It has no single `distance` -- it sweeps several -- so
            the figure would silently fall back to d=3 and caption it "sampled", which is
            neither the distance being run nor the unit being counted. */}
        {record.mode !== "sweep" && (
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
        )}
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
              {/* Not "Shots". This list mixes dataset runs with sweeps, and a sweep's
                  `completed_units` is a count of sinter tasks -- a task count under a
                  "Shots" header is precisely the well-formed reading of the wrong quantity
                  the unit field was introduced to prevent. One static header cannot serve
                  both, so the unit travels in the cell. */}
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
                <td className="num">{count(record.completed_units)}</td>
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
