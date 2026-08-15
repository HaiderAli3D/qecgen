import { useEffect, useRef, useState } from "react";
import { ApiError, api } from "../api";
import { bytes, count, shortHash, when } from "../format";
import type { Check, DatasetEntry } from "../types";

function Detail({ entry }: { entry: DatasetEntry }) {
  const [manifest, setManifest] = useState<Record<string, unknown> | null>(null);
  const [checks, setChecks] = useState<Check[] | null>(null);
  const [ok, setOk] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // The component survives a row switch -- only the prop changes -- so an in-flight
  // response for the previous row would land on the new one: a validate verdict shown
  // under the wrong file. Every response is keyed to the path it was requested for and
  // dropped if the row has moved on, whichever order responses arrive in.
  const shownPath = useRef(entry.path);

  useEffect(() => {
    shownPath.current = entry.path;
    setManifest(null);
    setChecks(null);
    setOk(null);
    setError(null);
    setBusy(false);
    api
      .manifest(entry.path)
      .then((result) => {
        if (shownPath.current === entry.path) setManifest(result);
      })
      .catch((err: unknown) => {
        if (shownPath.current !== entry.path) return;
        setError(err instanceof ApiError ? err.message : String(err));
      });
  }, [entry.path]);

  async function validate() {
    const path = entry.path;
    setBusy(true);
    setError(null);
    try {
      const report = await api.validate(path);
      if (shownPath.current !== path) return;
      setChecks(report.checks);
      setOk(report.ok);
    } catch (err) {
      if (shownPath.current !== path) return;
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      if (shownPath.current === path) setBusy(false);
    }
  }

  return (
    <div className="panel" style={{ padding: "1.25rem", marginTop: "0.75rem" }}>
      <div className="section-head">
        <h2 className="truncate">{entry.path}</h2>
        <div className="row">
          {/* One element, one tab stop: a button nested in an anchor is invalid
              interactive content, and the download is navigation, so the anchor is the
              control and merely dresses as a button. */}
          <a className="button" href={api.downloadUrl(entry.path)} download>
            Download
          </a>
          <button type="button" onClick={validate} disabled={busy}>
            {busy ? "Checking…" : "Validate"}
          </button>
        </div>
      </div>

      {error && <span className="flag flag--bad">{error}</span>}

      {ok !== null && (
        <span className={ok ? "flag flag--calm" : "flag flag--bad"}>
          {ok
            ? "Every structural check passed. The shot count and content hash match the arrays."
            : "This file failed structural validation. Details below."}
        </span>
      )}

      {checks && (
        <ul className="checks" style={{ marginTop: "0.75rem" }}>
          {checks.map((check) => (
            <li key={check.name}>
              <span className={`verdict ${check.passed ? "pass" : "fail"}`}>
                {check.passed ? "PASS" : "FAIL"}
              </span>
              <span>{check.name}</span>
              <span className="detail">
                {check.detail}
                {!check.passed && check.requirement ? ` — expected ${check.requirement}` : ""}
              </span>
            </li>
          ))}
        </ul>
      )}

      <h3 style={{ marginTop: "1.1rem" }}>Manifest</h3>
      <pre className="mono-block">
        {manifest ? JSON.stringify(manifest, null, 2) : "Loading…"}
      </pre>
      <p className="note" style={{ marginTop: "0.6rem" }}>
        The manifest is what a decoder sees. Circuit and DEM text live in the file's
        provenance block and are deliberately not shown here — under a frozen prior, that
        text is exactly what the condition withholds.
      </p>
    </div>
  );
}

export function Datasets() {
  const [entries, setEntries] = useState<DatasetEntry[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .datasets()
      .then((result) => {
        setEntries(result);
        setError(null);
      })
      .catch((err: unknown) => setError(err instanceof ApiError ? err.message : String(err)));
  }, []);

  if (error) return <span className="flag flag--bad">{error}</span>;
  if (!entries) return <p className="empty">Reading manifests…</p>;
  if (entries.length === 0) {
    return (
      <div className="panel empty">
        Nothing here yet. Start a run on <a href="#/new">New run</a> and it will appear.
      </div>
    );
  }

  const current = entries.find((entry) => entry.path === selected);

  return (
    <div>
      <div className="panel">
        <table>
          <thead>
            <tr>
              <th>File</th>
              <th>Format</th>
              <th className="num">d</th>
              <th className="num">Shots</th>
              <th className="num">Detectors</th>
              <th>Condition</th>
              <th>Content hash</th>
              <th className="num">Size</th>
              <th>Written</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((entry) => (
              // Same shape as the Runs rows: a clickable row is a button in all but
              // markup, so it takes a button's keys, and Space is preventDefaulted or
              // it scrolls the page instead of selecting.
              <tr
                key={entry.path}
                className="clickable"
                role="button"
                tabIndex={0}
                aria-pressed={entry.path === selected}
                onClick={() => setSelected(entry.path === selected ? null : entry.path)}
                onKeyDown={(event) => {
                  if (event.key !== "Enter" && event.key !== " ") return;
                  event.preventDefault();
                  setSelected(entry.path === selected ? null : entry.path);
                }}
              >
                <td className="truncate" title={entry.path}>
                  {entry.path}
                </td>
                <td>
                  <span className="tag">{entry.format}</span>
                </td>
                {entry.unreadable ? (
                  <td colSpan={5}>
                    <span className="flag flag--bad" style={{ marginTop: 0 }}>
                      Unreadable: {entry.unreadable}
                    </span>
                  </td>
                ) : (
                  <>
                    <td className="num">{entry.manifest?.distance ?? "—"}</td>
                    <td className="num">{count(entry.manifest?.shots ?? 0)}</td>
                    <td className="num">{count(entry.manifest?.n_detectors ?? 0)}</td>
                    <td>{entry.manifest?.drift_condition ?? "—"}</td>
                    <td>{shortHash(entry.manifest?.content_hash)}</td>
                  </>
                )}
                <td className="num">{bytes(entry.size_bytes)}</td>
                <td>{when(entry.modified_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="note" style={{ margin: "0.6rem 0 0" }}>
        Shot counts come from each file's manifest, which is a claim rather than a
        measurement. Validate reads the arrays and checks it.
      </p>
      {current && !current.unreadable && <Detail entry={current} />}
    </div>
  );
}
