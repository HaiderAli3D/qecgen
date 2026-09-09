import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api } from "../api";
import { ConfirmDelete } from "../components/ConfirmDelete";
import { Info } from "../components/Info";
import { EXPLAINERS } from "../explainers";
import { bytes, count, shortHash, when } from "../format";
import type { Check, DatasetEntry, DeletePreview, Provenance } from "../types";

/**
 * The delete control, shared by the full detail panel and the reduced one.
 *
 * The preview is fetched BEFORE the dialog opens, so the dialog's content is always the
 * plan and there is never a spinner inside a modal. It is also the strongest stale-response
 * guard available: the preview is keyed to the path it was requested for, and the pending
 * state carries that path so the confirm deletes what the user was shown rather than
 * whatever row happens to be selected when they press the button.
 */
function DeleteControl({
  path,
  onDeleted,
}: {
  path: string;
  onDeleted: () => void;
}) {
  const [pending, setPending] = useState<{ path: string; preview: DeletePreview } | null>(
    null,
  );
  const [preparing, setPreparing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const shownPath = useRef(path);
  shownPath.current = path;

  async function ask() {
    const target = path;
    setPreparing(true);
    setError(null);
    try {
      const preview = await api.datasetDeletePreview(target);
      if (shownPath.current !== target) return;
      setPending({ path: target, preview });
    } catch (err: unknown) {
      if (shownPath.current !== target) return;
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      if (shownPath.current === target) setPreparing(false);
    }
  }

  const orphaned = pending?.preview.orphaned_runs ?? [];

  return (
    <>
      <button type="button" className="danger" onClick={ask} disabled={preparing}>
        {preparing ? "Checking…" : "Delete"}
      </button>
      <Info topic={EXPLAINERS.delete} />
      {error && <span className="flag flag--bad">{error}</span>}
      {pending && (
        <ConfirmDelete
          title="Delete this dataset"
          lead={pending.preview.reason}
          files={pending.preview.files}
          filesCaption="These files go:"
          totalBytes={pending.preview.total_bytes}
          caveat={pending.preview.caveat}
          confirmLabel="Delete"
          extra={
            orphaned.length > 0 ? (
              <p className="note">
                {count(orphaned.length)} run record(s) wrote only these files and will be
                forgotten too: {orphaned.map((run) => run.id).join(", ")}.
              </p>
            ) : null
          }
          onConfirm={() => api.deleteDataset(pending.path, true)}
          onClose={(changed) => {
            setPending(null);
            if (changed) onDeleted();
          }}
        />
      )}
    </>
  );
}

function Detail({
  entry,
  onSubmitted,
  onDeleted,
}: {
  entry: DatasetEntry;
  onSubmitted: (id: string) => void;
  onDeleted: () => void;
}) {
  const [manifest, setManifest] = useState<string | null>(
    null,
  );
  const [checks, setChecks] = useState<Check[] | null>(null);
  const [ok, setOk] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [provenance, setProvenance] = useState<Provenance | null>(null);
  const [revealing, setRevealing] = useState(false);
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
    // Cleared on every row change, and never fetched here. Provenance is loaded only by
    // the button below: under a frozen prior this text is exactly what the condition
    // withholds, and a page that has it loaded because the row was selected has already
    // undone the separation the server built.
    setProvenance(null);
    api
      .manifestText(entry.path)
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

  async function runQa() {
    setBusy(true);
    setError(null);
    try {
      const record = await api.submit({ mode: "qa", dataset: entry.path });
      onSubmitted(record.id);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function runBenchmark() {
    setBusy(true);
    setError(null);
    try {
      const record = await api.submit({
        mode: "benchmark",
        dataset: entry.path,
      });
      onSubmitted(record.id);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function reveal() {
    const path = entry.path;
    setRevealing(true);
    setError(null);
    try {
      const payload = await api.provenance(path);
      if (shownPath.current === path) setProvenance(payload);
    } catch (err) {
      if (shownPath.current !== path) return;
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      if (shownPath.current === path) setRevealing(false);
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
          <button type="button" onClick={runQa} disabled={busy}>
            Statistical QA
          </button>
          <button type="button" onClick={runBenchmark} disabled={busy}>
            Decoder baseline
          </button>
          {/* Last, and spaced away from the three that add: the only control in this row
              that takes something away should not sit flush against them. */}
          <span style={{ marginLeft: "0.6rem" }} />
          <DeleteControl path={entry.path} onDeleted={onDeleted} />
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
                {!check.passed && check.requirement
                  ? ` — expected ${check.requirement}`
                  : ""}
              </span>
            </li>
          ))}
        </ul>
      )}

      <h3 style={{ marginTop: "1.1rem" }}>Manifest</h3>
      <pre className="mono-block">
        {manifest ?? "Loading…"}
      </pre>
      <p className="note" style={{ marginTop: "0.6rem" }}>
        The manifest is what a decoder sees. Circuit and DEM text are not part
        of it: they live in the file's provenance block, stored physically
        apart, and are fetched only when you ask for them below.
      </p>

      <h3 style={{ marginTop: "1.1rem" }}>Circuit and DEM text</h3>
      {provenance === null ? (
        <div className="row">
          <button type="button" onClick={reveal} disabled={revealing}>
            {revealing ? "Reading…" : "Show circuit & DEM text"}
          </button>
          <span className="note">
            Written only at structure level <code>full</code>, and only by
            formats that carry a provenance block.
          </span>
        </div>
      ) : !provenance.stored ? (
        <p className="note">
          No provenance stored (structure level{" "}
          <code>{provenance.structure_level}</code>). Circuit and DEM text are
          written only at <code>full</code>, and only by{" "}
          {provenance.formats_that_store_it.join(", ")}.
        </p>
      ) : (
        <>
          {/* The warning names this file's own condition rather than describing the
              hazard in the abstract. Under a frozen prior the text below is the test
              environment's own error model -- precisely what the experiment withholds
              from the decoder. It is shown rather than refused because the CLI shows it
              and the bytes are in the file either way; what is protected here is the
              reader's discipline, not the file's secrecy. */}
          {provenance.drift_condition === "frozen_prior" ? (
            <span className="flag flag--bad">
              This file was written under a <strong>frozen prior</strong>. The
              text below describes its own error model, which is exactly what
              the condition withholds from a decoder. Read it to audit what was
              generated; do not feed it to anything being evaluated on this
              file.
            </span>
          ) : (
            <span className="flag">
              Provenance is not decoder-visible. It is retained so a reviewer
              can audit what was generated, and kept out of the manifest so that
              reading the manifest cannot expose it.
            </span>
          )}
          {provenance.environments.map((environment) => (
            <div key={environment.environment_id}>
              <h3 style={{ marginTop: "0.9rem" }}>
                Environment {environment.environment_id} · circuit
              </h3>
              <pre className="mono-block">{environment.circuit}</pre>
              <h3>
                Environment {environment.environment_id} · detector error model
              </h3>
              <pre className="mono-block">{environment.dem}</pre>
            </div>
          ))}
        </>
      )}
    </div>
  );
}

export function Datasets({
  onSubmitted,
}: {
  onSubmitted: (id: string) => void;
}) {
  const [entries, setEntries] = useState<DatasetEntry[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Focus lands here after a delete. The panel and its buttons unmount with the selection,
  // so without somewhere to send it focus falls to <body> and a keyboard user loses their
  // place in the table entirely.
  const listRef = useRef<HTMLTableElement>(null);

  const refresh = useCallback(() => {
    api
      .datasets()
      .then((result) => {
        setEntries(result);
        setError(null);
      })
      // Report the failure; do NOT clear `entries`. Setting it to [] renders "Nothing here
      // yet", so a server restart or a dropped connection would claim the data root is
      // empty -- and because this now also runs after a delete, a blip at that moment would
      // say the root is empty immediately after one file was removed from it.
      .catch((err: unknown) =>
        setError(err instanceof ApiError ? err.message : String(err)),
      );
  }, []);

  useEffect(refresh, [refresh]);

  function afterDelete() {
    setSelected(null);
    refresh();
    listRef.current?.focus();
  }

  // Only blank the page when there is nothing to show. Replacing a good table with an
  // error was harmless when the listing was fetched once at mount; with a refresh it
  // destroys a perfectly good page on a transient failure.
  if (error && !entries) return <span className="flag flag--bad">{error}</span>;
  if (!entries) return <p className="empty">Reading manifests…</p>;
  if (entries.length === 0) {
    return (
      <div className="panel empty">
        Nothing here yet. Start a run on <a href="#/new">New run</a> and it will
        appear.
      </div>
    );
  }

  const current = entries.find((entry) => entry.path === selected);

  return (
    <div>
      {error && <span className="flag flag--bad">{error}</span>}
      <div className="panel">
        <table ref={listRef} tabIndex={-1}>
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
                  {entry.path}
                </td>
                <td>
                  <span className="tag">{entry.format}</span>
                </td>
                {entry.unreadable || entry.not_a_dataset ? (
                  <td colSpan={5}>
                    {/* Two different things, and the difference matters: a corruption
                        flag that also fires on every intact sweep results table is one
                        the reader learns to skip. */}
                    <span
                      className={
                        entry.unreadable ? "flag flag--bad" : "flag flag--calm"
                      }
                      style={{ marginTop: 0 }}
                    >
                      {entry.unreadable
                        ? `Unreadable: ${entry.unreadable}`
                        : entry.not_a_dataset}
                    </span>
                  </td>
                ) : (
                  <>
                    <td className="num">{entry.manifest?.distance ?? "—"}</td>
                    <td className="num">{count(entry.manifest?.shots ?? 0)}</td>
                    <td className="num">
                      {count(entry.manifest?.n_detectors ?? 0)}
                    </td>
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
        Shot counts come from each file's manifest, which is a claim rather than
        a measurement. Validate reads the arrays and checks it.
      </p>
      {current && !current.unreadable && !current.not_a_dataset && (
        <Detail entry={current} onSubmitted={onSubmitted} onDeleted={afterDelete} />
      )}
      {current && (current.unreadable || current.not_a_dataset) && (
        // A reduced panel, because these rows are the ones a delete is most often aimed
        // at: a half-written file a dead worker left, or a sweep results table. There is no
        // manifest to show and nothing to validate, but there is very much something to
        // remove -- and without this the only files you cannot delete are the broken ones.
        <div className="panel" style={{ padding: "1.25rem", marginTop: "0.75rem" }}>
          <div className="section-head">
            <h2 className="truncate">{current.path}</h2>
            <div className="row">
              <a className="button" href={api.downloadUrl(current.path)} download>
                Download
              </a>
              <DeleteControl path={current.path} onDeleted={afterDelete} />
            </div>
          </div>
          <span className={current.unreadable ? "flag flag--bad" : "flag flag--calm"}>
            {current.unreadable ?? current.not_a_dataset}
          </span>
        </div>
      )}
    </div>
  );
}
