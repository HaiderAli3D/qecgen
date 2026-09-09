import { useEffect, useId, useRef, useState } from "react";
import type { ReactNode } from "react";

import { ApiError } from "../api";
import { bytes, count } from "../format";
import type { DeletableFile, DeleteOutcome, Outcome } from "../types";

/**
 * The confirmation for every destructive action in this app.
 *
 * A native `<dialog>` opened with `showModal()`, not a div with a z-index. Focus
 * containment, Escape (the `cancel` event), the top layer and `::backdrop` all come with
 * the element; hand-rolling them would make this the app's first focus trap, in a tree with
 * no test runner. It is deliberately **not** portalled: a top-layer element already escapes
 * every ancestor's `overflow` and stacking context, which is the one thing `Info`'s
 * `createPortal` exists to work around.
 *
 * The dialog does not close on success. It swaps the file list for what actually happened to
 * each file, because that is the only place the recycle-bin caveat can be answered
 * concretely -- and because both panels that open it unmount when the selection clears, so a
 * page-level result banner would have nowhere to live.
 */

const OUTCOME_LABEL: Record<Outcome, string> = {
  // Never "in the Recycle Bin". Windows deletes an oversize file outright and reports
  // success either way, so "gone" is the whole of what the server observed.
  removed: "removed",
  already_missing: "already gone",
  vanished: "gone before we reached it",
  skipped: "not attempted",
  locked: "in use by another program",
  no_trash: "this volume has no recycle bin",
  failed: "could not be removed",
};

const FAILED: ReadonlySet<Outcome> = new Set<Outcome>([
  "skipped",
  "locked",
  "no_trash",
  "failed",
]);

interface ConfirmDeleteProps {
  /** Dialog heading. */
  title: string;
  /** One sentence above the list, describing what goes that is not a file. */
  lead: ReactNode;
  files: DeletableFile[];
  /** Caption over the table. Changes with the Runs checkbox. */
  filesCaption: string;
  /** True when the listed files will be LEFT ALONE. Dims the table. */
  filesKept?: boolean;
  totalBytes: number;
  caveat: string;
  /** Rendered between the list and the buttons — the Runs outputs checkbox goes here. */
  extra?: ReactNode;
  confirmLabel: string;
  onConfirm: () => Promise<DeleteOutcome>;
  /** `changed` is true only when something was actually removed. */
  onClose: (changed: boolean) => void;
}

export function ConfirmDelete({
  title,
  lead,
  files,
  filesCaption,
  filesKept = false,
  totalBytes,
  caveat,
  extra,
  confirmLabel,
  onConfirm,
  onClose,
}: ConfirmDeleteProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const cancelRef = useRef<HTMLButtonElement>(null);
  const downRef = useRef<EventTarget | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<DeleteOutcome | null>(null);
  const titleId = useId();
  const leadId = useId();

  // Mirrored so the `cancel` listener below can read it without closing over stale state.
  const busyRef = useRef(false);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    // Guarded: showModal() throws InvalidStateError on an already-open dialog, and
    // StrictMode mounts this effect, tears it down and mounts it again.
    if (!dialog.open) dialog.showModal();
    cancelRef.current?.focus();

    // `cancel`, never `close`. `close` also fires for a programmatic `close()`, so under
    // StrictMode's mount/unmount/mount a `close` listener reports a cancellation the user
    // never made -- in development only, which is the mode the frontend is worked on in.
    // Same family as the Info singleton bug. `cancel` fires only for Escape.
    const onCancel = (event: Event) => {
      if (busyRef.current) {
        // A delete is in flight. Dismissing now would throw away the only report of what
        // happened to each file.
        event.preventDefault();
      }
    };
    dialog.addEventListener("cancel", onCancel);
    return () => {
      dialog.removeEventListener("cancel", onCancel);
      // Close while the node is still connected, so the browser restores focus to the
      // trigger. Relying on React to unmount it instead loses the restore.
      if (dialog.open) dialog.close();
    };
  }, []);

  function finish(changed: boolean) {
    if (busyRef.current) return;
    onClose(changed);
  }

  async function run() {
    setBusy(true);
    busyRef.current = true;
    setError(null);
    try {
      setOutcome(await onConfirm());
    } catch (err: unknown) {
      // Surfaced, never swallowed. A failed delete leaves the row exactly where it was,
      // which is indistinguishable from a refresh that has not landed yet -- and the user's
      // next move would be to press an irreversible button again.
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setBusy(false);
      busyRef.current = false;
    }
  }

  const failures = outcome?.files.filter((file) => FAILED.has(file.outcome)) ?? [];

  return (
    <dialog
      ref={dialogRef}
      className="confirm"
      aria-labelledby={titleId}
      aria-describedby={leadId}
      onMouseDown={(event) => {
        downRef.current = event.target;
      }}
      onClick={(event) => {
        // `.confirm` has no padding and all content sits in `.confirm__body`, so an event
        // whose target is the dialog element itself can only have come from the backdrop.
        // Both press and release are checked, or selecting text and releasing outside the
        // box would read as a dismissal.
        if (event.target === dialogRef.current && downRef.current === dialogRef.current) {
          finish(outcome !== null);
        }
      }}
    >
      <div className="confirm__body">
        <h2 id={titleId}>{title}</h2>

        {outcome === null ? (
          <>
            <p id={leadId} className="note">
              {lead}
            </p>
            <span className="label">{filesCaption}</span>
            <div className={`confirm__files${filesKept ? " confirm__files--kept" : ""}`}>
              <table>
                <thead>
                  <tr>
                    <th>File</th>
                    <th>What it is</th>
                    <th>Size</th>
                  </tr>
                </thead>
                <tbody>
                  {files.map((file) => (
                    <tr key={file.path}>
                      {/* The root-relative path, not the basename: a confirmation that says
                          "dataset.h5" when two directories hold one confirms the wrong file. */}
                      <td className="truncate" title={file.path}>
                        {file.path}
                      </td>
                      <td>
                        <span className="tag">{file.role}</span>
                      </td>
                      <td>{file.exists ? bytes(file.size_bytes) : "already gone"}</td>
                    </tr>
                  ))}
                  {files.length === 0 ? (
                    <tr>
                      <td colSpan={3} className="empty">
                        No files — nothing on disk to remove.
                      </td>
                    </tr>
                  ) : null}
                </tbody>
              </table>
            </div>
            <p className="note">
              {count(files.filter((file) => file.exists).length)} file(s) · {bytes(totalBytes)}
            </p>
            <span className="flag">{caveat}</span>
            {extra}
            {error ? (
              <span className="flag flag--bad" role="alert">
                {error}
              </span>
            ) : null}
            <div className="row confirm__actions">
              <button ref={cancelRef} type="button" onClick={() => finish(false)} disabled={busy}>
                Cancel
              </button>
              <button type="button" className="danger" onClick={run} disabled={busy}>
                {busy ? "Deleting…" : confirmLabel}
              </button>
            </div>
          </>
        ) : (
          <>
            <p id={leadId} className="note">
              {/* Derived, never a single sentence for every ending. `complete` is
                  vacuously true when nothing was attempted, so "Everything named above is
                  gone" would appear over an empty table for a run deleted with its files
                  deliberately KEPT -- a well-formed statement of the opposite of what
                  happened, which is the whole failure this feature guards against. */}
              {outcome.files.length === 0
                ? outcome.deleted_files === false
                  ? "The run record is gone. The files it wrote were kept, as you asked."
                  : "The run record is gone. It had written no files."
                : outcome.complete
                  ? "Everything named above is gone."
                  : "Some files are still there — see below."}
            </p>
            <div className="confirm__files">
              <table>
                <thead>
                  <tr>
                    <th>File</th>
                    <th>What happened</th>
                    <th>Size</th>
                  </tr>
                </thead>
                <tbody>
                  {outcome.files.map((file) => (
                    <tr key={file.path}>
                      <td className="truncate" title={file.path}>
                        {file.path}
                      </td>
                      <td>
                        {FAILED.has(file.outcome) ? (
                          <span className="flag flag--bad">{OUTCOME_LABEL[file.outcome]}</span>
                        ) : (
                          OUTCOME_LABEL[file.outcome]
                        )}
                        {file.error ? <span className="note"> {file.error}</span> : null}
                      </td>
                      <td>{file.exists ? bytes(file.size_bytes) : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {outcome.forgotten_runs && outcome.forgotten_runs.length > 0 ? (
              <p className="note">
                Also forgot {count(outcome.forgotten_runs.length)} run record(s) that this left
                describing nothing.
              </p>
            ) : null}
            {outcome.record_removed === false ? (
              <span className="flag flag--bad" role="alert">
                The run record itself could not be removed
                {outcome.record_problem ? `: ${outcome.record_problem}` : ""}. It will come back
                the next time the server restarts.
              </span>
            ) : null}
            {failures.length === 0 ? <span className="flag">{outcome.caveat}</span> : null}
            <div className="row confirm__actions">
              <button ref={cancelRef} type="button" onClick={() => finish(true)}>
                Close
              </button>
            </div>
          </>
        )}
      </div>
    </dialog>
  );
}
