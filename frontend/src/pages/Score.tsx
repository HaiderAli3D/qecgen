import { useEffect, useMemo, useState } from "react";
import { ApiError, api } from "../api";
import { NumberField, Select } from "../components/Field";
import { EXPLAINERS } from "../explainers";
import { bytes, count, shortHash } from "../format";
import type { CorrectionEntry, DatasetEntry, ScorePreview } from "../types";

interface Props {
  onSubmitted: (id: string) => void;
}

/**
 * Grade a proposed correction by its logical effect.
 *
 * The framing matters as much as the number. This is not Contract C: nothing here infers
 * which physical fault occurred from a syndrome. The correction is an **input**, and this
 * runs the map forward — given a correction, what was its logical effect.
 */
export function Score({ onSubmitted }: Props) {
  const [datasets, setDatasets] = useState<DatasetEntry[] | null>(null);
  const [corrections, setCorrections] = useState<CorrectionEntry[] | null>(
    null,
  );
  const [dataset, setDataset] = useState("");
  const [correction, setCorrection] = useState("");
  const [alpha, setAlpha] = useState<number | "">(0.05);
  const [preview, setPreview] = useState<ScorePreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api
      .datasets()
      .then((entries) => {
        const usable = entries.filter(
          (entry) => !entry.unreadable && !entry.not_a_dataset,
        );
        setDatasets(usable);
        setDataset((current) => current || (usable[0]?.path ?? ""));
      })
      .catch(() => setDatasets([]));
    api
      .corrections()
      .then((entries) => {
        setCorrections(entries);
        setCorrection((current) => current || (entries[0]?.path ?? ""));
      })
      .catch(() => setCorrections([]));
  }, []);

  const body = useMemo(() => {
    if (!dataset || !correction || alpha === "") return null;
    return {
      mode: "score",
      dataset,
      correction,
      alpha,
      // Detected from the file's own dtype rather than asked. A user who answers this
      // wrong does not get an error: scoring unpacked bools as though they were packed
      // reads bit 0 of each byte and returns a plausible number for a correction nobody
      // proposed.
      unpacked: preview?.unpacked ?? false,
    };
  }, [dataset, correction, alpha, preview?.unpacked]);

  useEffect(() => {
    if (!dataset || !correction) {
      setPreview(null);
      setPreviewError(null);
      return;
    }
    let live = true;
    api
      .preview({ mode: "score", dataset, correction })
      .then((result) => {
        if (!live) return;
        setPreview(result as unknown as ScorePreview);
        setPreviewError(null);
      })
      .catch((error: unknown) => {
        if (!live) return;
        setPreview(null);
        setPreviewError(
          error instanceof ApiError ? error.message : String(error),
        );
      });
    return () => {
      live = false;
    };
  }, [dataset, correction]);

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

  const chosen = corrections?.find((entry) => entry.path === correction);

  return (
    <div className="run-layout">
      <div className="panel form-card">
        <fieldset style={{ borderTop: "none" }}>
          <legend>What to grade</legend>
          <div className="grid">
            <Select
              label="Dataset"
              value={dataset}
              options={(datasets ?? []).map((entry) => entry.path)}
              onChange={setDataset}
              hint="Holds the true observable flips."
            />
            <Select
              label="Correction"
              topic={EXPLAINERS.correction}
              value={correction}
              options={(corrections ?? []).map((entry) => entry.path)}
              onChange={setCorrection}
              hint="An .npz under the data root."
            />
            <NumberField
              label="Confidence"
              topic={EXPLAINERS.alpha}
              step="0.01"
              min={0}
              value={alpha}
              onChange={setAlpha}
              hint="1 − confidence level."
            />
          </div>

          {corrections !== null && corrections.length === 0 && (
            <span className="flag">
              No correction files found. Write one with{" "}
              <code>np.savez(path, correction_x=…, correction_z=…)</code> into{" "}
              <code>
                {datasets && datasets.length > 0
                  ? "the data root"
                  : "the data root"}
              </code>{" "}
              and it will appear here. There is no upload: this server has no
              authentication, so it reads what is already there rather than
              accepting arbitrary bytes.
            </span>
          )}

          {chosen && (
            <p className="note">
              {chosen.name}: {count(chosen.shots)} shots × {chosen.width}{" "}
              {chosen.unpacked ? "data qubits (bool)" : "packed bytes (uint8)"},{" "}
              {bytes(chosen.size_bytes)}.
            </p>
          )}
        </fieldset>

        <fieldset>
          <legend>Not Contract C</legend>
          <p className="prose">
            This applies the correction and measures the logical outcome. It
            does not infer which physical fault occurred from a syndrome — that
            inverts a deliberately many-to-one map and has no unique answer, and
            it stays refused. The correction is an input here, not a target, so
            no ground-truth fault label is needed and no file gains a label
            column.
          </p>
          <p className="note">
            The logical operators are rebuilt from a <strong>noiseless</strong>{" "}
            circuit generated from the dataset's own manifest parameters, so
            scoring a frozen-prior test file never reads that file's error
            model.
          </p>
        </fieldset>
      </div>

      <aside className="aside">
        <div className="panel">
          <h3>Compatibility</h3>
          {preview ? (
            <>
              <dl className="readout">
                <dt>Data qubits</dt>
                <dd className="big">{preview.n_data_qubits}</dd>
                <dt>Needs</dt>
                <dd>
                  {count(preview.shots)} ×{" "}
                  {preview.unpacked
                    ? preview.n_data_qubits
                    : preview.packed_width}
                </dd>
                <dt>Has</dt>
                <dd>
                  {count(preview.correction_shots)} × {preview.correction_width}
                </dd>
                <dt>Observables</dt>
                <dd>{preview.n_observables}</dd>
                <dt>Schema</dt>
                <dd>{shortHash(preview.schema_digest)}</dd>
              </dl>
              {preview.compatible ? (
                <span className="flag flag--calm">
                  Shapes agree. Checked against the dataset's own schema before
                  either file is read in full.
                </span>
              ) : (
                preview.problems.map((problem) => (
                  <span className="flag flag--bad" key={problem}>
                    {problem}
                  </span>
                ))
              )}
              {preview.drift_condition === "frozen_prior" && (
                <span className="flag">
                  This file was written under a frozen prior. Scoring it is safe
                  — the schema is a property of the code, not of the noise — but
                  a decoder that produced this correction must not have read the
                  file's own error model.
                </span>
              )}
            </>
          ) : (
            <p className="note">
              {previewError ??
                "Pick a dataset and a correction to check them against each other."}
            </p>
          )}
          <hr className="rule" />
          <button
            className="primary"
            onClick={submit}
            disabled={!body || busy || !(preview?.compatible ?? false)}
          >
            {busy ? "Starting…" : "Score correction"}
          </button>
          {submitError && <span className="flag flag--bad">{submitError}</span>}
        </div>
      </aside>
    </div>
  );
}
