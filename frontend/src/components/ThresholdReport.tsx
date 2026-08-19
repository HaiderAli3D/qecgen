/**
 * The numbers behind a threshold sweep, rendered from the sidecar it wrote.
 *
 * A component rather than page-local because two places show the same report: the Sweeps
 * browser, for a sweep on disk, and a finished sweep's own run record. It computes
 * nothing, exactly as `ThresholdChart` computes nothing — every figure here was written by
 * `sweep.threshold_summary`, and a second derivation in TypeScript is the silent drift the
 * generated-figure rule exists to prevent.
 *
 * Lambda is only meaningful below threshold, so each row says which side of the crossing
 * it sits on rather than letting a super-threshold number read as a suppression result.
 */
import type { ThresholdSummary } from "../types";

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
