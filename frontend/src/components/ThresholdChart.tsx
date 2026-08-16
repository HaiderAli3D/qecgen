import { useId, useMemo, useState } from "react";
import type { SweepPoint, SweepSeries } from "../types";

/**
 * The threshold curves, drawn from the results table.
 *
 * **This component computes no statistics.** Every number it draws — `rate`, `ci_low`,
 * `ci_high` — is read from a column `qecgen.sweep.write_csv` already wrote, and the server
 * parses that table for us. The chart is a *view* of the artifact; `sweep.png` is the
 * artifact of record. If the two ever disagree, the PNG and the CSV are right and this is
 * broken. Deriving an interval here would be a second implementation of the Clopper-Pearson
 * arithmetic, which is exactly the silent drift the generated-figure rule in CLAUDE.md
 * exists to prevent.
 *
 * One piece of *drawing* semantics is deliberately mirrored from
 * `qecgen.sweep.plot_threshold`: a zero-error point has no place on a log axis at its point
 * estimate of 0, but it still carries an informative one-sided upper bound — and it is
 * usually the most suppressed point of a quick sweep. Dropping it silently would hide
 * exactly the data a reader most wants, so it is drawn as a downward caret at `ci_high`.
 * That rule lives in two places; the matplotlib docstring says so too.
 */

const PALETTE = ["var(--z)", "var(--x)", "var(--warn)", "var(--bad)", "var(--good)"];

const WIDTH = 620;
const HEIGHT = 400;
const PAD = { top: 16, right: 16, bottom: 44, left: 60 };

const PLOT_W = WIDTH - PAD.left - PAD.right;
const PLOT_H = HEIGHT - PAD.top - PAD.bottom;

function label(series: SweepSeries): string {
  return `${series.decoder}, d=${series.distance}`;
}

/** Every y value a series contributes, so the log axis covers the carets too. */
function yValues(point: SweepPoint): number[] {
  return point.errors > 0 ? [point.rate, point.ci_low, point.ci_high] : [point.ci_high];
}

/** Whole decades spanning [low, high], as exponents so the bounds need no indexing. */
function decadeExponents(low: number, high: number): { first: number; last: number } {
  const first = Math.floor(Math.log10(low));
  const last = Math.ceil(Math.log10(high));
  // A single decade would give a zero-height axis; widen it so the curve has somewhere
  // to sit.
  return first === last ? { first: first - 1, last } : { first, last };
}

function formatRate(value: number): string {
  if (value === 0) return "0";
  const exponent = Math.log10(value);
  if (Number.isInteger(exponent)) return `1e${exponent}`;
  return value.toPrecision(2);
}

interface Props {
  series: SweepSeries[];
}

export function ThresholdChart({ series }: Props) {
  const titleId = useId();
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [hover, setHover] = useState<{ point: SweepPoint; x: number; y: number } | null>(null);

  const shown = series.filter((entry) => !hidden.has(label(entry)));

  // Keyed on the series' position in the FULL list, never in the filtered one. Colouring
  // from the filtered index meant that hiding any series but the last shifted every curve
  // below it onto the next palette entry while its legend key kept the old colour — so
  // d=3 and d=5 swapped identities on screen, and a reader attributes a suppression curve
  // to the wrong distance. Hiding a series is the only interaction this chart has.
  const colours = useMemo(
    () => new Map(series.map((entry, index) => [label(entry), PALETTE[index % PALETTE.length]])),
    [series],
  );

  const scales = useMemo(() => {
    const points = shown.flatMap((entry) => entry.points);
    const xs = points.map((point) => point.p);
    const ys = points.flatMap(yValues).filter((value) => value > 0);
    if (xs.length === 0 || ys.length === 0) return null;

    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const { first, last } = decadeExponents(Math.min(...ys), Math.max(...ys));
    const decades: number[] = [];
    for (let exponent = first; exponent <= last; exponent += 1) decades.push(10 ** exponent);
    const yMin = 10 ** first;

    // A single sampled rate would give a zero-width domain and divide by zero.
    const xSpan = xMax - xMin || 1;
    const logSpan = last - first || 1;
    return {
      x: (p: number) => PAD.left + ((p - xMin) / xSpan) * PLOT_W,
      y: (rate: number) =>
        PAD.top + PLOT_H - ((Math.log10(Math.max(rate, yMin)) - Math.log10(yMin)) / logSpan) * PLOT_H,
      decades,
      xMin,
      xMax,
    };
  }, [shown]);

  const anyZeroError = series.some((entry) => entry.points.some((point) => point.errors === 0));

  function toggle(name: string) {
    setHidden((current) => {
      const next = new Set(current);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  return (
    <figure className="chart">
      {scales === null ? (
        <p className="empty">Nothing to plot: every series is hidden, or no point was measured.</p>
      ) : (
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          role="img"
          aria-labelledby={titleId}
          onMouseLeave={() => setHover(null)}
        >
          <title id={titleId}>
            Logical error rate against physical error rate, one curve per decoder and
            distance, with Clopper-Pearson intervals.
          </title>

          {scales.decades.map((tick) => (
            <g key={tick}>
              <line
                className="chart-grid"
                x1={PAD.left}
                x2={PAD.left + PLOT_W}
                y1={scales.y(tick)}
                y2={scales.y(tick)}
              />
              <text className="chart-tick" x={PAD.left - 8} y={scales.y(tick)} textAnchor="end">
                {formatRate(tick)}
              </text>
            </g>
          ))}

          {[scales.xMin, (scales.xMin + scales.xMax) / 2, scales.xMax].map((tick: number, index) => (
            <text
              key={index}
              className="chart-tick"
              x={scales.x(tick)}
              y={PAD.top + PLOT_H + 20}
              textAnchor="middle"
            >
              {tick.toPrecision(2)}
            </text>
          ))}

          <text
            className="chart-axis"
            x={PAD.left + PLOT_W / 2}
            y={HEIGHT - 6}
            textAnchor="middle"
          >
            physical error rate p
          </text>
          <text
            className="chart-axis"
            transform={`translate(14 ${PAD.top + PLOT_H / 2}) rotate(-90)`}
            textAnchor="middle"
          >
            logical error rate per shot
          </text>

          {shown.map((entry) => {
            const colour = colours.get(label(entry));
            const measured = entry.points.filter((point) => point.errors > 0);
            const zeroError = entry.points.filter((point) => point.errors === 0);
            const path = measured
              .map(
                (point, position) =>
                  `${position === 0 ? "M" : "L"} ${scales.x(point.p)} ${scales.y(point.rate)}`,
              )
              .join(" ");
            return (
              <g key={label(entry)}>
                {measured.length > 1 && <path className="chart-line" d={path} stroke={colour} />}
                {measured.map((point) => (
                  <g key={`e${point.p}`}>
                    <line
                      className="chart-bar"
                      stroke={colour}
                      x1={scales.x(point.p)}
                      x2={scales.x(point.p)}
                      y1={scales.y(point.ci_low)}
                      y2={scales.y(point.ci_high)}
                    />
                    <circle
                      className="chart-dot"
                      fill={colour}
                      cx={scales.x(point.p)}
                      cy={scales.y(point.rate)}
                      r={4}
                      onMouseEnter={() =>
                        setHover({ point, x: scales.x(point.p), y: scales.y(point.rate) })
                      }
                    />
                  </g>
                ))}
                {/* Zero-error points: a caret at the one-sided upper bound, never dropped. */}
                {zeroError.map((point) => (
                  <path
                    key={`z${point.p}`}
                    className="chart-caret"
                    fill={colour}
                    d={`M ${scales.x(point.p) - 5} ${scales.y(point.ci_high) - 5}
                        L ${scales.x(point.p) + 5} ${scales.y(point.ci_high) - 5}
                        L ${scales.x(point.p)} ${scales.y(point.ci_high) + 4} Z`}
                    onMouseEnter={() =>
                      setHover({ point, x: scales.x(point.p), y: scales.y(point.ci_high) })
                    }
                  />
                ))}
              </g>
            );
          })}

          {hover && (
            <g className="chart-hover" transform={`translate(${hover.x} ${hover.y})`}>
              <rect x={10} y={-46} width={186} height={54} rx={3} />
              <text x={18} y={-30}>
                {hover.point.decoder}, d={hover.point.distance}, p={hover.point.p.toPrecision(3)}
              </text>
              <text x={18} y={-16}>
                {hover.point.errors === 0
                  ? `0 errors in ${hover.point.shots.toLocaleString()} shots`
                  : `rate ${hover.point.rate.toPrecision(3)}`}
              </text>
              <text x={18} y={-2}>
                {hover.point.errors === 0
                  ? `upper bound ${hover.point.ci_high.toPrecision(3)}`
                  : `95% CI [${hover.point.ci_low.toPrecision(3)}, ${hover.point.ci_high.toPrecision(3)}]`}
              </text>
            </g>
          )}
        </svg>
      )}

      <figcaption className="chart-legend">
        {series.map((entry) => {
          const name = label(entry);
          const off = hidden.has(name);
          return (
            <button
              key={name}
              type="button"
              className="chart-key"
              aria-pressed={!off}
              onClick={() => toggle(name)}
            >
              <span
                className="chart-swatch"
                style={{ background: off ? "var(--muted)" : colours.get(name) }}
              />
              {name}
            </button>
          );
        })}
        {anyZeroError && (
          <span className="chart-key chart-key--static">
            <span className="chart-swatch chart-swatch--caret" />0 errors: upper bound
          </span>
        )}
      </figcaption>
    </figure>
  );
}
