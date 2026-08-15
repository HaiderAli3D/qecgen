/**
 * The rotated surface-code lattice, drawn from the parameters currently in the form.
 *
 * Not decoration. A rotated distance-d code is d x d data qubits with (d-1)^2 interior
 * plaquettes and 2(d-1) boundary ones — which is exactly the d^2-1 stabilizers the
 * detector count is built from. Seeing the figure answers "what does d=7 actually mean"
 * faster than the number does.
 *
 * Rotated only, and labelled as such. The unrotated layout is a different geometry —
 * more data qubits, more stabilizers — and this component has no drawing for it, so it
 * takes no `rotated` flag: a flag that only changed the caption produced a figure
 * claiming to be an unrotated code with d^2-1 stabilizers, which is wrong on both
 * counts. A caller whose run is unrotated says so beside the figure, not through it.
 *
 * It doubles as the progress display for a running job: plaquettes fill in as chunks
 * land, so progress is shown in the geometry of the thing being sampled. A numeric bar
 * sits beside it for the precision the figure cannot give.
 */

interface Plaquette {
  x: number;
  y: number;
  kind: "x" | "z";
  boundary: "top" | "bottom" | "left" | "right" | null;
}

const CELL = 14;

/**
 * Plaquettes of a rotated distance-d code, in reading order.
 *
 * Interior plaquettes checkerboard between X and Z. Boundary plaquettes are the
 * half-weight ones along two opposite edges per type, which is what makes the count come
 * out at d^2-1 rather than (d-1)^2.
 */
function rotatedPlaquettes(distance: number): Plaquette[] {
  const cells: Plaquette[] = [];
  for (let row = 0; row < distance - 1; row++) {
    for (let col = 0; col < distance - 1; col++) {
      cells.push({
        x: col + 1,
        y: row + 1,
        kind: (row + col) % 2 === 0 ? "z" : "x",
        boundary: null,
      });
    }
  }
  for (let col = 0; col < distance - 1; col++) {
    if (col % 2 === 0) {
      cells.push({ x: col + 1, y: 0, kind: "x", boundary: "top" });
    } else {
      cells.push({ x: col + 1, y: distance, kind: "x", boundary: "bottom" });
    }
  }
  for (let row = 0; row < distance - 1; row++) {
    if (row % 2 === 0) {
      cells.push({ x: distance, y: row + 1, kind: "z", boundary: "right" });
    } else {
      cells.push({ x: 0, y: row + 1, kind: "z", boundary: "left" });
    }
  }
  return cells;
}

function boundaryPath(cell: Plaquette): string {
  const x = cell.x * CELL;
  const y = cell.y * CELL;
  const r = CELL / 2;
  switch (cell.boundary) {
    case "top":
      return `M ${x} ${y + CELL} A ${r} ${r} 0 0 1 ${x + CELL} ${y + CELL} Z`;
    case "bottom":
      return `M ${x} ${y} A ${r} ${r} 0 0 0 ${x + CELL} ${y} Z`;
    case "left":
      return `M ${x + CELL} ${y} A ${r} ${r} 0 0 0 ${x + CELL} ${y + CELL} Z`;
    default:
      return `M ${x} ${y} A ${r} ${r} 0 0 1 ${x} ${y + CELL} Z`;
  }
}

export interface LatticeProps {
  distance: number;
  basis: string;
  /** 0 to 1. Below 1, later plaquettes are drawn pending. */
  progress?: number;
  label?: string;
}

export function Lattice({ distance, basis, progress = 1, label }: LatticeProps) {
  // Clamped so the drawing stays legible, but the caption keeps quoting the requested
  // distance: reporting the clamped one as if it were asked for would have d=21 in the
  // form and "d=15" under the figure with no hint they differ.
  const d = Math.min(Math.max(distance, 2), 15);
  const clamped = d !== distance;
  const cells = rotatedPlaquettes(d);
  const filled = Math.round(cells.length * Math.min(Math.max(progress, 0), 1));
  // Content runs from CELL/2 to (d + 0.5) * CELL on both axes -- the boundary
  // semicircles bulge half a cell past the data-qubit grid on every side -- so the
  // tight box is d cells square. The old (d + 1) * CELL box left a full cell of dead
  // margin at the right and bottom and none at the top-left, off-centering the figure.
  const span = d * CELL;
  // The stabilizer count belongs to the drawn lattice, so it stays next to the drawn
  // distance when the two differ.
  const drawn = clamped
    ? `drawn at distance ${d}, ${cells.length} stabilizers`
    : `${cells.length} stabilizers`;

  const qubits = [];
  for (let row = 0; row < d; row++) {
    for (let col = 0; col < d; col++) {
      qubits.push({ cx: (col + 1) * CELL, cy: (row + 1) * CELL });
    }
  }

  return (
    <figure className="lattice">
      <svg
        viewBox={`${CELL / 2} ${CELL / 2} ${span} ${span}`}
        role="img"
        aria-label={`Distance ${distance} rotated surface code, ${drawn}, memory ${basis.toUpperCase()}`}
      >
        <g className="lattice-plaquettes">
          {cells.map((cell, index) => {
            const state = index < filled ? "done" : "pending";
            const className = `plaquette plaquette--${cell.kind} plaquette--${state}`;
            return cell.boundary ? (
              <path key={index} className={className} d={boundaryPath(cell)} />
            ) : (
              <rect
                key={index}
                className={className}
                x={cell.x * CELL}
                y={cell.y * CELL}
                width={CELL}
                height={CELL}
              />
            );
          })}
        </g>
        <g className="lattice-qubits">
          {qubits.map((q, index) => (
            <circle key={index} cx={q.cx} cy={q.cy} r={2.1} />
          ))}
        </g>
      </svg>
      <figcaption>
        {label ?? (
          <>
            d={distance}
            {clamped ? ` (drawn at ${d})` : ""} · rotated · memory {basis.toUpperCase()}
          </>
        )}
      </figcaption>
    </figure>
  );
}
