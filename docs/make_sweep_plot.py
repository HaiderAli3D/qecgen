"""Re-plot docs/images/sweep-threshold.png from the committed evidence sweep.

``docs/evidence/sweep.csv`` (and its ``.threshold.json`` sidecar) is the untouched
output of one real ``qecgen sweep`` run — d=3/5/7, p 0.002..0.020, PyMatching,
``--max-errors 300 --max-shots 2000000``, about 1.2M shots. The README's figure is
re-plotted from it through the tool's own ``plot_threshold`` rather than drawn by hand,
so the figure cannot drift from what the tool actually produces.

Run from the repo root: ``python docs/make_sweep_plot.py``.
"""

from __future__ import annotations

import csv
from pathlib import Path

from qecgen.sweep import SweepPoint, plot_threshold

DOCS = Path(__file__).parent
SRC = DOCS / "evidence" / "sweep.csv"
OUT = DOCS / "images" / "sweep-threshold.png"


def main() -> None:
    points = []
    with SRC.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            points.append(
                SweepPoint(
                    decoder=row["decoder"],
                    distance=int(row["distance"]),
                    p=float(row["p"]),
                    rounds=int(row["rounds"]),
                    noise_model=row["noise_model"],
                    basis=row["basis"],
                    shots=int(row["shots"]),
                    errors=int(row["errors"]),
                    discards=int(row["discards"]),
                )
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plot_threshold(
        points,
        OUT,
        title="Rotated surface code, memory-Z, uniform circuit-level noise (PyMatching)",
    )
    print(f"wrote {OUT} from {len(points)} points")


if __name__ == "__main__":
    main()
