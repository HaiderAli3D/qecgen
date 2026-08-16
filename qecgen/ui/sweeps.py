"""Browsing threshold sweeps that already exist on disk.

The sibling of :mod:`qecgen.ui.datasets`, and shaped like it on purpose. Two rules differ,
both because a sweep is three files rather than one.

**The ``.threshold.json`` sidecar is what identifies a sweep.** Not the ``.csv``, which is
also a dataset extension and which :mod:`qecgen.ui.datasets` already has to disambiguate;
not the ``.png``, which could be any image someone dropped in the data root. The three
files share a stem, so finding the sidecar finds the set — and a sweep whose plot is
missing is listed with the plot missing rather than hidden.

**The results table is parsed, never recomputed.** Every number a chart draws — the rate
and both Clopper-Pearson bounds — is read from the columns :func:`qecgen.sweep.write_csv`
already wrote. Nothing here derives a statistic, and neither does the browser: the chart is
a *view* of the artifact, and ``sweep.png`` remains the artifact of record. A second
implementation of the interval arithmetic is exactly the drift the generated-figure rule in
CLAUDE.md exists to prevent.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qecgen.run import PARTIAL_PREFIX
from qecgen.ui.datasets import resolve_within

__all__ = [
    "SUMMARY_SUFFIX",
    "SweepEntry",
    "list_sweeps",
    "resolve_within",
    "sweep_detail",
]

SUMMARY_SUFFIX = ".threshold.json"
"""What :func:`qecgen.run.run_sweep_job` names the summary sidecar. A multi-part suffix, so
``Path.suffix`` (which returns only ``.json``) cannot be used to match it."""


def _stem_of(summary: Path) -> str:
    """The shared stem of a sweep's three files, from its summary path."""
    return summary.name[: -len(SUMMARY_SUFFIX)]


@dataclass(frozen=True, slots=True)
class SweepEntry:
    """One sweep in the browser listing."""

    stem: str
    """Path relative to the data root, without any of the three suffixes."""
    summary_path: str
    results_path: str | None
    plot_path: str | None
    modified_at: float
    size_bytes: int
    """The results table's size. The sidecar is a few kilobytes whatever the sweep cost."""
    decoders: list[str]
    crossings: dict[str, float | None]
    unreadable: str | None

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {
            "stem": self.stem,
            "summary_path": self.summary_path,
            "results_path": self.results_path,
            "plot_path": self.plot_path,
            "modified_at": self.modified_at,
            "size_bytes": self.size_bytes,
            "decoders": self.decoders,
            "crossings": self.crossings,
            "unreadable": self.unreadable,
        }


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def list_sweeps(root: Path) -> list[SweepEntry]:
    """Every sweep under ``root``, newest first.

    Staging directories are skipped for the same reason datasets skip them: their contents
    are mid-write by definition, and :func:`qecgen.run.staged` commits all three files
    together, so a sweep that is still running has nothing worth listing yet.
    """
    if not root.is_dir():
        return []
    entries: list[SweepEntry] = []
    for summary in root.rglob(f"*{SUMMARY_SUFFIX}"):
        if not summary.is_file():
            continue
        if any(part.startswith(PARTIAL_PREFIX) for part in summary.relative_to(root).parts):
            continue
        stem = _stem_of(summary)
        if not stem:
            # A file named exactly `.threshold.json` -- rglob matches dotfiles. Its stem is
            # empty, so the sibling paths would collapse onto the containing directory and
            # resolve *outside* the data root, where `_relative` raises and 500s the whole
            # listing. No sweep this tool writes can have an empty stem.
            continue
        stem_path = summary.parent / stem
        results = stem_path.with_name(f"{stem_path.name}.csv")
        plot = stem_path.with_name(f"{stem_path.name}.png")

        decoders: list[str] = []
        crossings: dict[str, float | None] = {}
        modified_at = 0.0
        size_bytes = 0
        unreadable: str | None = None
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
            by_decoder = dict(payload["decoders"])
            decoders = sorted(by_decoder)
            # Subscript, not `.get`. `crossing_p: null` is a *measurement* -- the interval
            # ordering never demonstrably reversed in the swept range -- while an absent
            # key means the file does not carry the field at all. Defaulting the second to
            # the first would render a quotable negative result ("no threshold here")
            # synthesised from a missing field. The KeyError is what flags the file below.
            crossings = {name: by_decoder[name]["crossing_p"] for name in decoders}
            # Inside the guard: a file removed between rglob and stat raises OSError, and
            # letting that escape would 500 the whole listing over one vanished sweep.
            modified_at = summary.stat().st_mtime
            size_bytes = results.stat().st_size if results.is_file() else 0
        except Exception as exc:
            # Listed with the reason attached rather than skipped, matching how a corrupt
            # dataset is reported: a sweep that silently vanishes from the list looks like
            # it was never run.
            unreadable = f"{type(exc).__name__}: {exc}"

        entries.append(
            SweepEntry(
                stem=_relative(root, stem_path),
                summary_path=_relative(root, summary),
                results_path=_relative(root, results) if results.is_file() else None,
                plot_path=_relative(root, plot) if plot.is_file() else None,
                modified_at=modified_at,
                size_bytes=size_bytes,
                decoders=decoders,
                crossings=crossings,
                unreadable=unreadable,
            )
        )
    entries.sort(key=lambda entry: entry.modified_at, reverse=True)
    return entries


def _read_points(results: Path) -> list[dict[str, Any]]:
    """Parse the results table into typed rows.

    ``errors`` is carried through because it decides how a point is *drawn*, not what it
    is worth: a zero-error point has no place on a log axis at its estimate of 0, so
    :func:`qecgen.sweep.plot_threshold` draws it as a downward caret at its one-sided upper
    bound. The chart mirrors that rule, and it is the only piece of plotting semantics that
    exists in two places.
    """
    with results.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "decoder": row["decoder"],
            "distance": int(row["distance"]),
            "p": float(row["p"]),
            "shots": int(row["shots"]),
            "errors": int(row["errors"]),
            "discards": int(row["discards"]),
            "rate": float(row["logical_error_rate"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
        }
        for row in rows
    ]


def sweep_detail(root: Path, summary: Path) -> dict[str, Any]:
    """One sweep's full contents: its summary verbatim, and its points grouped for drawing.

    The summary is passed through as written rather than reshaped. It already carries the
    crossing with the method that produced it, the suppression fits, the censored points
    and the "reported, not asserted" disclaimer — and that disclaimer travelling with the
    numbers is the point of it being a field rather than a comment.

    Raises:
        FileNotFoundError: when the results table is missing, which means the sweep is not
            renderable even though its sidecar survived.
    """
    payload = json.loads(summary.read_text(encoding="utf-8"))
    stem_path = summary.parent / _stem_of(summary)
    results = stem_path.with_name(f"{stem_path.name}.csv")
    plot = stem_path.with_name(f"{stem_path.name}.png")
    if not results.is_file():
        raise FileNotFoundError(
            f"no results table beside {_relative(root, summary)}; the sweep's numbers are "
            "in the .csv, and without it there is nothing to plot"
        )

    points = _read_points(results)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for point in points:
        grouped.setdefault((point["decoder"], point["distance"]), []).append(point)

    return {
        "stem": _relative(root, stem_path),
        "results_path": _relative(root, results),
        "plot_path": _relative(root, plot) if plot.is_file() else None,
        "summary_path": _relative(root, summary),
        "series": [
            {
                "decoder": decoder,
                "distance": distance,
                "points": sorted(group, key=lambda point: float(point["p"])),
            }
            for (decoder, distance), group in sorted(grouped.items())
        ],
        "summary": payload,
    }
