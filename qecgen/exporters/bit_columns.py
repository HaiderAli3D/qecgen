"""Bit-to-column encoding shared by every format that writes one column per bit.

It lives here rather than in ``csv_table.py``, where it was written, because ``csv`` and
``ml_csv`` now depend on it byte for byte -- the same reason ``structure_json.py`` was
lifted out of ``jsonl.py``. Two copies of these rules would be two chances to get the
little-endian order or the literal ``"0"``/``"1"`` comparison subtly wrong, and both
failure modes produce a file that parses cleanly and means something else.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np

__all__ = [
    "COLUMN_ORDER",
    "SIZE_WARNING_THRESHOLD",
    "bit_cells",
    "bits_from_cells",
    "block_slices",
    "ordered_header",
    "require_row_in_order",
    "require_zero_padding",
    "warn_if_large",
]

COLUMN_ORDER = ("index", "target", "feature", "environment", "mechanism")
"""The order the tabular formats lay their column blocks out in.

**One tuple, three consumers.** The header builder, the row writer and the read offsets all
derive from this. They used to be three independent hardcodings that agreed only by
convention, and `write` never consulted the header it had just emitted -- so changing two of
the three stored detector bits under the target's column name. Every guard in those modules
is a *format* guard rather than an order guard: `bits_from_cells` compares cells literally
against "0"/"1", and a detector bit is indistinguishable from an observable bit to it. The
round trip stayed green because both sides shared the header builder, and the damage only
appeared as a model that trained on nothing.

Blocks absent from a file contribute no columns, so a single-environment dataset simply has
an empty ``environment`` block rather than a different order.
"""


def ordered_header(blocks: Mapping[str, list[str]]) -> list[str]:
    """The header row implied by a set of named column blocks."""
    return [name for block in COLUMN_ORDER for name in blocks.get(block, ())]


def block_slices(widths: Mapping[str, int]) -> dict[str, slice]:
    """Where each block sits in a data row, resolved once rather than per row.

    Same input as :func:`ordered_header` reduced to widths, so a row's slices and the header
    cannot describe different layouts.
    """
    slices: dict[str, slice] = {}
    start = 0
    for block in COLUMN_ORDER:
        width = widths.get(block, 0)
        slices[block] = slice(start, start + width)
        start += width
    return slices


SIZE_WARNING_THRESHOLD = 100_000
"""Shot count above which a one-column-per-bit text format is an actively bad idea."""

_BIT = {"0": False, "1": True}


def require_zero_padding(array: np.ndarray, n_bits: int, name: str) -> None:
    """Refuse packed input whose padding bits are set.

    A one-column-per-bit format stores ``n_bits`` columns, so a bit past the declared
    width has nowhere to go: unpacking drops it and repacking recreates it as zero. Stim
    zeroes its padding (measured: no padding bit set across 500 shots of an unrotated d=3
    circuit, which pads 36 detectors into 5 bytes), so this never fires on generated data.
    On a hand-built or foreign array it would otherwise round-trip to *different bytes*,
    invisibly -- the table looks identical either way, and the damage only surfaces later
    as a ``content_hash`` that no longer matches.
    """
    remainder = n_bits % 8
    if remainder == 0 or array.size == 0:
        return
    if bool(np.any(array[:, -1] >> remainder)):
        raise ValueError(
            f"{name} has bits set past the declared width of {n_bits}; this format writes "
            f"one column per bit and cannot represent them, so the file would read back as "
            f"different bytes than were written"
        )


def bit_cells(bits: np.ndarray) -> np.ndarray:
    """Render a bool array as an array of ``"0"``/``"1"`` cells."""
    return np.where(bits, "1", "0")


def bits_from_cells(cells: list[str], where: str, column: str) -> np.ndarray:
    """Parse a row slice of ``0``/``1`` cells into a bool row.

    Strict by literal comparison. A spreadsheet re-saves a 0/1 column formatted as boolean
    into ``TRUE``/``FALSE``, and folding an empty cell to ``0`` would fabricate "no
    detection event" for a shot that had one. Both are refused rather than coerced.
    """
    try:
        return np.fromiter((_BIT[cell] for cell in cells), dtype=bool, count=len(cells))
    except KeyError as exc:
        raise ValueError(
            f"{where}: {column} cell {exc.args[0]!r} is neither '0' nor '1'. Values are "
            f"compared literally: a spreadsheet writes TRUE/FALSE for a boolean-formatted "
            f"column, and an empty cell folded to 0 would invent a shot with no detection "
            f"events"
        ) from None


def require_row_in_order(cell: str, row_index: int, where: str, shot_column: str) -> None:
    """Refuse a table whose rows have been reordered.

    Sorting is the one thing a spreadsheet -- and ``df.sort_values``, and a shuffled
    ``DataLoader`` written back -- makes trivially easy, and it severs the correspondence
    between a shot's detectors, its environment id and its mechanism labels while leaving
    a perfectly well-formed file. The shot number is compared as text against the row's
    own position so that no numeric coercion can paper over a gap.
    """
    if cell != str(row_index):
        raise ValueError(
            f"{where}: {shot_column} is {cell!r} but this is row {row_index}. Rows must "
            f"stay in the order they were written: sorting them severs the correspondence "
            f"between a shot's detectors, its environment and its mechanism labels while "
            f"leaving a perfectly well-formed file"
        )


def warn_if_large(n_shots: int, n_columns: int, format_name: str) -> None:
    """Warn before writing a text table that will be unreasonably large.

    ``stacklevel=2`` so the warning points at the exporter's ``write``, not in here. The
    suite runs under ``filterwarnings = ["error::DeprecationWarning"]``; this is a
    ``UserWarning`` and so still warns rather than failing a run.
    """
    if n_shots <= SIZE_WARNING_THRESHOLD:
        return
    warnings.warn(
        f"{format_name} writes one column per bit: {n_shots:,} shots x {n_columns:,} "
        f"columns is a very large text file. Prefer hdf5 or npz above "
        f"{SIZE_WARNING_THRESHOLD:,} shots.",
        UserWarning,
        stacklevel=2,
    )
