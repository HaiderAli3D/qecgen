"""Regenerate the SVG diagrams embedded in README.md.

The lattice figure reuses the exact plaquette algorithm and palette of
``frontend/src/components/Lattice.tsx`` rather than approximating them: the README's
figure must be the figure the tool draws, or the README teaches a different code than
the files contain. The bit-packing figure computes its byte values instead of quoting
them, for the same reason.

Run from the repo root: ``python docs/make_diagrams.py``.
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).parent / "images"

# The frontend's light palette (frontend/src/styles.css). Painted explicitly so the
# figures read identically on GitHub's light and dark themes.
GROUND = "#edeee9"
SURFACE = "#f7f7f4"
INK = "#1b1f1d"
MUTED = "#5d645f"
Z_FILL = "#3a2e6e"
X_FILL = "#0e7c7b"
GOOD = "#0e7c3f"
BAD = "#a33232"

FONT = 'font-family="Segoe UI, Helvetica, Arial, sans-serif"'


def text(
    x: float,
    y: float,
    content: str,
    size: float = 15,
    fill: str = INK,
    weight: str = "400",
    anchor: str = "start",
) -> str:
    """One SVG text element; anchors and typography vary, the font never does."""
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="{fill}" {FONT}>{content}</text>'
    )


def rotated_plaquettes(distance: int) -> list[tuple[int, int, str, str | None]]:
    """(x, y, kind, boundary) — a line-for-line port of Lattice.tsx's algorithm."""
    cells: list[tuple[int, int, str, str | None]] = []
    for row in range(distance - 1):
        for col in range(distance - 1):
            cells.append((col + 1, row + 1, "z" if (row + col) % 2 == 0 else "x", None))
    for col in range(distance - 1):
        if col % 2 == 0:
            cells.append((col + 1, 0, "x", "top"))
        else:
            cells.append((col + 1, distance, "x", "bottom"))
    for row in range(distance - 1):
        if row % 2 == 0:
            cells.append((distance, row + 1, "z", "right"))
        else:
            cells.append((0, row + 1, "z", "left"))
    return cells


def boundary_path(x: float, y: float, cell: float, boundary: str) -> str:
    """The half-disc arcs, ported from Lattice.tsx's boundaryPath."""
    r = cell / 2
    if boundary == "top":
        return f"M {x} {y + cell} A {r} {r} 0 0 1 {x + cell} {y + cell} Z"
    if boundary == "bottom":
        return f"M {x} {y} A {r} {r} 0 0 0 {x + cell} {y} Z"
    if boundary == "left":
        return f"M {x + cell} {y} A {r} {r} 0 0 0 {x + cell} {y + cell} Z"
    return f"M {x} {y} A {r} {r} 0 0 1 {x} {y + cell} Z"


def lattice_svg(distance: int = 3, rounds: int = 3) -> str:
    cell = 58.0
    ox, oy = 30.0, 46.0
    n_stab = distance * distance - 1
    parts: list[str] = []
    for x, y, kind, boundary in rotated_plaquettes(distance):
        fill = Z_FILL if kind == "z" else X_FILL
        px, py = ox + x * cell, oy + y * cell
        if boundary is None:
            parts.append(f'<rect x="{px}" y="{py}" width="{cell}" height="{cell}" fill="{fill}"/>')
        else:
            parts.append(f'<path d="{boundary_path(px, py, cell, boundary)}" fill="{fill}"/>')
    for row in range(distance):
        for col in range(distance):
            cx, cy = ox + (col + 1) * cell, oy + (row + 1) * cell
            parts.append(
                f'<circle cx="{cx}" cy="{cy}" r="9" fill="{GROUND}" '
                f'stroke="{INK}" stroke-width="2.5"/>'
            )

    lx = ox + (distance + 1) * cell + 44
    # RUF001-flagged characters below (multiplication sign, superscript two, minus) are
    # deliberate display typography, not identifiers.
    parts.append(
        f'<circle cx="{lx + 10}" cy="104" r="9" fill="{GROUND}" stroke="{INK}" stroke-width="2.5"/>'
    )
    parts.append(text(lx + 32, 109, f"data qubit — {distance}×{distance} = {distance**2}"))  # noqa: RUF001
    parts.append(f'<rect x="{lx}" y="134" width="20" height="20" fill="{Z_FILL}"/>')
    parts.append(text(lx + 32, 149, f"Z stabilizer — {n_stab // 2}"))
    parts.append(f'<rect x="{lx}" y="174" width="20" height="20" fill="{X_FILL}"/>')
    parts.append(text(lx + 32, 189, f"X stabilizer — {n_stab // 2}"))
    parts.append(text(lx, 232, f"{n_stab} stabilizers = d² − 1", fill=MUTED))  # noqa: RUF001
    parts.append(
        text(lx, 256, f"× {rounds} rounds = {n_stab * rounds} detectors per shot", fill=MUTED)  # noqa: RUF001
    )
    parts.append(text(lx, 280, "(rounds defaults to d)", fill=MUTED))

    width = lx + 320
    # Tall enough for the last legend line (y=280) plus the two footer lines; a height
    # computed from the lattice alone clipped both.
    height = 344.0
    title = f"Rotated distance-{distance} surface code — what one shot measures"
    footer1 = "Half-disc plaquettes are the weight-2 boundary checks; interior checks are weight 4."
    footer2 = "Geometry identical to the figure the tool itself draws."
    body = "\n  ".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}"'
        f" {FONT}>\n"
        f'  <rect width="{width:.0f}" height="{height:.0f}" fill="{GROUND}" rx="10"/>\n'
        f"  {text(ox, 30, title, size=17, weight='600')}\n"
        f"  {body}\n"
        f"  {text(ox, height - 36, footer1, size=13.5, fill=MUTED)}\n"
        f"  {text(ox, height - 16, footer2, size=13.5, fill=MUTED)}\n"
        "</svg>\n"
    )


def bit_packing_svg() -> str:
    bits = [1, 0, 1, 1, 0, 0, 1, 0]  # detectors 0..7 of one shot
    byte = sum(b << i for i, b in enumerate(bits))
    big_read = [(byte >> (7 - i)) & 1 for i in range(8)]  # what bitorder="big" yields
    assert byte == 0x4D
    assert big_read != bits

    cw, top_y, byte_y = 64.0, 74.0, 210.0
    ox = 60.0
    parts: list[str] = []
    for i, b in enumerate(bits):
        x = ox + i * cw
        mid = x + (cw - 8) / 2
        parts.append(
            f'<rect x="{x}" y="{top_y}" width="{cw - 8}" height="40" fill="{SURFACE}" '
            f'stroke="{MUTED}" rx="6"/>'
        )
        parts.append(text(mid, top_y + 27, str(b), size=19, weight="600", anchor="middle"))
        parts.append(text(mid, top_y - 12, f"det {i}", size=13, fill=MUTED, anchor="middle"))
    # Byte cells displayed b7..b0 left-to-right, the way a byte is conventionally
    # written. The crossing lines ARE the point: the leftmost detector lands in the
    # rightmost displayed bit.
    for pos in range(8):
        bit_index = 7 - pos
        x = ox + pos * cw
        mid = x + (cw - 8) / 2
        parts.append(
            f'<rect x="{x}" y="{byte_y}" width="{cw - 8}" height="40" fill="{SURFACE}" '
            f'stroke="{MUTED}" rx="6"/>'
        )
        parts.append(
            text(mid, byte_y + 27, str(bits[bit_index]), size=19, weight="600", anchor="middle")
        )
        parts.append(
            text(mid, byte_y + 58, f"bit {bit_index}", size=13, fill=MUTED, anchor="middle")
        )
        det_cx = ox + bit_index * cw + (cw - 8) / 2
        parts.append(
            f'<line x1="{det_cx}" y1="{top_y + 44}" x2="{mid}" y2="{byte_y - 4}" '
            f'stroke="{MUTED}" stroke-width="1.1" opacity="0.55"/>'
        )

    width = ox * 2 + 8 * cw + 240
    little = " ".join(str(b) for b in bits)
    big = " ".join(str(b) for b in big_read)
    bx = ox + 8 * cw + 26
    title = "Little-endian bit packing — detector i is bit i of byte i//8"
    good_line = f'✓ np.unpackbits(row, count=n, bitorder="little")  →  {little}'
    bad_line = f'✗ np.unpackbits(row, count=n) — NumPy defaults to "big"  →  {big}'
    # Notes stay under ~90 characters per line so nothing runs past the right edge.
    note1a = "The default silently reverses every byte: detector 0 reads detector 7's value,"
    note1b = "nothing crashes, and the file stays well-formed."
    note2a = "Packed width never implies true width either — a 3-byte row holds 17 to 24 detectors."
    note2b = "Read n_detectors from the manifest, never from the array shape."
    height = 460.0
    body = "\n  ".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}"'
        f" {FONT}>\n"
        f'  <rect width="{width:.0f}" height="{height:.0f}" fill="{GROUND}" rx="10"/>\n'
        f"  {text(ox, 34, title, size=17, weight='600')}\n"
        f"  {body}\n"
        f"  {text(bx, byte_y + 27, f'= 0x{byte:02X}', size=19)}\n"
        f"  {text(ox, byte_y + 100, good_line, fill=GOOD)}\n"
        f"  {text(ox, byte_y + 128, bad_line, fill=BAD)}\n"
        f"  {text(ox, byte_y + 162, note1a, size=13.5, fill=MUTED)}\n"
        f"  {text(ox, byte_y + 182, note1b, size=13.5, fill=MUTED)}\n"
        f"  {text(ox, byte_y + 212, note2a, size=13.5, fill=MUTED)}\n"
        f"  {text(ox, byte_y + 232, note2b, size=13.5, fill=MUTED)}\n"
        "</svg>\n"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lattice-d3.svg").write_text(lattice_svg(), encoding="utf-8")
    (OUT / "bit-packing.svg").write_text(bit_packing_svg(), encoding="utf-8")
    print(f"wrote {OUT / 'lattice-d3.svg'}")
    print(f"wrote {OUT / 'bit-packing.svg'}")


if __name__ == "__main__":
    main()
