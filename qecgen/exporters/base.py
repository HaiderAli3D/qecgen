"""The :class:`Exporter` protocol.

Adding a new format — including the Nexus format, once its schema is known — means
writing one module that satisfies this protocol and registering it in
:mod:`qecgen.exporters`. No other file needs to change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from qecgen.dataset import InMemoryDataset, StructureLevel

__all__ = [
    "Exporter",
    "NotAQecgenDatasetError",
    "recorded_structure_level",
    "require_level_agreement",
]


class NotAQecgenDatasetError(ValueError):
    """A file carrying a registered extension that this tool did not write.

    Distinct from a corrupt file, and the distinction is load-bearing. ``.csv`` is both a
    dataset extension *and* the extension ``qecgen sweep`` writes its threshold-results
    table with, so a data root can legitimately hold intact ``.csv`` files that are simply
    not datasets. Reporting those as corruption puts a red flag on a healthy file, and a
    corruption flag with routine false positives is one a reader learns to skip past —
    which is exactly how the flag that means "a worker died mid-write" stops being seen.

    A ``ValueError`` subclass so every existing ``except ValueError`` boundary — the CLI's
    argument errors, the UI's listing fallback — keeps working unchanged.
    """


def require_level_agreement(dataset: InMemoryDataset, level: StructureLevel) -> None:
    """Refuse to write a file whose manifest would misdescribe its own payload.

    ``write()`` used to take a ``structure_level`` entirely independent of
    ``meta.structure_level``, so a file could record ``dem`` while carrying nothing, or
    the reverse. Downstream that is indistinguishable from a corrupt file.
    """
    if level is not dataset.meta.structure_level:
        raise ValueError(
            f"structure_level={level} disagrees with meta.structure_level="
            f"{dataset.meta.structure_level}; the file would describe itself "
            "incorrectly. Rebuild the dataset at the level you intend to write."
        )
    # Level agreement alone is not enough: a dataset whose manifest claims structure
    # while carrying none used to write cleanly in the formats that iterate over
    # whatever is present (HDF5, NPZ), producing a manifest that over-claims — which is
    # indistinguishable downstream from a corrupt file. JSONL refused this state
    # explicitly; the refusal belongs here so every exporter inherits it.
    if level is not StructureLevel.NONE and dataset.structure is None:
        raise ValueError(
            f"structure_level={level} was requested but the dataset carries no "
            "structure, so the manifest would claim structure the file cannot hold"
        )


@runtime_checkable
class Exporter(Protocol):
    """Writes and reads one on-disk representation of a dataset."""

    @property
    def format_name(self) -> str:
        """Short name used by ``--format`` and ``qecgen formats``."""
        ...

    @property
    def extension(self) -> str:
        """Conventional file extension, including the leading dot."""
        ...

    @property
    def streaming(self) -> bool:
        """True when this format supports appending chunks without buffering."""
        ...

    @property
    def structure_round_trip(self) -> bool:
        """True when :meth:`read` reconstructs the structure :meth:`write` was given.

        Formats that declare False may still *record* structure, but must downgrade
        the recorded ``structure_level`` to what they can actually reproduce, so a
        manifest never claims more than the payload holds.
        """
        ...

    @property
    def carries_provenance(self) -> bool:
        """True when this format stores the circuit and DEM text at ``FULL``.

        The same honesty rule as :attr:`structure_round_trip`, applied to the other half
        of what ``FULL`` promises. ``StructureLevel.FULL`` is documented as "everything,
        including per-environment circuit and DEM text", so a format that declines to
        carry that text must not record ``full`` — it must downgrade to the level it
        actually delivers. Declining is a legitimate choice: in a flat text format read
        by iterating lines, the provenance block is one loop away from a decoder, and
        under ``FROZEN_PRIOR`` it holds precisely the distribution being withheld.

        :func:`recorded_structure_level` applies the rule; declaring the property is all
        an exporter has to do.
        """
        ...

    def write(
        self,
        dataset: InMemoryDataset,
        path: Path,
        structure_level: StructureLevel = StructureLevel.NONE,
    ) -> None:
        """Write ``dataset`` to ``path``.

        The round-trip contract covers **arrays and manifest parameters**:
        ``read(write(d))`` reproduces every array and every field of
        :meth:`DatasetMeta.to_json_dict` exactly.

        Circuit and DEM text are *not* manifest fields. They are provenance, written
        only at ``StructureLevel.FULL`` and stored physically apart from the manifest
        so a decoder reading the manifest cannot recover a frozen-prior test set's own
        error model. A format that omits provenance must also stop *claiming* it — see
        :attr:`carries_provenance` and :func:`recorded_structure_level`.

        ``structure_level`` must agree with ``dataset.meta.structure_level``; a
        disagreement means the file would describe itself incorrectly.
        """
        ...

    def read(self, path: Path) -> InMemoryDataset:
        """Read a dataset previously written by :meth:`write`."""
        ...


def recorded_structure_level(exporter: Exporter, level: StructureLevel) -> StructureLevel:
    """The level ``exporter`` may honestly record when asked to write ``level``.

    One place for both downgrade rules, because they are one rule: **a manifest never
    claims more than its file holds.** Parquet cannot rebuild structure on read, so it
    records ``none``; JSONL rebuilds structure but declines to carry provenance, so it
    records ``dem`` rather than ``full``. Those were previously two independent
    conventions implemented in two exporters, which is how ``full`` came to mean three
    different things across five formats — and the one that lied did so in a field a
    reader has no way to check.

    The in-memory ``dataset.meta`` is never modified; only the copy being persisted is.
    """
    if not exporter.structure_round_trip:
        return StructureLevel.NONE
    if level is StructureLevel.FULL and not exporter.carries_provenance:
        return StructureLevel.DEM
    return level
