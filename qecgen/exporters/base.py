"""The :class:`Exporter` protocol.

Adding a new format — including the Nexus format, once its schema is known — means
writing one module that satisfies this protocol and registering it in
:mod:`qecgen.exporters`. No other file needs to change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from qecgen.dataset import InMemoryDataset, StructureLevel

__all__ = ["Exporter", "require_level_agreement"]


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
        error model. Formats that cannot carry provenance simply omit it.

        ``structure_level`` must agree with ``dataset.meta.structure_level``; a
        disagreement means the file would describe itself incorrectly.
        """
        ...

    def read(self, path: Path) -> InMemoryDataset:
        """Read a dataset previously written by :meth:`write`."""
        ...
