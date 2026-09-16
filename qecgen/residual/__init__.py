"""PyMatching residual-error datasets: features + labels for a two-stage decoder study.

This package decodes Contract A shots (detection events -> logical flip) with one frozen
``pymatching.Matching`` per dataset and records, per shot, a small set of summary
features, PyMatching's guess and solution weight, the recorded truth and ``pm_wrong``.
It exists to *test* the hypothesis that PyMatching's residual errors are predictable; it
makes no claim that they are.

The artifacts it writes (a feature CSV, a compact raw HDF5, a note, validation and sanity
reports) are research outputs, **not** qecgen datasets: they are not registered in
``qecgen.exporters``, carry no qecgen manifest, and the raw HDF5 keeps its arrays under the
``/residual`` group precisely so that ``qecgen.exporters.hdf5`` reads it as "not ours"
rather than as an interrupted generation run. Nothing here is a physical Pauli fault label,
and ``pm_weight`` is a sum of matching-edge weights, not a fault count.
"""

from __future__ import annotations

SCHEMA_VERSION = 1
"""Version of the shared residual feature schema (column names, order and definitions)."""

__all__ = ["SCHEMA_VERSION"]
