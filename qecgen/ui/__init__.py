"""Local web UI for the dataset-producing commands.

Import cost is deliberate here. ``qecgen.ui.protocol`` and ``qecgen.ui.worker`` depend on
nothing beyond the standard library and :mod:`qecgen.run`, so a generation worker starts
in about half a second and runs on an install without the ``ui`` extra.
``qecgen.ui.app`` is the only module that pulls in FastAPI, and nothing imports it at
package level.
"""

from __future__ import annotations

__all__ = ["DEFAULT_PORT"]

DEFAULT_PORT = 8765
"""Loopback port for ``qecgen ui``. High and unremarkable, to avoid colliding with the
usual 3000/5000/8000 crowd."""
