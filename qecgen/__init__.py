"""qecgen: surface code QEC syndrome-to-logical-frame dataset generation.

Every dataset this package writes maps per-shot **detection events** to per-shot
**logical observable flips** (Contract A in ``DATA_CONTRACT.md``). Optionally it also
records which abstract detector-error-model mechanisms fired (Contract B).

It never records physical Pauli faults on data qubits. See ``DATA_CONTRACT.md`` for why
that is not merely unimplemented but underdetermined.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
