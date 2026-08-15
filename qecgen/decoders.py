"""Decoder name resolution against sinter's registry.

``qecgen`` implements no decoder and adapts no decoder. This module exists for one
reason: ``sinter.collect`` discovers a bad decoder name, or a missing backend package,
only *inside a worker process* after every task has already been built and dispatched --
as ``NotImplementedError(f"Unrecognized decoder: {decoder!r}")`` from ``sinter._decoding``,
or as an ``ImportError`` raised from ``compile_decoder_for_dem``. Answering both questions
before collection starts turns a multiprocessing traceback into one actionable line.

Nothing here imports ``mwpf`` or ``fusion_blossom``. Availability is probed by module
*name* via :func:`importlib.util.find_spec`. That is deliberate. Importing a Rust
extension merely to ask whether it exists is wasteful, and an ``import mwpf`` anywhere in
this package would be the first brick of the decoder adapter layer the README lists as out
of scope. If an ``mwpf.*`` entry ever appears in the mypy overrides in ``pyproject.toml``,
that boundary has been crossed.

Adding a bespoke decoder later needs no code here: ``sinter.collect`` already accepts
``custom_decoders: dict[str, sinter.Decoder]``, and ``sinter.Decoder`` is the protocol such
a thing would satisfy. That is why this module resolves *names* and defines no protocol of
its own -- see the README section on why there is no ``Decoder`` protocol.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass

import sinter

__all__ = [
    "ALWAYS_AVAILABLE",
    "BACKING_PACKAGES",
    "DEFAULT_DECODERS",
    "DecoderAvailability",
    "check_decoder",
    "known_decoder_names",
    "resolve_decoders",
]

DEFAULT_DECODERS: tuple[str, ...] = ("pymatching",)
"""Selected when ``--decoder`` is not passed. Preserves pre-existing behaviour exactly."""

BACKING_PACKAGES: dict[str, tuple[str, str]] = {
    # sinter decoder name -> (importable module name, pip install target)
    "fusion_blossom": ("fusion_blossom", "fusion-blossom"),
    "hypergraph_union_find": ("mwpf", "mwpf"),
    "mw_parity_factor": ("mwpf", "mwpf"),
}
"""Third-party package each sinter decoder needs at decode time.

Names absent from this mapping are served by packages ``qecgen`` already depends on and
are always usable. This is a claim about sinter's *implementation* rather than its API, so
it can go stale across a sinter bump; ``tests/test_decoders.py`` asserts it partitions
``sinter.BUILT_IN_DECODERS`` exactly. That test is deterministic because sinter is pinned.
"""

ALWAYS_AVAILABLE: frozenset[str] = frozenset(
    {
        "vacuous",
        "pymatching",
        "pymatching-correlated",
    }
)
"""Decoders backed by ``qecgen``'s own dependencies, so always usable."""


def known_decoder_names() -> tuple[str, ...]:
    """Every decoder name the installed sinter recognises, sorted.

    ``sinter.BUILT_IN_DECODERS`` is public API and is used directly rather than mirrored
    into a local allowlist. A mirrored list is a second source of truth that goes stale
    silently on the next sinter bump.
    """
    # sinter ships no py.typed marker, so its attributes are `Any` under --strict and a
    # bare return would trip warn_return_any. Launder through an annotated local, matching
    # the pattern already used in sweep.py.
    names: list[str] = sorted(sinter.BUILT_IN_DECODERS)
    return tuple(names)


@dataclass(frozen=True, slots=True)
class DecoderAvailability:
    """Whether one decoder name can run here, and precisely why not if it cannot."""

    name: str
    known: bool
    """True when the installed sinter recognises the name."""
    backing_module: str | None
    """Importable module that must be present, or None when qecgen's own deps suffice.

    Tracked separately from :attr:`backing_package` because the two differ: the
    ``fusion-blossom`` distribution installs a ``fusion_blossom`` module. Reporting the
    pip target as the missing module name sends a reader looking for something that was
    never going to exist under that spelling.
    """
    backing_package: str | None
    """pip target that must be installed, or None when qecgen's own deps suffice."""
    installed: bool
    """True when the backing module is importable (or none is needed) and the name is
    known.

    An unknown name reports False even though it needs no module: this field answers
    "is the backend present for something sinter can run", and an unrecognised decoder
    has no backend to be present. :meth:`problem` reports the unknown-name case first,
    so the two are never conflated in a message.
    """

    @property
    def usable(self) -> bool:
        """True when a sweep can actually run this decoder."""
        return self.known and self.installed

    def problem(self) -> str | None:
        """An actionable message, or None when usable.

        Distinguishes "no such decoder" from "valid decoder, absent backend". Collapsing
        the two produces the worst possible message: a user who typed a correct name is
        told the name is wrong.
        """
        if self.known and self.installed:
            return None
        if not self.known:
            valid = ", ".join(known_decoder_names())
            return f"unknown decoder {self.name!r}; the installed sinter provides: {valid}"
        return (
            f"decoder {self.name!r} is valid but its backend is not installed: no module "
            f'named {self.backing_module!r}. Install it with: pip install -e ".[decoders]" '
            f"(or pip install {self.backing_package}). Nothing else about qecgen requires it."
        )


def check_decoder(name: str) -> DecoderAvailability:
    """Report whether one decoder name is usable in this environment."""
    known = name in sinter.BUILT_IN_DECODERS
    backing = BACKING_PACKAGES.get(name)
    if backing is None:
        return DecoderAvailability(
            name=name,
            known=known,
            backing_module=None,
            backing_package=None,
            installed=known,
        )
    module_name, pip_target = backing
    installed = importlib.util.find_spec(module_name) is not None
    return DecoderAvailability(
        name=name,
        known=known,
        backing_module=module_name,
        backing_package=pip_target,
        installed=installed,
    )


def resolve_decoders(names: Sequence[str] | None) -> tuple[str, ...]:
    """Normalise, deduplicate and validate a decoder selection.

    Preserves first-seen order, so ``--decoder pymatching --decoder pymatching`` does not
    collect the same decoder twice. sinter would happily do so, and ``write_csv`` would
    then emit duplicate rows for it.

    Raises:
        ValueError: for an unknown name, listing every valid name; or for a known name
            whose backing package is absent, naming the pip target and the extra.
    """
    if not names:
        return DEFAULT_DECODERS

    ordered: list[str] = []
    for name in names:
        if name not in ordered:
            ordered.append(name)

    problems = [problem for n in ordered if (problem := check_decoder(n).problem())]
    if problems:
        raise ValueError("; ".join(problems))
    return tuple(ordered)
