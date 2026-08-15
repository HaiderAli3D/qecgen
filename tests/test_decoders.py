"""Decoder name resolution.

Every test here runs green without the optional `decoders` extra installed. The one test
that needs `mwpf` is gated on it, because the suite must not require a Rust extension in
order to pass.
"""

from __future__ import annotations

import importlib.util

import pytest
import sinter
from conftest import requires_mwpf

from qecgen.decoders import (
    ALWAYS_AVAILABLE,
    BACKING_PACKAGES,
    DEFAULT_DECODERS,
    check_decoder,
    known_decoder_names,
    resolve_decoders,
)


def test_backing_package_map_partitions_the_sinter_registry() -> None:
    """Every sinter decoder is either always available or has a named backing package.

    Deterministic because sinter is exact-pinned. This is the guard that a sinter bump
    cannot silently introduce a decoder qecgen mis-describes as always usable.
    """
    assert set(BACKING_PACKAGES) | ALWAYS_AVAILABLE == set(sinter.BUILT_IN_DECODERS)
    assert not (set(BACKING_PACKAGES) & ALWAYS_AVAILABLE)


def test_known_decoder_names_matches_sinter() -> None:
    assert known_decoder_names() == tuple(sorted(sinter.BUILT_IN_DECODERS))
    assert "hypergraph_union_find" in known_decoder_names()


def test_unknown_name_is_rejected_and_lists_valid_names() -> None:
    with pytest.raises(ValueError, match="unknown decoder 'pymatchign'") as exc:
        resolve_decoders(["pymatchign"])
    # The message must be usable on its own: it names every alternative.
    assert "hypergraph_union_find" in str(exc.value)
    assert "pymatching" in str(exc.value)


def test_known_but_missing_backend_says_install_not_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A correct name with an absent backend must not be reported as a typo.

    Collapsing the two produces the worst possible message: a user who typed a valid
    decoder name is told the name is wrong.
    """

    def no_module(name: str, package: str | None = None) -> None:
        return None

    monkeypatch.setattr(importlib.util, "find_spec", no_module)
    availability = check_decoder("hypergraph_union_find")
    assert availability.known
    assert not availability.installed
    problem = availability.problem()
    assert problem is not None
    assert "unknown" not in problem
    assert "mwpf" in problem
    assert '".[decoders]"' in problem


def test_missing_backend_message_names_the_module_not_the_pip_target() -> None:
    """`fusion-blossom` installs a `fusion_blossom` module; the message must not conflate them.

    Reporting the pip target as the missing module sends a reader looking for something
    that was never going to exist under that spelling.
    """
    availability = check_decoder("fusion_blossom")
    assert availability.backing_module == "fusion_blossom"
    assert availability.backing_package == "fusion-blossom"
    if not availability.installed:
        problem = availability.problem()
        assert problem is not None
        assert "no module named 'fusion_blossom'" in problem
        assert "pip install fusion-blossom" in problem


def test_pymatching_is_always_usable() -> None:
    availability = check_decoder("pymatching")
    assert availability.usable
    assert availability.backing_package is None
    assert availability.problem() is None


def test_none_resolves_to_pymatching_only() -> None:
    """The no-regression lock on the default: passing no --decoder changes nothing."""
    assert resolve_decoders(None) == DEFAULT_DECODERS == ("pymatching",)
    assert resolve_decoders([]) == ("pymatching",)


def test_repeated_names_are_deduplicated_in_first_seen_order() -> None:
    """sinter would collect a duplicated decoder twice and write_csv would emit two rows."""
    assert resolve_decoders(["vacuous", "pymatching", "vacuous"]) == ("vacuous", "pymatching")


@requires_mwpf
def test_union_find_is_reported_usable_when_installed() -> None:
    assert check_decoder("hypergraph_union_find").usable
    assert resolve_decoders(["pymatching", "hypergraph_union_find"]) == (
        "pymatching",
        "hypergraph_union_find",
    )
