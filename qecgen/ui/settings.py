"""Server configuration, resolved once at startup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qecgen.ui import DEFAULT_PORT

__all__ = ["LOOPBACK_HOSTS", "WebSettings"]

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})
"""Addresses the server will bind to.

Not a configuration choice. This API writes files to disk and spawns processes at the
request of whoever can reach it, with no authentication; the only safe audience is the
person sitting at the machine.

IPv4 loopback only, and deliberately so. ``::1`` used to be listed, but Starlette's
TrustedHost middleware parses the Host header with ``host.split(":")[0]``, which can
never match an IPv6 literal — ``[::1]:8765`` parses to ``"["`` — so binding ``::1``
served a UI that answered every browser request with 400. Refusing the bind up front is
honest; allowlisting cannot be made to work without replacing the middleware.
"""


@dataclass(frozen=True, slots=True)
class WebSettings:
    """Everything the app needs that is not a request."""

    data_root: Path
    """The only directory jobs may write to and the browser may read from."""

    runs_dir: Path
    """Where durable run records live, so history survives a restart."""

    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    max_concurrent_jobs: int = 1

    dev: bool = False
    """Allow the Vite dev server's origin. Off unless asked for: it is the one setting
    that opens the API to a page this server did not serve."""

    @classmethod
    def create(
        cls,
        data_root: Path,
        *,
        runs_dir: Path | None = None,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        max_concurrent_jobs: int = 1,
        dev: bool = False,
    ) -> WebSettings:
        """Resolve paths and create the directories the server relies on."""
        root = data_root.resolve()
        runs = (runs_dir or root / "runs").resolve()
        root.mkdir(parents=True, exist_ok=True)
        runs.mkdir(parents=True, exist_ok=True)
        return cls(
            data_root=root,
            runs_dir=runs,
            host=host,
            port=port,
            max_concurrent_jobs=max_concurrent_jobs,
            dev=dev,
        )
