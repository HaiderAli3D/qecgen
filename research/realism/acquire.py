"""Bounded, resumable research intake; never treat an unverified partial as data.

Run from the repository root with ``python -m research.realism.acquire ASSET_ID``.
The catalogue is tracked; downloaded bytes and acquisition receipts live below the
gitignored data directory. This tool does not import shots into qecgen's data contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Source:
    asset_id: str
    filename: str
    urls: tuple[str, ...]
    source: str
    licence: str
    licence_url: str
    version: str
    max_bytes: int
    publisher_checksum: str | None = None
    expected_bytes: int | None = None
    notes: str = ""
    deferred: bool = False

    def __post_init__(self) -> None:
        for value in (self.asset_id, self.filename):
            if not value or value in {".", ".."} or re.search(r"[^A-Za-z0-9._-]", value):
                raise ValueError("Asset IDs and filenames must be plain portable basenames")
        if not self.urls or any(not u.startswith(("https://", "http://")) for u in self.urls):
            raise ValueError("Every asset requires an HTTP(S) source URL")
        if self.max_bytes <= 0 or (
            self.expected_bytes is not None and not 0 < self.expected_bytes <= self.max_bytes
        ):
            raise ValueError("Expected size must fit within the positive asset budget")
        if self.publisher_checksum is not None:
            algorithm, separator, digest = self.publisher_checksum.partition(":")
            widths = {"md5": 32, "sha256": 64}
            if (
                not separator
                or algorithm not in widths
                or len(digest) != widths[algorithm]
                or re.fullmatch("[0-9a-f]+", digest) is None
            ):
                raise ValueError("Publisher checksum must be md5:<hex> or sha256:<hex>")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def hashes(path: Path) -> dict[str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            sha256.update(block)
            md5.update(block)
    return {"sha256": sha256.hexdigest(), "md5": md5.hexdigest()}


def tree_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


@contextmanager
def intake_lock(root: Path) -> Iterator[None]:
    """Serialise receipts and the cumulative byte counter across local processes."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".acquire.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            lock.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def charged_bytes(root: Path) -> int:
    """Include discarded retries and reserved reads whose byte count became unknowable."""
    ledger = root / "intake/transfer-budget.json"
    if ledger.exists():
        return int(json.loads(ledger.read_text(encoding="utf-8"))["charged_bytes"])
    observed = 0
    for path in (root / "intake").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "source" in record:
            observed += sum(
                attempt.get("received_bytes", 0) for attempt in record.get("attempts", [])
            )
        elif "assets" in record:
            observed += sum(
                asset.get("bytes", 0)
                for asset in record["assets"]
                if asset.get("status") == "verified" and "source_url" in asset
            )
    return max(observed, tree_bytes(root / "raw"))


def verify(path: Path, source: Source) -> dict[str, str]:
    size = path.stat().st_size
    if size > source.max_bytes or (
        source.expected_bytes is not None and size != source.expected_bytes
    ):
        raise ValueError(f"Size mismatch: got {size}, expected {source.expected_bytes}")
    if size == 0:
        raise ValueError("An empty response is not an acquired asset")
    result = hashes(path)
    if source.publisher_checksum is not None:
        algorithm, expected = source.publisher_checksum.split(":", 1)
        if result[algorithm] != expected:
            raise ValueError(f"Publisher {algorithm} checksum mismatch")
    return result


def catalogue(path: Path) -> dict[str, Source]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, Source] = {}
    for item in raw["assets"]:
        source = Source(**{**item, "urls": tuple(item["urls"])})
        if source.asset_id in result:
            raise ValueError(f"Duplicate catalogue asset: {source.asset_id}")
        result[source.asset_id] = source
    return result


def acquire(
    source: Source,
    root: Path,
    *,
    attempts: int = 2,
    timeout: float = 25,
    download_budget: int = 10_000_000_000,
    working_budget: int = 30_000_000_000,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    with intake_lock(root):
        return _acquire_locked(
            source,
            root,
            attempts=attempts,
            timeout=timeout,
            download_budget=download_budget,
            working_budget=working_budget,
            opener=opener,
        )


def _acquire_locked(
    source: Source,
    root: Path,
    *,
    attempts: int,
    timeout: float,
    download_budget: int,
    working_budget: int,
    opener: Callable[..., Any],
) -> dict[str, Any]:
    """Return a persisted receipt; failed checks never publish the final filename.

    A server may ignore Range. In that case the partial is truncated before writing,
    never appended to a second copy. A changed catalogue cannot reuse old partials.
    The storage limits include all existing assets, including separately acquired ones.
    """
    if attempts < 1 or timeout <= 0:
        raise ValueError("Attempts and timeout must be positive")
    root = root.resolve()
    asset_dir = root / "raw" / source.asset_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    target = asset_dir / source.filename
    partial = asset_dir / (source.filename + ".part")
    receipt_path = root / "intake" / (source.asset_id + ".json")
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "source": asdict(source),
        "started_at": utc_now(),
        "status": "pending",
        "path": str(target),
        "attempts": [],
    }
    identity = json.dumps(asdict(source), sort_keys=True)
    previous: dict[str, Any] = {}
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text(encoding="utf-8"))
        if json.dumps(previous["source"], sort_keys=True) != identity:
            raise ValueError("Existing acquisition receipt belongs to a different source")
        receipt["attempts"] = previous.get("attempts", [])
        receipt["restart_partial"] = previous.get("restart_partial", False)
        if "hashes" in previous:
            receipt["hashes"] = previous["hashes"]
    elif partial.exists():
        raise ValueError("Partial has no acquisition receipt; provenance cannot be established")
    write_json(receipt_path, receipt)
    try:
        if target.exists():
            result = verify(target, source)
            previous_hash = previous.get("hashes", {}).get("sha256")
            if previous_hash is not None and result["sha256"] != previous_hash:
                raise ValueError("Cached asset differs from the acquisition receipt SHA-256")
            if previous_hash is None and source.publisher_checksum is None:
                raise ValueError(
                    "Existing file has neither a publisher checksum nor a receipt hash"
                )
            receipt.update(status="verified", hashes=result, bytes=target.stat().st_size)
            receipt["verified_at"] = utc_now()
            write_json(receipt_path, receipt)
            return receipt
        if partial.exists() and source.publisher_checksum is not None:
            # A process may stop after its last byte but before the rename. Google
            # publishes a checksum but only rounded sizes; Range at EOF returns 416.
            # A matching publisher hash proves completeness without another request.
            try:
                partial_result = verify(partial, source)
            except ValueError:
                pass  # A partial prefix normally cannot match the complete-object hash.
            else:
                os.replace(partial, target)
                receipt.update(
                    status="verified",
                    hashes=partial_result,
                    bytes=target.stat().st_size,
                    verified_at=utc_now(),
                    restart_partial=False,
                )
                write_json(receipt_path, receipt)
                return receipt
        for attempt_index in range(attempts):
            url = source.urls[attempt_index % len(source.urls)]
            offset = partial.stat().st_size if partial.exists() else 0
            if receipt.get("restart_partial") or (
                source.expected_bytes is not None and offset >= source.expected_bytes
            ):
                offset = 0
            attempt: dict[str, Any] = {
                "url": url,
                "started_at": utc_now(),
                "resume_from": offset,
                "received_bytes": 0,
            }
            receipt["attempts"].append(attempt)
            write_json(receipt_path, receipt)
            try:
                request_headers = {
                    "User-Agent": "qecgen-research-intake/1",
                    "Accept-Encoding": "identity",
                }
                if offset:
                    request_headers["Range"] = f"bytes={offset}-"
                request = urllib.request.Request(url, headers=request_headers)
                with opener(request, timeout=timeout) as response:
                    attempt["http_status"] = response.status
                    # CDN redirects can carry short-lived signatures; the immutable
                    # source URL, not those tokens, is the reproducibility identifier.
                    resolved = urllib.parse.urlsplit(response.geturl())
                    attempt["resolved_url"] = urllib.parse.urlunsplit(
                        (resolved.scheme, resolved.netloc, resolved.path, "", "")
                    )
                    attempt["etag"] = response.headers.get("ETag")
                    attempt["last_modified"] = response.headers.get("Last-Modified")
                    full_size: int | None
                    if response.status == 206:
                        match = re.fullmatch(
                            r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                        )
                        if match is None or int(match[1]) != offset:
                            raise ValueError("Server returned an inconsistent resumed byte range")
                        full_size = int(match[3])
                        if int(match[2]) < offset or int(match[2]) >= full_size:
                            raise ValueError("Server returned an invalid byte range")
                    elif response.status == 200:
                        offset = 0
                        raw_length = response.headers.get("Content-Length")
                        full_size = int(raw_length) if raw_length is not None else None
                    else:
                        raise ValueError(f"Unexpected HTTP status {response.status}")
                    if full_size is not None and (
                        full_size > source.max_bytes
                        or (
                            source.expected_bytes is not None and full_size != source.expected_bytes
                        )
                    ):
                        raise ValueError("HTTP object size conflicts with catalogue budget or size")
                    work_usage = tree_bytes(root)
                    transfer_usage = charged_bytes(root)
                    # Existing partial bytes are already included; a 200 replaces them.
                    if not offset and partial.exists():
                        work_usage -= partial.stat().st_size
                    with partial.open("ab" if offset else "wb") as destination:
                        if not offset:
                            # Only the rejected prefix required restarting. If this
                            # fresh transfer interrupts, its new prefix is resumable.
                            receipt["restart_partial"] = False
                            write_json(receipt_path, receipt)
                        while True:
                            remaining = download_budget - transfer_usage
                            if remaining <= 0:
                                known_size = full_size or source.expected_bytes
                                if known_size is not None and destination.tell() == known_size:
                                    break
                                raise ValueError("Cumulative download byte budget exceeded")
                            requested = min(4 * 1024 * 1024, remaining)
                            # A timed-out buffered read can have received bytes without
                            # returning them. Charge its reservation before the read;
                            # an exception or process death must not erase that cost.
                            write_json(
                                root / "intake/transfer-budget.json",
                                {
                                    "charged_bytes": transfer_usage + requested,
                                    "accounting": "Includes interrupted read reservations",
                                    "updated_at": utc_now(),
                                },
                            )
                            block = response.read(requested)
                            transfer_usage += len(block)
                            write_json(
                                root / "intake/transfer-budget.json",
                                {"charged_bytes": transfer_usage, "updated_at": utc_now()},
                            )
                            if not block:
                                break
                            attempt["received_bytes"] += len(block)
                            write_json(receipt_path, receipt)
                            next_size = destination.tell() + len(block)
                            if next_size > source.max_bytes:
                                raise ValueError("Asset exceeded its byte budget")
                            if work_usage + len(block) > working_budget:
                                raise ValueError("Working-data byte budget exceeded")
                            if shutil.disk_usage(root).free < len(block) + 64 * 1024 * 1024:
                                raise ValueError("Insufficient disk space for bounded download")
                            destination.write(block)
                            work_usage += len(block)
                        destination.flush()
                        os.fsync(destination.fileno())
                    if full_size is not None and partial.stat().st_size != full_size:
                        raise ValueError("HTTP body ended before the declared object size")
                try:
                    result = verify(partial, source)
                except ValueError:
                    receipt["restart_partial"] = True
                    attempt["invalid_hashes"] = hashes(partial)
                    raise
                os.replace(partial, target)
                attempt.update(status="verified", finished_at=utc_now())
                receipt.update(
                    status="verified",
                    hashes=result,
                    bytes=target.stat().st_size,
                    verified_at=utc_now(),
                    restart_partial=False,
                )
                write_json(receipt_path, receipt)
                return receipt
            except Exception as error:
                if isinstance(error, urllib.error.HTTPError) and error.code == 416:
                    receipt["restart_partial"] = True
                    if partial.exists():
                        attempt["rejected_range_hashes"] = hashes(partial)
                attempt.update(status="failed", finished_at=utc_now(), error=str(error))
                receipt["status"] = "failed"
                receipt["partial_bytes"] = partial.stat().st_size if partial.exists() else 0
                write_json(receipt_path, receipt)
                if attempt_index + 1 < attempts:
                    time.sleep(1)
        return receipt
    except Exception as error:
        receipt.update(status="failed", error=str(error), finished_at=utc_now())
        write_json(receipt_path, receipt)
        return receipt


def zip_inventory(path: Path, *, max_uncompressed: int = 30_000_000_000) -> dict[str, Any]:
    """Inspect ZIP metadata without extraction; a large inventory is not permission to unpack it."""
    with zipfile.ZipFile(path) as archive:
        members: list[dict[str, Any]] = [
            {"name": member.filename, "bytes": member.file_size, "compressed": member.compress_size}
            for member in archive.infolist()
        ]
    total = sum(member["bytes"] for member in members)
    return {
        "archive": str(path.resolve()),
        "sha256": hashes(path)["sha256"],
        "members": members,
        "uncompressed_bytes": total,
        "within_extraction_budget": total <= max_uncompressed,
        "extracted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset_ids", nargs="+")
    parser.add_argument("--catalogue", type=Path, default=Path(__file__).with_name("sources.json"))
    parser.add_argument("--root", type=Path, default=Path("data/realism"))
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=25)
    args = parser.parse_args()
    sources = catalogue(args.catalogue)
    failed = False
    for asset_id in args.asset_ids:
        source = sources[asset_id]
        if source.deferred:
            parser.error(f"{asset_id} is deferred; revise the reviewed catalogue before fetching")
        receipt = acquire(source, args.root, attempts=args.attempts, timeout=args.timeout)
        print(
            json.dumps(
                {
                    key: receipt[key]
                    for key in ("status", "path", "bytes", "hashes", "error")
                    if key in receipt
                }
            ),
            flush=True,
        )
        failed |= receipt["status"] != "verified"
        if receipt["status"] == "verified" and source.filename.endswith(".zip"):
            inventory = zip_inventory(Path(receipt["path"]))
            write_json(args.root / "intake" / (asset_id + ".inventory.json"), inventory)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
