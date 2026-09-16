"""Read individual members of a remote ZIP archive through HTTP range requests.

The official Willow release is a single 5.7 GB ZIP on Zenodo, and the members this
pipeline needs total about 1.3 MB. Downloading the archive blindly is what the brief
forbids, and ``zipfile`` cannot open an object that is not seekable on disk, so this module
parses the archive's own index — the end-of-central-directory record, its ZIP64
extension (a 5.7 GB archive is necessarily ZIP64) and the central directory — from a
tail range, then fetches exactly the bytes of each member it is asked for.

Every member read is checked twice: the byte count against the central directory's
sizes and the extracted payload's CRC32 against the central directory's CRC. A range
server that silently returns a 200 body, a truncated body or the wrong bytes would
otherwise produce a well-formed file of the wrong contents, which is the failure mode
this repository is organised against. The fetch callable is injected so the parser is
tested offline against archives built in memory.
"""

from __future__ import annotations

import struct
import urllib.request
import zlib
from collections.abc import Callable
from dataclasses import dataclass

FetchRange = Callable[[int, int], bytes]
"""``fetch(start, end)`` returns exactly the inclusive byte range ``[start, end]``."""

_EOCD_SIGNATURE = b"PK\x05\x06"
_ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
_ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
_CENTRAL_SIGNATURE = b"PK\x01\x02"
_LOCAL_SIGNATURE = b"PK\x03\x04"
_ZIP64_EXTRA_ID = 0x0001
# EOCD (22 B) + comment (<= 65535 B) + ZIP64 locator (20 B) + ZIP64 EOCD (56 B + extensible
# data). 128 KiB covers all of it with room for a comment of the maximum length.
_TAIL_BYTES = 128 * 1024
_STORED = 0
_DEFLATED = 8
_USER_AGENT = "qecgen-residual/0.1 (range reader)"


class RemoteZipError(ValueError):
    """The archive bytes, the range server or a member failed a structural check."""


@dataclass(frozen=True)
class ZipMember:
    """One central-directory entry, with ZIP64 sizes/offsets already resolved."""

    name: str
    method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


class RemoteZip:
    """Central-directory parser and member reader over an injected range fetcher."""

    def __init__(self, fetch: FetchRange, size: int) -> None:
        if size <= 0:
            raise RemoteZipError(f"archive size must be positive, got {size}")
        self._fetch = fetch
        self._size = size
        self._members: tuple[ZipMember, ...] | None = None

    @property
    def size(self) -> int:
        return self._size

    def _range(self, start: int, end: int) -> bytes:
        """Fetch ``[start, end]`` and refuse anything but exactly that many bytes.

        A server that ignores the Range header answers 200 with the whole 5.7 GB body; a
        flaky one returns a short body. Both would be parsed as garbage further down, so
        the length is checked at the one place every byte enters.
        """
        if start < 0 or end < start or end >= self._size:
            raise RemoteZipError(f"range [{start}, {end}] outside archive of {self._size} bytes")
        data = self._fetch(start, end)
        expected = end - start + 1
        if len(data) != expected:
            raise RemoteZipError(
                f"range [{start}, {end}] returned {len(data)} bytes, expected {expected} bytes"
            )
        return data

    def members(self) -> tuple[ZipMember, ...]:
        """Parse EOCD, the ZIP64 records when present, and the central directory."""
        if self._members is None:
            self._members = self._parse_central_directory()
        return self._members

    def _parse_central_directory(self) -> tuple[ZipMember, ...]:
        tail_len = min(self._size, _TAIL_BYTES)
        base = self._size - tail_len
        tail = self._range(base, self._size - 1)
        eocd_pos = tail.rfind(_EOCD_SIGNATURE)
        if eocd_pos < 0 or eocd_pos + 22 > len(tail):
            raise RemoteZipError("no end of central directory record in the archive tail")
        (_sig, _disk, _cd_disk, _entries_disk, entries, cd_size, cd_offset, _comment_len) = (
            struct.unpack("<IHHHHIIH", tail[eocd_pos : eocd_pos + 22])
        )
        if entries == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
            entries, cd_size, cd_offset = self._parse_zip64(tail, base, eocd_pos)
        if cd_offset + cd_size > self._size:
            raise RemoteZipError("central directory extends past the end of the archive")
        if cd_offset >= base:
            central = tail[cd_offset - base : cd_offset - base + cd_size]
        else:
            central = self._range(cd_offset, cd_offset + cd_size - 1)
        return _parse_entries(central, entries)

    def _parse_zip64(self, tail: bytes, base: int, eocd_pos: int) -> tuple[int, int, int]:
        locator_pos = tail.rfind(_ZIP64_LOCATOR_SIGNATURE, 0, eocd_pos)
        if locator_pos < 0:
            raise RemoteZipError("EOCD carries ZIP64 sentinels but no ZIP64 locator precedes it")
        (_sig, _disk, z64_offset, _disks) = struct.unpack(
            "<IIQI", tail[locator_pos : locator_pos + 20]
        )
        if z64_offset >= base:
            record = tail[z64_offset - base : z64_offset - base + 56]
        else:
            record = self._range(z64_offset, z64_offset + 55)
        if len(record) < 56 or record[:4] != _ZIP64_EOCD_SIGNATURE:
            raise RemoteZipError("ZIP64 locator does not point at a ZIP64 EOCD record")
        (
            _sig,
            _record_size,
            _made_by,
            _needed,
            _disk,
            _cd_disk,
            _entries_disk,
            entries,
            cd_size,
            cd_offset,
        ) = struct.unpack("<IQHHIIQQQQ", record)
        return entries, cd_size, cd_offset

    def read(self, member: ZipMember) -> bytes:
        """Fetch one member's data via its local header; verify sizes and CRC32.

        The local header's name/extra lengths may differ from the central directory's
        (ZIP64 writers pad the local extra field), so they are read from the local header
        rather than assumed — the central entry only tells us where the header starts.
        """
        if member.method not in (_STORED, _DEFLATED):
            raise RemoteZipError(
                f"{member.name}: compression method {member.method} is not supported "
                "(only stored=0 and deflated=8)"
            )
        offset = member.local_header_offset
        header = self._range(offset, offset + 29)
        if header[:4] != _LOCAL_SIGNATURE:
            raise RemoteZipError(f"{member.name}: no local file header at offset {offset}")
        name_len, extra_len = struct.unpack("<HH", header[26:30])
        start = offset + 30 + name_len + extra_len
        if member.compressed_size == 0:
            data = b""
        else:
            data = self._range(start, start + member.compressed_size - 1)
        if member.method == _DEFLATED:
            try:
                decompressor = zlib.decompressobj(-15)
                payload = decompressor.decompress(data) + decompressor.flush()
            except zlib.error as exc:
                raise RemoteZipError(f"{member.name}: deflate stream is corrupt: {exc}") from exc
            if decompressor.unused_data:
                raise RemoteZipError(f"{member.name}: trailing bytes after the deflate stream")
        else:
            payload = data
        if len(payload) != member.uncompressed_size:
            raise RemoteZipError(
                f"{member.name}: extracted {len(payload)} bytes, central directory says "
                f"{member.uncompressed_size}"
            )
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        if crc != member.crc32:
            raise RemoteZipError(
                f"{member.name}: CRC32 {crc:08x} does not match the central directory's "
                f"{member.crc32:08x}"
            )
        return payload


def _parse_entries(central: bytes, entries: int) -> tuple[ZipMember, ...]:
    members: list[ZipMember] = []
    pos = 0
    for _ in range(entries):
        if pos + 46 > len(central) or central[pos : pos + 4] != _CENTRAL_SIGNATURE:
            raise RemoteZipError(
                f"central directory truncated after {len(members)} of {entries} entries"
            )
        (
            _sig,
            _made_by,
            _needed,
            _flags,
            method,
            _mtime,
            _mdate,
            crc,
            csize,
            usize,
            name_len,
            extra_len,
            comment_len,
            _disk,
            _internal,
            _external,
            lho,
        ) = struct.unpack("<IHHHHHHIIIHHHHHII", central[pos : pos + 46])
        name = central[pos + 46 : pos + 46 + name_len].decode("utf-8")
        extra = central[pos + 46 + name_len : pos + 46 + name_len + extra_len]
        if 0xFFFFFFFF in (usize, csize, lho):
            usize, csize, lho = _resolve_zip64_extra(extra, usize, csize, lho)
        members.append(
            ZipMember(
                name=name,
                method=method,
                crc32=crc,
                compressed_size=csize,
                uncompressed_size=usize,
                local_header_offset=lho,
            )
        )
        pos += 46 + name_len + extra_len + comment_len
    return tuple(members)


def _resolve_zip64_extra(extra: bytes, usize: int, csize: int, lho: int) -> tuple[int, int, int]:
    """Read the ZIP64 extra field; only the fields set to the 0xFFFFFFFF sentinel are present.

    The field's layout is positional and conditional (uncompressed, compressed, offset,
    disk — each present only when the 32-bit value overflowed), so reading three fixed
    quadwords would misassign sizes for a member whose offset alone overflowed.
    """
    pos = 0
    while pos + 4 <= len(extra):
        header_id, header_size = struct.unpack("<HH", extra[pos : pos + 4])
        body = extra[pos + 4 : pos + 4 + header_size]
        if header_id == _ZIP64_EXTRA_ID:
            cursor = 0
            if usize == 0xFFFFFFFF:
                usize = struct.unpack("<Q", body[cursor : cursor + 8])[0]
                cursor += 8
            if csize == 0xFFFFFFFF:
                csize = struct.unpack("<Q", body[cursor : cursor + 8])[0]
                cursor += 8
            if lho == 0xFFFFFFFF:
                lho = struct.unpack("<Q", body[cursor : cursor + 8])[0]
                cursor += 8
            return usize, csize, lho
        pos += 4 + header_size
    raise RemoteZipError("central entry uses ZIP64 sentinels but carries no ZIP64 extra field")


def http_range_fetcher(url: str, timeout: float = 180.0) -> tuple[FetchRange, int]:
    """Return a range fetcher for ``url`` plus the archive's total size.

    The size comes from the ``Content-Range`` header of a one-byte probe request rather
    than from a HEAD, because Zenodo's download endpoint redirects and a HEAD's
    ``Content-Length`` describes whichever hop answered it. Every response must be 206:
    a 200 means the server ignored the Range header and is about to stream the whole
    archive, which is exactly what this reader exists to avoid.

    ``urllib.request.urlopen`` is looked up at call time so tests can monkeypatch it.
    """

    def request(start: int, end: int) -> tuple[bytes, int]:
        req = urllib.request.Request(url)
        req.add_header("Range", f"bytes={start}-{end}")
        req.add_header("User-Agent", _USER_AGENT)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = int(response.status)
            if status != 206:
                raise RemoteZipError(
                    f"range request for bytes {start}-{end} answered {status}, not 206 "
                    "(server ignored the Range header)"
                )
            content_range = response.headers.get("Content-Range")
            body = bytes(response.read())
        if not content_range or "/" not in content_range:
            raise RemoteZipError("206 response without a Content-Range header")
        total_text = content_range.rsplit("/", 1)[1].strip()
        if not total_text.isdigit():
            raise RemoteZipError(f"Content-Range total is not a byte count: {content_range!r}")
        return body, int(total_text)

    _probe, total = request(0, 0)

    def fetch(start: int, end: int) -> bytes:
        body, seen_total = request(start, end)
        if seen_total != total:
            raise RemoteZipError(
                f"archive size changed mid-session: {seen_total} bytes now, {total} at open"
            )
        return body

    return fetch, total
