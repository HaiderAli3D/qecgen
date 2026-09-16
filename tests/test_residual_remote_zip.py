"""Offline tests for the range-request ZIP reader and the Willow Zenodo member fetch.

Nothing here touches the network: archives are built in memory with ``zipfile`` and served
through a slicing closure, ``urllib`` is monkeypatched, and the Zenodo record JSON comes
from an injected callable. The ZIP64 central-directory path is exercised by rewriting a
small archive's central directory into ZIP64 form, because ``zipfile`` only emits ZIP64
central records for archives that are actually larger than 4 GiB.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import urllib.error
import zipfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import stim

from qecgen.hardware import ImportedCohort
from qecgen.residual import remote_zip
from qecgen.residual.remote_zip import RemoteZip, RemoteZipError, http_range_fetcher
from qecgen.residual.zenodo import (
    WILLOW_COHORT_MEMBERS,
    WillowSourceBlockedError,
    fetch_cohort,
    verify_cohort,
)

# --------------------------------------------------------------------------- helpers


def _serve(data: bytes) -> remote_zip.FetchRange:
    def fetch(start: int, end: int) -> bytes:
        assert 0 <= start <= end < len(data), (start, end, len(data))
        return data[start : end + 1]

    return fetch


def _build_zip(members: dict[str, tuple[bytes, int]], force_zip64: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", allowZip64=True) as archive:
        for name, (payload, method) in members.items():
            if name == force_zip64:
                with archive.open(name, "w", force_zip64=True) as handle:
                    handle.write(payload)
            else:
                archive.writestr(name, payload, compress_type=method)
    return buffer.getvalue()


def _zip64_rewrite(data: bytes) -> bytes:
    """Rewrite a small archive's central directory and EOCD into ZIP64 form.

    Local headers and data are untouched; every central entry gets a ZIP64 extra field
    carrying the sizes and offset, and the EOCD is replaced by a ZIP64 EOCD record, a
    ZIP64 locator and an EOCD whose counts and offsets are the 0xFFFF sentinels.
    """
    eocd_pos = data.rfind(b"PK\x05\x06")
    assert eocd_pos >= 0
    (_sig, _dn, _dcd, _ed, entries, cd_size, cd_offset, _cl) = struct.unpack(
        "<IHHHHIIH", data[eocd_pos : eocd_pos + 22]
    )
    central = data[cd_offset : cd_offset + cd_size]
    rebuilt = bytearray()
    pos = 0
    for _ in range(entries):
        fields = list(struct.unpack("<IHHHHHHIIIHHHHHII", central[pos : pos + 46]))
        nlen, elen, clen = fields[10], fields[11], fields[12]
        name = central[pos + 46 : pos + 46 + nlen]
        extra = central[pos + 46 + nlen : pos + 46 + nlen + elen]
        comment = central[pos + 46 + nlen + elen : pos + 46 + nlen + elen + clen]
        csize, usize, lho = fields[8], fields[9], fields[16]
        zip64_extra = struct.pack("<HHQQQ", 1, 24, usize, csize, lho)
        fields[8] = fields[9] = fields[16] = 0xFFFFFFFF
        fields[11] = len(extra) + len(zip64_extra)
        rebuilt += struct.pack("<IHHHHHHIIIHHHHHII", *fields) + name + extra + zip64_extra + comment
        pos += 46 + nlen + elen + clen
    body = data[:cd_offset] + bytes(rebuilt)
    z64_offset = len(body)
    z64 = struct.pack(
        "<IQHHIIQQQQ", 0x06064B50, 44, 45, 45, 0, 0, entries, entries, len(rebuilt), cd_offset
    )
    locator = struct.pack("<IIQI", 0x07064B50, 0, z64_offset, 1)
    eocd = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0)
    return body + z64 + locator + eocd


@pytest.fixture
def sample_members() -> dict[str, tuple[bytes, int]]:
    rng = np.random.default_rng(7)
    big = rng.integers(0, 256, size=70_000, dtype=np.uint8).tobytes()  # incompressible, > 64 KiB
    return {
        "root/a.txt": (b"alpha\n" * 10, zipfile.ZIP_STORED),
        "root/b.bin": (big, zipfile.ZIP_STORED),
        "root/c.txt": (b"charlie " * 5000, zipfile.ZIP_DEFLATED),
        "root/sub/d.json": (json.dumps({"k": list(range(200))}).encode(), zipfile.ZIP_DEFLATED),
    }


# --------------------------------------------------------------------------- RemoteZip


def test_members_match_zipfile(sample_members: dict[str, tuple[bytes, int]]) -> None:
    data = _build_zip(sample_members)
    reader = RemoteZip(_serve(data), len(data))
    members = reader.members()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
    assert [m.name for m in members] == [i.filename for i in infos]
    for member, info in zip(members, infos, strict=True):
        assert member.crc32 == info.CRC
        assert member.compressed_size == info.compress_size
        assert member.uncompressed_size == info.file_size
        assert member.method == info.compress_type
        assert member.local_header_offset == info.header_offset


def test_read_returns_exact_bytes(sample_members: dict[str, tuple[bytes, int]]) -> None:
    data = _build_zip(sample_members)
    reader = RemoteZip(_serve(data), len(data))
    for member in reader.members():
        assert reader.read(member) == sample_members[member.name][0]


def test_corrupted_byte_fails_crc(sample_members: dict[str, tuple[bytes, int]]) -> None:
    data = bytearray(_build_zip(sample_members))
    reader = RemoteZip(_serve(bytes(data)), len(data))
    member = next(m for m in reader.members() if m.name == "root/b.bin")
    header = bytes(data[member.local_header_offset : member.local_header_offset + 30])
    nlen, elen = struct.unpack("<HH", header[26:30])
    start = member.local_header_offset + 30 + nlen + elen
    data[start + 123] ^= 0x01
    corrupted = RemoteZip(_serve(bytes(data)), len(data))
    with pytest.raises(RemoteZipError, match="CRC32"):
        corrupted.read(member)


def test_zip64_central_directory(sample_members: dict[str, tuple[bytes, int]]) -> None:
    data = _zip64_rewrite(_build_zip(sample_members, force_zip64="root/c.txt"))
    # zipfile must still accept the rewritten archive, or the test is checking a forgery.
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert archive.testzip() is None
        infos = {i.filename: i for i in archive.infolist()}
    reader = RemoteZip(_serve(data), len(data))
    members = reader.members()
    assert len(members) == len(sample_members)
    for member in members:
        info = infos[member.name]
        assert (member.compressed_size, member.uncompressed_size, member.local_header_offset) == (
            info.compress_size,
            info.file_size,
            info.header_offset,
        )
        assert reader.read(member) == sample_members[member.name][0]


def test_unsupported_method_refused(sample_members: dict[str, tuple[bytes, int]]) -> None:
    data = _build_zip(sample_members)
    reader = RemoteZip(_serve(data), len(data))
    member = reader.members()[0]
    bogus = remote_zip.ZipMember(
        name=member.name,
        method=12,
        crc32=member.crc32,
        compressed_size=member.compressed_size,
        uncompressed_size=member.uncompressed_size,
        local_header_offset=member.local_header_offset,
    )
    with pytest.raises(RemoteZipError, match="method"):
        reader.read(bogus)


def test_not_a_zip_refused() -> None:
    data = b"not a zip archive at all" * 100
    with pytest.raises(RemoteZipError, match="end of central directory"):
        RemoteZip(_serve(data), len(data)).members()


def test_short_range_response_refused(sample_members: dict[str, tuple[bytes, int]]) -> None:
    data = _build_zip(sample_members)

    def truncating(start: int, end: int) -> bytes:
        return data[start:end]  # one byte short

    with pytest.raises(RemoteZipError, match="bytes"):
        RemoteZip(truncating, len(data)).members()


# --------------------------------------------------------------------------- http fetcher


class _FakeResponse:
    def __init__(self, status: int, body: bytes, headers: dict[str, str]) -> None:
        self.status = status
        self._body = body
        self.headers = headers

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _install_fake_urlopen(
    monkeypatch: pytest.MonkeyPatch, data: bytes, *, status: int = 206
) -> list[str]:
    ranges: list[str] = []

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeResponse:
        header = request.get_header("Range")
        assert header is not None
        ranges.append(header)
        start, end = (int(v) for v in header.removeprefix("bytes=").split("-"))
        body = data[start : end + 1]
        if status != 206:
            return _FakeResponse(status, data, {"Content-Length": str(len(data))})
        return _FakeResponse(
            206,
            body,
            {"Content-Range": f"bytes {start}-{start + len(body) - 1}/{len(data)}"},
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return ranges


def test_http_range_fetcher_reads_archive(
    monkeypatch: pytest.MonkeyPatch, sample_members: dict[str, tuple[bytes, int]]
) -> None:
    data = _build_zip(sample_members)
    ranges = _install_fake_urlopen(monkeypatch, data)
    fetch, size = http_range_fetcher("https://example.invalid/archive.zip")
    assert size == len(data)
    assert ranges == ["bytes=0-0"]
    reader = RemoteZip(fetch, size)
    for member in reader.members():
        assert reader.read(member) == sample_members[member.name][0]
    assert all(r.startswith("bytes=") for r in ranges)


def test_http_range_fetcher_refuses_200(
    monkeypatch: pytest.MonkeyPatch, sample_members: dict[str, tuple[bytes, int]]
) -> None:
    data = _build_zip(sample_members)
    _install_fake_urlopen(monkeypatch, data, status=200)
    with pytest.raises(RemoteZipError, match="206"):
        http_range_fetcher("https://example.invalid/archive.zip")


# --------------------------------------------------------------------------- Willow cohort

ROOT = "google_105Q_surface_code_d3_d5_d7"
ARCHIVE = f"{ROOT}.zip"
PREFIX = f"{ROOT}/d3_at_q10_7/Z/r10/"
PATHWAYS = (
    "correlated_matching_decoder_with_si1000_prior",
    "correlated_matching_decoder_with_rl_optimized_prior",
)


def _qubit_partition(circuit: stim.Circuit) -> tuple[list[list[float]], list[list[float]]]:
    counts: dict[int, int] = {}
    resetting: set[int] = set()
    for instruction in circuit.flattened():
        if instruction.name in ("M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"):
            for target in instruction.targets_copy():
                counts[target.value] = counts.get(target.value, 0) + 1
                if instruction.name.startswith("MR"):
                    resetting.add(target.value)
    coords = circuit.get_final_qubit_coordinates()
    data = [coords[q] for q, n in sorted(counts.items()) if n == 1 and q not in resetting]
    meas = [coords[q] for q, n in sorted(counts.items()) if n > 1 or q in resetting]
    return data, meas


def _synthetic_cohort(tmp_path: Path) -> tuple[dict[str, bytes], ImportedCohort, dict[str, Any]]:
    """Four rows on a d=3 memory-Z circuit with b8 bytes packed little-endian."""
    ideal = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=2)
    noisy = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=2,
        after_clifford_depolarization=0.01,
    )
    n_det = ideal.num_detectors
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, size=(4, n_det), dtype=np.uint8).astype(bool)
    bits[0, :] = False
    bits[1, 0] = True  # asymmetric byte: 0b0000_0001 little-endian, 0b1000_0000 big-endian
    dets = np.packbits(bits, axis=1, bitorder="little")
    truth = np.array([[0], [1], [1], [0]], dtype=np.uint8)
    obs = np.packbits(truth.astype(bool), axis=1, bitorder="little")
    circuit_bytes = str(ideal).encode()
    circuit_path = tmp_path / "mirror.stim"
    circuit_path.write_bytes(circuit_bytes)
    data_coords, meas_coords = _qubit_partition(ideal)
    metadata = {
        "basis": "Z",
        "rounds": 2,
        "shots": 4,
        "distance": 3,
        "data_qubit_coords": data_coords,
        "meas_qubit_coords": meas_coords,
    }
    dem = noisy.detector_error_model(decompose_errors=True)
    predicted = np.packbits(np.array([[0], [1], [0], [0]], dtype=bool), axis=1, bitorder="little")
    files = {
        "circuit_ideal.stim": circuit_bytes,
        "circuit_noisy_si1000.stim": str(noisy).encode(),
        "detection_events.b8": dets.tobytes(),
        "obs_flips_actual.b8": obs.tobytes(),
        "sweep_bits.b8": bytes(4),
        "metadata.json": json.dumps(metadata).encode(),
    }
    for pathway in PATHWAYS:
        files[f"decoding_results/{pathway}/error_model.dem"] = str(dem).encode()
        files[f"decoding_results/{pathway}/obs_flips_predicted.b8"] = predicted.tobytes()
    expected = {
        "distance": 3,
        "basis": "Z",
        "rounds": 2,
        "orientation": "q10_7",
        "circuit_sha256": hashlib.sha256(circuit_bytes).hexdigest(),
        "table_sha256": "0" * 64,
    }
    cohort = ImportedCohort(
        detectors=dets,
        observables=obs,
        ideal=ideal,
        source={**expected, "circuit_path": str(circuit_path), "shots": 4},
    )
    return files, cohort, expected


def _write_cache(cache_dir: Path, files: dict[str, bytes]) -> None:
    for name, payload in files.items():
        target = cache_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)


def test_verify_cohort_accepts_consistent_cache(tmp_path: Path) -> None:
    files, cohort, expected = _synthetic_cohort(tmp_path)
    cache = tmp_path / "cache"
    _write_cache(cache, files)
    report = verify_cohort(cache, cohort, expected["circuit_sha256"], expected)
    assert report["circuit_bytes_equal_mirror"] is True
    assert report["detectors_equal_cohort"] is True
    assert report["observables_equal_cohort"] is True
    assert report["metadata"]["shots"] == 4
    assert report["qubit_partition_matches_metadata"] is True
    assert report["noisy_circuit"]["without_noise_equals_ideal"] is True
    pathway = report["google_decoders"]["correlated_matching_decoder_with_si1000_prior"]
    assert pathway["failures"] == 1
    assert pathway["error_rate"] == pytest.approx(0.25)
    assert pathway["ci95"][0] <= 0.25 <= pathway["ci95"][1]


def test_verify_cohort_reports_agreement_with_supplied_guess(tmp_path: Path) -> None:
    files, cohort, expected = _synthetic_cohort(tmp_path)
    cache = tmp_path / "cache"
    _write_cache(cache, files)
    guess = np.array([0, 1, 1, 0], dtype=np.uint8)
    report = verify_cohort(cache, cohort, expected["circuit_sha256"], expected, pm_guess=guess)
    pathway = report["google_decoders"]["correlated_matching_decoder_with_si1000_prior"]
    assert pathway["agreement_with_pm_guess"] == pytest.approx(0.75)


def test_verify_cohort_rejects_detector_mismatch(tmp_path: Path) -> None:
    files, cohort, expected = _synthetic_cohort(tmp_path)
    tampered = bytearray(files["detection_events.b8"])
    tampered[0] ^= 0x02
    files["detection_events.b8"] = bytes(tampered)
    cache = tmp_path / "cache"
    _write_cache(cache, files)
    with pytest.raises(ValueError, match="detection_events"):
        verify_cohort(cache, cohort, expected["circuit_sha256"], expected)


def test_verify_cohort_rejects_big_endian_b8(tmp_path: Path) -> None:
    files, cohort, expected = _synthetic_cohort(tmp_path)
    from qecgen.sampling import unpack_bits

    bits = unpack_bits(cohort.detectors, cohort.ideal.num_detectors)
    files["detection_events.b8"] = np.packbits(bits, axis=1, bitorder="big").tobytes()
    cache = tmp_path / "cache"
    _write_cache(cache, files)
    with pytest.raises(ValueError, match="detection_events"):
        verify_cohort(cache, cohort, expected["circuit_sha256"], expected)


def test_verify_cohort_rejects_metadata_disagreement(tmp_path: Path) -> None:
    files, cohort, expected = _synthetic_cohort(tmp_path)
    metadata = json.loads(files["metadata.json"])
    metadata["rounds"] = 3
    files["metadata.json"] = json.dumps(metadata).encode()
    cache = tmp_path / "cache"
    _write_cache(cache, files)
    with pytest.raises(ValueError, match="rounds"):
        verify_cohort(cache, cohort, expected["circuit_sha256"], expected)


def test_verify_cohort_rejects_qubit_coordinate_disagreement(tmp_path: Path) -> None:
    files, cohort, expected = _synthetic_cohort(tmp_path)
    metadata = json.loads(files["metadata.json"])
    moved = metadata["data_qubit_coords"].pop()
    metadata["meas_qubit_coords"].append(moved)
    files["metadata.json"] = json.dumps(metadata).encode()
    cache = tmp_path / "cache"
    _write_cache(cache, files)
    with pytest.raises(ValueError, match="qubit_coords"):
        verify_cohort(cache, cohort, expected["circuit_sha256"], expected)


def test_verify_cohort_rejects_circuit_disagreement(tmp_path: Path) -> None:
    files, cohort, expected = _synthetic_cohort(tmp_path)
    files["circuit_ideal.stim"] = files["circuit_ideal.stim"] + b"\n# trailing comment\n"
    cache = tmp_path / "cache"
    _write_cache(cache, files)
    with pytest.raises(ValueError, match="circuit_ideal"):
        verify_cohort(cache, cohort, expected["circuit_sha256"], expected)


def _cohort_archive(files: dict[str, bytes]) -> bytes:
    members = {PREFIX + name: (payload, zipfile.ZIP_DEFLATED) for name, payload in files.items()}
    members[f"{ROOT}/README/README.md"] = (b"# readme\n", zipfile.ZIP_DEFLATED)
    members[f"{ROOT}/README/patches.png"] = (b"\x89PNG", zipfile.ZIP_STORED)
    members[f"{ROOT}/d3_at_q10_7/Z/r10/measurements.b8"] = (bytes(100), zipfile.ZIP_DEFLATED)
    members[f"{ROOT}/d5_at_q10_7/Z/r10/metadata.json"] = (b"{}", zipfile.ZIP_DEFLATED)
    return _build_zip(members)


def _record_json(archive_size: int, archive: str = ARCHIVE) -> bytes:
    record = {
        "id": 13273331,
        "doi": "10.5281/zenodo.13273331",
        "created": "2024-08-08T00:00:00+00:00",
        "metadata": {
            "title": "Data for Quantum error correction below the surface code threshold",
            "version": "1.0.0",
            "publication_date": "2024-08-08",
            "license": {"id": "cc-by-4.0"},
        },
        "files": [
            {
                "key": archive,
                "size": archive_size,
                "checksum": "md5:21fa6ad35b395d838ebcdbc92e364a12",
                "links": {
                    "self": f"https://zenodo.org/api/records/13273331/files/{archive}/content"
                },
            }
        ],
    }
    return json.dumps(record).encode()


def _fetch_kwargs(files: dict[str, bytes]) -> dict[str, Any]:
    data = _cohort_archive(files)
    calls: dict[str, list[str]] = {"urls": [], "archives": []}

    def fetch_url(url: str) -> bytes:
        calls["urls"].append(url)
        return _record_json(len(data))

    def open_archive(url: str) -> tuple[remote_zip.FetchRange, int]:
        calls["archives"].append(url)
        return _serve(data), len(data)

    return {"fetch_url": fetch_url, "open_archive": open_archive, "_calls": calls, "_data": data}


def test_fetch_cohort_writes_members_and_receipts(tmp_path: Path) -> None:
    files, _cohort, _expected = _synthetic_cohort(tmp_path)
    kwargs = _fetch_kwargs(files)
    cache = tmp_path / "zenodo"
    receipts = fetch_cohort(
        13273331,
        ARCHIVE,
        PREFIX,
        cache,
        fetch_url=kwargs["fetch_url"],
        open_archive=kwargs["open_archive"],
        archive_md5_published="21fa6ad35b395d838ebcdbc92e364a12",
    )
    assert kwargs["_calls"]["urls"] == ["https://zenodo.org/api/records/13273331"]
    for name in WILLOW_COHORT_MEMBERS:
        assert (cache / name).read_bytes() == files[name]
    assert (cache / "README" / "README.md").read_bytes() == b"# readme\n"
    assert not (cache / "measurements.b8").exists()
    on_disk = json.loads((cache / "receipts.json").read_text(encoding="utf-8"))
    assert on_disk == receipts
    assert receipts["record"]["version"] == "1.0.0"
    assert receipts["record"]["doi"] == "10.5281/zenodo.13273331"
    assert receipts["archive"]["size"] == len(kwargs["_data"])
    assert receipts["archive"]["md5_published"] == "21fa6ad35b395d838ebcdbc92e364a12"
    assert receipts["archive"]["md5_status"] == "not_reverified"
    by_member = {m["member"]: m for m in receipts["members"]}
    dem_member = (
        PREFIX + "decoding_results/correlated_matching_decoder_with_si1000_prior/error_model.dem"
    )
    entry = by_member[dem_member]
    payload = files[
        "decoding_results/correlated_matching_decoder_with_si1000_prior/error_model.dem"
    ]
    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["crc32"] == f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"
    assert entry["uncompressed_size"] == len(payload)
    assert entry["compression_method"] == 8
    assert entry["status"] == "fetched"
    assert entry["fetched_at"].endswith("+00:00") and len(entry["fetched_at"]) == 25


def test_fetch_cohort_reuses_intact_cache_and_refetches_damaged(tmp_path: Path) -> None:
    files, _cohort, _expected = _synthetic_cohort(tmp_path)
    kwargs = _fetch_kwargs(files)
    cache = tmp_path / "zenodo"
    first = fetch_cohort(
        13273331,
        ARCHIVE,
        PREFIX,
        cache,
        fetch_url=kwargs["fetch_url"],
        open_archive=kwargs["open_archive"],
    )
    (cache / "metadata.json").write_bytes(b'{"damaged": true}')
    second = fetch_cohort(
        13273331,
        ARCHIVE,
        PREFIX,
        cache,
        fetch_url=kwargs["fetch_url"],
        open_archive=kwargs["open_archive"],
    )
    status = {m["member"]: m["status"] for m in second["members"]}
    assert status[PREFIX + "metadata.json"] == "fetched"
    assert status[PREFIX + "circuit_ideal.stim"] == "cached"
    assert (cache / "metadata.json").read_bytes() == files["metadata.json"]
    first_at = {m["member"]: m["fetched_at"] for m in first["members"]}
    second_at = {m["member"]: m["fetched_at"] for m in second["members"]}
    assert second_at[PREFIX + "circuit_ideal.stim"] == first_at[PREFIX + "circuit_ideal.stim"]


def test_fetch_cohort_network_failure_is_blocked(tmp_path: Path) -> None:
    def failing(url: str) -> bytes:
        raise urllib.error.URLError("no route to host")

    with pytest.raises(WillowSourceBlockedError, match="no route to host"):
        fetch_cohort(13273331, ARCHIVE, PREFIX, tmp_path / "zenodo", fetch_url=failing)
    assert not (tmp_path / "zenodo" / "receipts.json").exists()


def test_fetch_cohort_archive_open_failure_is_blocked(tmp_path: Path) -> None:
    files, _cohort, _expected = _synthetic_cohort(tmp_path)
    kwargs = _fetch_kwargs(files)

    def failing(url: str) -> tuple[remote_zip.FetchRange, int]:
        raise TimeoutError("timed out")

    with pytest.raises(WillowSourceBlockedError, match="timed out"):
        fetch_cohort(
            13273331,
            ARCHIVE,
            PREFIX,
            tmp_path / "zenodo",
            fetch_url=kwargs["fetch_url"],
            open_archive=failing,
        )


def test_fetch_cohort_refuses_size_disagreement(tmp_path: Path) -> None:
    files, _cohort, _expected = _synthetic_cohort(tmp_path)
    kwargs = _fetch_kwargs(files)

    def wrong_size(url: str) -> bytes:
        return _record_json(len(kwargs["_data"]) + 1)

    with pytest.raises(WillowSourceBlockedError, match="size"):
        fetch_cohort(
            13273331,
            ARCHIVE,
            PREFIX,
            tmp_path / "zenodo",
            fetch_url=wrong_size,
            open_archive=kwargs["open_archive"],
        )


def test_fetch_cohort_refuses_md5_disagreement(tmp_path: Path) -> None:
    files, _cohort, _expected = _synthetic_cohort(tmp_path)
    kwargs = _fetch_kwargs(files)
    with pytest.raises(WillowSourceBlockedError, match="md5"):
        fetch_cohort(
            13273331,
            ARCHIVE,
            PREFIX,
            tmp_path / "zenodo",
            fetch_url=kwargs["fetch_url"],
            open_archive=kwargs["open_archive"],
            archive_md5_published="0" * 32,
        )


def test_fetch_cohort_refuses_missing_member(tmp_path: Path) -> None:
    files, _cohort, _expected = _synthetic_cohort(tmp_path)
    del files["sweep_bits.b8"]
    kwargs = _fetch_kwargs(files)
    with pytest.raises(WillowSourceBlockedError, match=r"sweep_bits\.b8"):
        fetch_cohort(
            13273331,
            ARCHIVE,
            PREFIX,
            tmp_path / "zenodo",
            fetch_url=kwargs["fetch_url"],
            open_archive=kwargs["open_archive"],
        )


def test_default_fetchers_go_through_urllib(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The default network path is urllib with a User-Agent, patched here so nothing leaves."""
    files, _cohort, _expected = _synthetic_cohort(tmp_path)
    data = _cohort_archive(files)
    seen: list[tuple[str, str | None]] = []

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeResponse:
        url = request.full_url
        seen.append((url, request.get_header("Range")))
        if url == "https://zenodo.org/api/records/13273331":
            return _FakeResponse(200, _record_json(len(data)), {"Content-Type": "application/json"})
        header = request.get_header("Range")
        assert header is not None
        start, end = (int(v) for v in header.removeprefix("bytes=").split("-"))
        body = data[start : end + 1]
        return _FakeResponse(
            206, body, {"Content-Range": f"bytes {start}-{start + len(body) - 1}/{len(data)}"}
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    receipts = fetch_cohort(13273331, ARCHIVE, PREFIX, tmp_path / "zenodo")
    assert (
        receipts["archive"]["url"]
        == f"https://zenodo.org/records/13273331/files/{ARCHIVE}?download=1"
    )
    assert seen[0] == ("https://zenodo.org/api/records/13273331", None)
    assert all(rng is not None for _url, rng in seen[1:])
