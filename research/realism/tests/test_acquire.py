"""Downloader failure paths must retain evidence without naming bad bytes as complete."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import urllib.request
import zipfile
from dataclasses import replace
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from research.realism.acquire import Source, acquire, catalogue, intake_lock, zip_inventory


class Response(io.BytesIO):
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        interrupt_after_first_read: bool = False,
    ) -> None:
        super().__init__(body)
        self.status = status
        self.headers = headers or {}
        self.interrupt_after_first_read = interrupt_after_first_read
        self.reads = 0

    def geturl(self) -> str:
        return "https://cdn.example/data?short_lived_signature=omit-this"

    def read(self, size: int | None = -1) -> bytes:
        self.reads += 1
        if self.interrupt_after_first_read and self.reads > 1:
            raise TimeoutError("Transfer interrupted")
        return super().read(size)


def source(payload: bytes = b"abcdef") -> Source:
    return Source(
        asset_id="sample",
        filename="data.bin",
        urls=("https://example.org/data",),
        source="https://doi.org/example",
        licence="CC-BY-4.0",
        licence_url="https://example.org/licence",
        version="fixed-version",
        max_bytes=len(payload),
        expected_bytes=len(payload),
        publisher_checksum="sha256:" + hashlib.sha256(payload).hexdigest(),
    )


def run_response(
    tmp_path: Path, response: Response, entry: Source | None = None, **kwargs: Any
) -> dict[str, Any]:
    return acquire(
        entry or source(),
        tmp_path,
        attempts=1,
        opener=lambda _request, **_kwargs: response,
        **kwargs,
    )


def test_checked_bytes_get_final_name_and_receipt(tmp_path: Path) -> None:
    result = run_response(tmp_path, Response(b"abcdef", headers={"Content-Length": "6"}))
    assert result["status"] == "verified"
    assert Path(result["path"]).read_bytes() == b"abcdef"
    receipt = json.loads((tmp_path / "intake/sample.json").read_text())
    assert receipt["source"]["licence"] == "CC-BY-4.0"
    assert receipt["hashes"]["sha256"] == hashlib.sha256(b"abcdef").hexdigest()
    assert receipt["attempts"][0]["resolved_url"] == "https://cdn.example/data"
    assert receipt["attempts"][0]["received_bytes"] == 6
    assert not (tmp_path / "raw/sample/data.bin.part").exists()


def test_checksum_mismatch_never_publishes(tmp_path: Path) -> None:
    result = run_response(tmp_path, Response(b"xxxxxx"))
    assert result["status"] == "failed"
    assert "checksum mismatch" in result["attempts"][0]["error"]
    assert not Path(result["path"]).exists()
    assert (tmp_path / "raw/sample/data.bin.part").read_bytes() == b"xxxxxx"


def interrupt(tmp_path: Path) -> None:
    result = run_response(tmp_path, Response(b"abc", interrupt_after_first_read=True))
    assert result["status"] == "failed"
    assert result["partial_bytes"] == 3


def test_interrupted_download_resumes_exactly_once(tmp_path: Path) -> None:
    interrupt(tmp_path)

    def open_range(request: urllib.request.Request, **_kwargs: Any) -> Response:
        assert request.get_header("Range") == "bytes=3-"
        return Response(b"def", status=206, headers={"Content-Range": "bytes 3-5/6"})

    result = acquire(source(), tmp_path, attempts=1, opener=open_range)
    assert result["status"] == "verified"
    assert Path(result["path"]).read_bytes() == b"abcdef"
    assert len(result["attempts"]) == 2
    assert result["attempts"][1]["resume_from"] == 3


def test_server_ignoring_range_restarts_without_duplicate_bytes(tmp_path: Path) -> None:
    interrupt(tmp_path)
    result = run_response(tmp_path, Response(b"abcdef", headers={"Content-Length": "6"}))
    assert result["status"] == "verified"
    assert Path(result["path"]).read_bytes() == b"abcdef"


def test_bad_content_range_does_not_append(tmp_path: Path) -> None:
    interrupt(tmp_path)
    result = run_response(
        tmp_path, Response(b"def", status=206, headers={"Content-Range": "bytes 2-4/6"})
    )
    assert result["status"] == "failed"
    assert (tmp_path / "raw/sample/data.bin.part").read_bytes() == b"abc"


def test_short_response_is_not_accepted_without_publisher_checksum(tmp_path: Path) -> None:
    entry = replace(source(), expected_bytes=None, publisher_checksum=None)
    result = run_response(tmp_path, Response(b"abc", headers={"Content-Length": "6"}), entry)
    assert result["status"] == "failed"
    assert "ended before" in result["attempts"][0]["error"]


@pytest.mark.parametrize("content_length", [None, "7"])
def test_oversize_object_is_not_published(tmp_path: Path, content_length: str | None) -> None:
    headers = {"Content-Length": content_length} if content_length is not None else {}
    result = run_response(tmp_path, Response(b"abcdefg", headers=headers))
    assert result["status"] == "failed"
    assert not Path(result["path"]).exists()


def test_total_budget_includes_preexisting_other_assets(tmp_path: Path) -> None:
    directory = tmp_path / "raw/another-asset"
    directory.mkdir(parents=True)
    (directory / "file").write_bytes(b"0123456789")
    result = run_response(tmp_path, Response(b"abcdef"), download_budget=15)
    assert result["status"] == "failed"
    assert "Cumulative download byte budget" in result["attempts"][0]["error"]
    assert not Path(result["path"]).exists()


def test_changed_source_cannot_reuse_partial(tmp_path: Path) -> None:
    interrupt(tmp_path)
    changed = replace(source(), version="different-version")
    with pytest.raises(ValueError, match="different source"):
        run_response(tmp_path, Response(b"def"), changed)
    assert (tmp_path / "raw/sample/data.bin.part").read_bytes() == b"abc"


def test_corrupted_existing_asset_is_not_silently_trusted(tmp_path: Path) -> None:
    result = run_response(tmp_path, Response(b"abcdef"))
    Path(result["path"]).write_bytes(b"wrong!")
    result = run_response(tmp_path, Response(b"abcdef"))
    assert result["status"] == "failed"
    assert "checksum mismatch" in result["error"]
    assert Path(result["path"]).read_bytes() == b"wrong!"


def test_cached_asset_without_publisher_hash_retains_original_receipt_hash(tmp_path: Path) -> None:
    entry = replace(source(), publisher_checksum=None)
    result = run_response(tmp_path, Response(b"abcdef"), entry)
    original_hash = result["hashes"]["sha256"]
    Path(result["path"]).write_bytes(b"wrong!")
    result = run_response(tmp_path, Response(b"abcdef"), entry)
    assert result["status"] == "failed"
    assert "receipt SHA-256" in result["error"]
    assert result["hashes"]["sha256"] == original_hash


def test_checksum_invalid_partial_restarts_without_range(tmp_path: Path) -> None:
    result = run_response(tmp_path, Response(b"xxxxxx"))
    assert result["status"] == "failed"

    def fresh_response(request: urllib.request.Request, **_kwargs: Any) -> Response:
        assert request.get_header("Range") is None
        return Response(b"abcdef")

    result = acquire(source(), tmp_path, attempts=1, opener=fresh_response)
    assert result["status"] == "verified"
    assert Path(result["path"]).read_bytes() == b"abcdef"
    assert (
        result["attempts"][0]["invalid_hashes"]["sha256"] == hashlib.sha256(b"xxxxxx").hexdigest()
    )


def test_retry_bytes_count_even_when_server_replaces_partial(tmp_path: Path) -> None:
    run_response(tmp_path, Response(b"abc", interrupt_after_first_read=True), download_budget=6)
    result = run_response(
        tmp_path,
        Response(b"abcdef", headers={"Content-Length": "6"}),
        download_budget=6,
    )
    assert result["status"] == "failed"
    assert "Cumulative download byte budget" in result["attempts"][-1]["error"]
    assert sum(attempt["received_bytes"] for attempt in result["attempts"]) == 3
    ledger = json.loads((tmp_path / "intake/transfer-budget.json").read_text())
    assert ledger["charged_bytes"] == 6
    assert not Path(result["path"]).exists()


def test_interrupted_replacement_becomes_resumable(tmp_path: Path) -> None:
    run_response(tmp_path, Response(b"xxxxxx"))
    run_response(tmp_path, Response(b"abc", interrupt_after_first_read=True))

    def resumed_response(request: urllib.request.Request, **_kwargs: Any) -> Response:
        assert request.get_header("Range") == "bytes=3-"
        return Response(b"def", status=206, headers={"Content-Range": "bytes 3-5/6"})

    result = acquire(source(), tmp_path, attempts=1, opener=resumed_response)
    assert result["status"] == "verified"
    assert Path(result["path"]).read_bytes() == b"abcdef"


def test_complete_partial_with_unknown_size_is_verified_without_request(tmp_path: Path) -> None:
    entry = replace(source(), expected_bytes=None)
    result = run_response(tmp_path, Response(b"abcdef", interrupt_after_first_read=True), entry)
    assert result["status"] == "failed"

    def no_request(_request: urllib.request.Request, **_kwargs: Any) -> Response:
        pytest.fail("A complete publisher-verified partial must not request Range at EOF")

    result = acquire(entry, tmp_path, attempts=1, opener=no_request)
    assert result["status"] == "verified"
    assert Path(result["path"]).read_bytes() == b"abcdef"


def test_range_rejection_restarts_unverified_unknown_size_partial(tmp_path: Path) -> None:
    entry = replace(source(), expected_bytes=None)
    run_response(tmp_path, Response(b"xxxxxx", interrupt_after_first_read=True), entry)

    def range_rejected(request: urllib.request.Request, **_kwargs: Any) -> Response:
        assert request.get_header("Range") == "bytes=6-"
        raise urllib.error.HTTPError(request.full_url, 416, "Range invalid", Message(), None)

    result = acquire(entry, tmp_path, attempts=1, opener=range_rejected)
    assert result["restart_partial"] is True
    result = run_response(tmp_path, Response(b"abcdef"), entry)
    assert result["status"] == "verified"
    assert result["attempts"][-1]["resume_from"] == 0


def test_concurrent_acquisition_cannot_race_the_budget_counter(tmp_path: Path) -> None:
    with intake_lock(tmp_path), pytest.raises(OSError):
        run_response(tmp_path, Response(b"abcdef"))
    assert run_response(tmp_path, Response(b"abcdef"))["status"] == "verified"


@pytest.mark.parametrize("filename", ["../outside", "..", "a/b", "C:drive", "a\\b"])
def test_catalogue_rejects_nonportable_paths(filename: str) -> None:
    with pytest.raises(ValueError, match="basenames"):
        replace(source(), filename=filename)


def test_zip_inventory_does_not_extract_members(tmp_path: Path) -> None:
    path = tmp_path / "archive.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("nested/shots.b8", b"0123456789")
    result = zip_inventory(path, max_uncompressed=5)
    assert result["uncompressed_bytes"] == 10
    assert result["within_extraction_budget"] is False
    assert result["extracted"] is False
    assert not (tmp_path / "nested").exists()


def test_source_catalogue_retains_deferred_google_leads() -> None:
    entries = catalogue(Path(__file__).parents[1] / "sources.json")
    assert entries["google-dynamic-circuits"].deferred
    assert entries["google-reinforcement-learning"].deferred
    assert not entries["google-2022"].deferred
    assert "THIRD-PARTY DERIVED PILOT" in entries["willow-derived-z-r010"].notes
