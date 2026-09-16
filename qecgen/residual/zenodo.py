"""Fetch the Willow d3/Z/r10 cohort members from Zenodo record 13273331, with receipts.

Google's release ships an ideal circuit, a noisy circuit, detection events, actual
observable flips and, per decoding pathway, the error model the decoder ran on. This
pipeline uses those shipped ``error_model.dem`` files verbatim, so the provenance of every
byte must be recorded at the moment it is fetched: record metadata, archive URL and size,
member path, central-directory sizes and CRC32, the sha256 of the extracted bytes and the
fetch time. ``receipts.json`` is that record.

The network calls are injectable (``fetch_url`` for the record JSON, ``open_archive`` for
the range reader) so the tests run against an in-memory archive; a failure on either path
raises :class:`WillowSourceBlockedError`, and the caller reports the dataset as blocked
rather than fitting a substitute model. The archive-level MD5 published by Zenodo is
recorded but *not* re-verified — doing so would require the 5.7 GB download — and the
receipt says so explicitly so nobody reads its presence as a check that ran.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import urllib.request
import zlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import stim

from qecgen.hardware import ImportedCohort, parse_b8
from qecgen.qa import clopper_pearson
from qecgen.residual.remote_zip import (
    FetchRange,
    RemoteZip,
    RemoteZipError,
    ZipMember,
    http_range_fetcher,
)
from qecgen.sampling import unpack_bits

WILLOW_COHORT_MEMBERS: tuple[str, ...] = (
    "circuit_ideal.stim",
    "circuit_noisy_si1000.stim",
    "detection_events.b8",
    "obs_flips_actual.b8",
    "sweep_bits.b8",
    "metadata.json",
    "decoding_results/correlated_matching_decoder_with_si1000_prior/error_model.dem",
    "decoding_results/correlated_matching_decoder_with_si1000_prior/obs_flips_predicted.b8",
    "decoding_results/correlated_matching_decoder_with_rl_optimized_prior/error_model.dem",
    "decoding_results/correlated_matching_decoder_with_rl_optimized_prior/obs_flips_predicted.b8",
)
"""Cohort-relative member paths fetched under ``cache_dir``. ``measurements.b8`` is
deliberately absent: it is the raw measurement record, tens of megabytes, and nothing in
this pipeline reads it."""

README_MEMBER = "README/README.md"
"""Archive-root-relative path of the release README, cached at ``cache_dir/README/README.md``."""

ZENODO_API = "https://zenodo.org/api/records/{record}"
ZENODO_DOWNLOAD = "https://zenodo.org/records/{record}/files/{archive}?download=1"
_USER_AGENT = "qecgen-residual/0.1 (zenodo receipts)"
_MEASUREMENT_GATES = frozenset({"M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"})

FetchUrl = Callable[[str], bytes]
OpenArchive = Callable[[str], tuple[FetchRange, int]]


class WillowSourceBlockedError(RuntimeError):
    """The Zenodo source could not be reached or disagrees with the pinned record.

    Raised before any staging directory exists so the build reports "blocked" with the
    reason and never fabricates Willow predictions from a substitute model.
    """


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _default_fetch_url(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url)
    req.add_header("User-Agent", _USER_AGENT)
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        status = int(response.status)
        if status != 200:
            raise WillowSourceBlockedError(f"{url} answered HTTP {status}")
        return bytes(response.read())


def _record_summary(record: int, payload: bytes, record_url: str) -> dict[str, Any]:
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WillowSourceBlockedError(f"{record_url}: record JSON unreadable: {exc}") from exc
    if not isinstance(parsed, dict):
        raise WillowSourceBlockedError(f"{record_url}: record JSON is not an object")
    metadata = parsed.get("metadata") or {}
    licence = metadata.get("license")
    files = parsed.get("files") or []
    return {
        "id": record,
        "url": record_url,
        "title": metadata.get("title"),
        "version": metadata.get("version"),
        "doi": parsed.get("doi") or metadata.get("doi"),
        "publication_date": metadata.get("publication_date"),
        "license": licence.get("id") if isinstance(licence, dict) else licence,
        "created": parsed.get("created"),
        "files": [
            {"key": f.get("key"), "size": f.get("size"), "checksum": f.get("checksum")}
            for f in files
            if isinstance(f, dict)
        ],
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    """Write via a sibling temp file and ``os.replace`` so a crash leaves no half member.

    A truncated ``.dem`` still parses as a smaller DEM; a truncated ``.b8`` still splits
    into whole rows. Neither would fail loudly on the next run, so the file must either be
    complete or absent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-download")
    with open(tmp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _cached_is_intact(path: Path, member: ZipMember) -> bool:
    if not path.is_file() or path.stat().st_size != member.uncompressed_size:
        return False
    return (zlib.crc32(path.read_bytes()) & 0xFFFFFFFF) == member.crc32


def fetch_cohort(
    record: int,
    archive: str,
    cohort_prefix: str,
    cache_dir: Path,
    *,
    fetch_url: FetchUrl | None = None,
    open_archive: OpenArchive | None = None,
    archive_md5_published: str | None = None,
    members: tuple[str, ...] = WILLOW_COHORT_MEMBERS,
) -> dict[str, Any]:
    """Fetch the cohort members by HTTP range into ``cache_dir`` and write ``receipts.json``.

    Members already present with the central directory's size and CRC32 are kept (their
    original ``fetched_at`` is carried over from the previous receipts); anything absent
    or damaged is fetched again. The record JSON is fetched first so the listed archive
    size can be checked against what the range server reports before any member is read.
    """
    if not cohort_prefix.endswith("/") or "/" not in cohort_prefix.rstrip("/"):
        raise ValueError(
            "cohort_prefix must be an archive path ending in '/' with at least two components"
        )
    fetch_json = fetch_url or _default_fetch_url
    open_range = open_archive or http_range_fetcher
    record_url = ZENODO_API.format(record=record)
    archive_url = ZENODO_DOWNLOAD.format(record=record, archive=archive)
    try:
        record_summary = _record_summary(record, fetch_json(record_url), record_url)
    except (OSError, RemoteZipError) as exc:
        raise WillowSourceBlockedError(f"{record_url}: {exc}") from exc
    listed = [f for f in record_summary["files"] if f["key"] == archive]
    if len(listed) != 1:
        raise WillowSourceBlockedError(
            f"record {record} lists {len(listed)} files named {archive!r}; expected exactly one"
        )
    listed_size = listed[0]["size"]
    checksum = str(listed[0]["checksum"] or "")
    published_md5 = checksum.removeprefix("md5:") if checksum.startswith("md5:") else None
    if archive_md5_published is not None and published_md5 != archive_md5_published.lower():
        raise WillowSourceBlockedError(
            f"record {record} publishes md5 {published_md5!r} for {archive}, the configuration "
            f"expects {archive_md5_published!r}"
        )
    try:
        fetch_range, size = open_range(archive_url)
        if size != listed_size:
            raise WillowSourceBlockedError(
                f"{archive}: archive size disagreement, range server reports {size} bytes but "
                f"the record lists {listed_size}"
            )
        reader = RemoteZip(fetch_range, size)
        by_name = {m.name: m for m in reader.members()}
    except (OSError, RemoteZipError) as exc:
        raise WillowSourceBlockedError(f"{archive_url}: {exc}") from exc
    root = cohort_prefix.split("/", 1)[0]
    wanted: list[tuple[str, str]] = [(cohort_prefix + name, name) for name in members]
    wanted.append((f"{root}/{README_MEMBER}", README_MEMBER))
    missing = [name for name, _ in wanted if name not in by_name]
    if missing:
        raise WillowSourceBlockedError(f"{archive}: members absent from the archive: {missing}")

    previous = _previous_receipts(cache_dir / "receipts.json")
    entries: list[dict[str, Any]] = []
    for member_name, relative in wanted:
        member = by_name[member_name]
        target = cache_dir / relative
        if _cached_is_intact(target, member):
            payload = target.read_bytes()
            status = "cached"
            fetched_at = previous.get(member_name) or _now()
        else:
            try:
                payload = reader.read(member)
            except (OSError, RemoteZipError) as exc:
                raise WillowSourceBlockedError(f"{member_name}: {exc}") from exc
            _atomic_write(target, payload)
            status = "fetched"
            fetched_at = _now()
        entries.append(
            {
                "member": member_name,
                "path": relative,
                "compressed_size": member.compressed_size,
                "uncompressed_size": member.uncompressed_size,
                "crc32": f"{member.crc32:08x}",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "compression_method": member.method,
                "status": status,
                "fetched_at": fetched_at,
            }
        )
    receipts: dict[str, Any] = {
        "record": record_summary,
        "archive": {
            "key": archive,
            "url": archive_url,
            "size": size,
            "md5_published": published_md5,
            "md5_status": "not_reverified",
            "member_count": len(by_name),
            "fetch_method": "HTTP range requests against the ZIP64 central directory",
        },
        "cohort_prefix": cohort_prefix,
        "cache_dir": str(cache_dir.resolve()),
        "members": entries,
        "receipts_written_at": _now(),
    }
    _atomic_write(
        cache_dir / "receipts.json",
        json.dumps(receipts, indent=2, sort_keys=True, allow_nan=False).encode("utf-8"),
    )
    return receipts


def _previous_receipts(path: Path) -> dict[str, str]:
    """``member -> fetched_at`` from an earlier run, or empty if there is none or it is unreadable.

    A damaged receipts file must not block a re-fetch: the members are re-verified against
    the central directory regardless, and the file is rewritten whole afterwards.
    """
    if not path.is_file():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        return {
            str(m["member"]): str(m["fetched_at"])
            for m in parsed.get("members", [])
            if isinstance(m, dict) and "member" in m and "fetched_at" in m
        }
    except (OSError, ValueError, AttributeError):
        return {}


def qubit_partition(
    circuit: stim.Circuit,
) -> tuple[list[tuple[float, ...]], list[tuple[float, ...]]]:
    """Coordinates of the data qubits and of the measurement (ancilla) qubits.

    Google's Willow circuit measures ancillas with plain ``M`` (not ``MR``) every cycle and
    the data qubits with ``M`` once at the end, so the "trailing run of non-resetting
    measurements" rule from ``qecgen.correction`` would merge both layers. The rule here
    counts instead: a qubit measured with a resetting gate or more than once is a
    measurement qubit; a qubit measured exactly once without reset is a data qubit; a
    qubit that is never measured (the Willow file lists two such ``QUBIT_COORDS``) belongs
    to neither and must not appear in the metadata.
    """
    counts: dict[int, int] = {}
    resetting: set[int] = set()
    for instruction in circuit.flattened():
        name = instruction.name
        if name not in _MEASUREMENT_GATES:
            if stim.gate_data(name).produces_measurements:
                raise ValueError(f"unsupported measurement gate {name} for qubit partition")
            continue
        for target in instruction.targets_copy():
            if not target.is_qubit_target:
                raise ValueError(f"{name} has a non-qubit target; cannot partition qubits")
            counts[target.value] = counts.get(target.value, 0) + 1
            if name.startswith("MR"):
                resetting.add(target.value)
    coordinates = circuit.get_final_qubit_coordinates()
    data: list[tuple[float, ...]] = []
    meas: list[tuple[float, ...]] = []
    for qubit, count in sorted(counts.items()):
        if qubit not in coordinates:
            raise ValueError(f"measured qubit {qubit} has no QUBIT_COORDS entry")
        coords = tuple(float(c) for c in coordinates[qubit])
        if count > 1 or qubit in resetting:
            meas.append(coords)
        else:
            data.append(coords)
    return data, meas


def _coordinate_set(values: Any, key: str) -> set[tuple[float, ...]]:
    if not isinstance(values, list):
        raise ValueError(f"metadata.json {key} must be a list of coordinate pairs")
    result: set[tuple[float, ...]] = set()
    for entry in values:
        if not isinstance(entry, list) or not all(isinstance(v, int | float) for v in entry):
            raise ValueError(f"metadata.json {key} entry {entry!r} is not a numeric coordinate")
        result.add(tuple(float(v) for v in entry))
    return result


def verify_cohort(
    cache_dir: Path,
    cohort: ImportedCohort,
    circuit_sha256: str,
    expected: Mapping[str, Any],
    *,
    pm_guess: np.ndarray | None = None,
) -> dict[str, Any]:
    """Cross-check the cached Zenodo members against the local mirror cohort.

    Byte equality of ``circuit_ideal.stim`` and of the b8 arrays is the provenance
    statement the pipeline makes ("the mirror rows equal what Zenodo serves"); the
    ``metadata.json`` and ``QUBIT_COORDS`` checks are the structural evidence the brief asks
    for beyond byte equality. Google's ``obs_flips_predicted.b8`` is read only for the
    diagnostic block returned here; it is never a feature or a target.
    """
    report: dict[str, Any] = {"cache_dir": str(cache_dir.resolve())}
    ideal = cohort.ideal
    n_detectors = ideal.num_detectors
    n_observables = ideal.num_observables

    circuit_bytes = (cache_dir / "circuit_ideal.stim").read_bytes()
    actual_sha = hashlib.sha256(circuit_bytes).hexdigest()
    if actual_sha != circuit_sha256:
        raise ValueError(
            f"circuit_ideal.stim sha256 {actual_sha} does not equal the expected {circuit_sha256}"
        )
    mirror_bytes = Path(str(cohort.source["circuit_path"])).read_bytes()
    if circuit_bytes != mirror_bytes:
        raise ValueError("circuit_ideal.stim bytes differ from the local mirror circuit file")
    if stim.Circuit(circuit_bytes.decode("utf-8")) != ideal:
        raise ValueError("circuit_ideal.stim parses to a different circuit than the cohort's")
    report["circuit_ideal_sha256"] = actual_sha
    report["circuit_bytes_equal_mirror"] = True

    detectors = parse_b8((cache_dir / "detection_events.b8").read_bytes(), n_detectors)
    if detectors.shape != cohort.detectors.shape or not np.array_equal(detectors, cohort.detectors):
        rows = (
            int((detectors != cohort.detectors).any(axis=1).sum())
            if detectors.shape == cohort.detectors.shape
            else None
        )
        raise ValueError(
            f"detection_events.b8 ({detectors.shape}) differs from the cohort detectors "
            f"({cohort.detectors.shape}); rows differing: {rows}"
        )
    observables = parse_b8((cache_dir / "obs_flips_actual.b8").read_bytes(), n_observables)
    if observables.shape != cohort.observables.shape or not np.array_equal(
        observables, cohort.observables
    ):
        raise ValueError("obs_flips_actual.b8 differs from the cohort observables")
    report["detectors_equal_cohort"] = True
    report["observables_equal_cohort"] = True
    report["shots"] = len(detectors)

    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json is not a JSON object")
    for key in ("distance", "rounds"):
        if metadata.get(key) != expected[key]:
            raise ValueError(
                f"metadata.json {key}={metadata.get(key)!r} disagrees with expected "
                f"{expected[key]!r}"
            )
    if str(metadata.get("basis", "")).upper() != str(expected["basis"]).upper():
        raise ValueError(
            f"metadata.json basis={metadata.get('basis')!r} disagrees with expected "
            f"{expected['basis']!r}"
        )
    if metadata.get("shots") != len(detectors):
        raise ValueError(
            f"metadata.json shots={metadata.get('shots')!r} disagrees with the {len(detectors)} "
            "cohort rows"
        )
    data_coords, meas_coords = qubit_partition(ideal)
    meta_data = _coordinate_set(metadata.get("data_qubit_coords"), "data_qubit_coords")
    meta_meas = _coordinate_set(metadata.get("meas_qubit_coords"), "meas_qubit_coords")
    if meta_data != set(data_coords) or meta_meas != set(meas_coords):
        raise ValueError(
            "metadata.json data_qubit_coords/meas_qubit_coords are not the circuit's "
            f"QUBIT_COORDS partition (metadata {len(meta_data)}/{len(meta_meas)}, circuit "
            f"{len(data_coords)}/{len(meas_coords)})"
        )
    report["metadata"] = {
        "basis": metadata["basis"],
        "rounds": metadata["rounds"],
        "distance": metadata["distance"],
        "shots": metadata["shots"],
    }
    report["qubit_partition_matches_metadata"] = True
    report["n_data_qubits"] = len(data_coords)
    report["n_meas_qubits"] = len(meas_coords)

    noisy_bytes = (cache_dir / "circuit_noisy_si1000.stim").read_bytes()
    noisy = stim.Circuit(noisy_bytes.decode("utf-8"))
    if noisy.num_detectors != n_detectors or noisy.num_observables != n_observables:
        raise ValueError(
            "circuit_noisy_si1000.stim declares a different detector/observable count than "
            "the ideal circuit"
        )
    report["noisy_circuit"] = {
        "sha256": hashlib.sha256(noisy_bytes).hexdigest(),
        "num_detectors": noisy.num_detectors,
        "num_observables": noisy.num_observables,
        # False for the real Willow file (Google's noisy circuit is not the ideal one plus
        # noise); recorded as a fact, never asserted.
        "without_noise_equals_ideal": noisy.without_noise() == ideal,
    }

    sweep = (cache_dir / "sweep_bits.b8").read_bytes()
    report["sweep_bits"] = {
        "bytes": len(sweep),
        "nonzero_bytes": int(np.count_nonzero(np.frombuffer(sweep, dtype=np.uint8))),
    }

    truth = unpack_bits(observables, n_observables)[:, 0]
    if pm_guess is not None:
        guess = np.asarray(pm_guess).reshape(-1)
        if guess.shape[0] != truth.shape[0]:
            raise ValueError("pm_guess length differs from the cohort row count")
    report["google_decoders"] = _google_diagnostics(cache_dir, truth, pm_guess)
    return report


def _google_diagnostics(
    cache_dir: Path, truth: np.ndarray, pm_guess: np.ndarray | None
) -> dict[str, dict[str, Any]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    results_dir = cache_dir / "decoding_results"
    if not results_dir.is_dir():
        return diagnostics
    for pathway_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        predicted_path = pathway_dir / "obs_flips_predicted.b8"
        dem_path = pathway_dir / "error_model.dem"
        entry: dict[str, Any] = {}
        if dem_path.is_file():
            entry["error_model_sha256"] = hashlib.sha256(dem_path.read_bytes()).hexdigest()
        if predicted_path.is_file():
            predicted = unpack_bits(parse_b8(predicted_path.read_bytes(), 1), 1)[:, 0]
            if predicted.shape[0] != truth.shape[0]:
                raise ValueError(
                    f"{predicted_path.name} in {pathway_dir.name} has {predicted.shape[0]} rows, "
                    f"cohort has {truth.shape[0]}"
                )
            failures = int((predicted != truth).sum())
            interval = clopper_pearson(failures, int(truth.shape[0]), 0.05)
            entry["failures"] = failures
            entry["error_rate"] = interval.point
            entry["ci95"] = [interval.low, interval.high]
            if pm_guess is not None:
                guess = np.asarray(pm_guess).reshape(-1).astype(np.uint8)
                entry["agreement_with_pm_guess"] = float((guess == predicted).mean())
        if entry:
            diagnostics[pathway_dir.name] = entry
    return diagnostics
