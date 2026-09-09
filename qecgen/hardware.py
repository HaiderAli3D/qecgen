"""Checked import of published detector events and actual logical outcomes.

Only actual observable outcomes are targets. Published decoder predictions and noisy
circuit probabilities are not exposed to calibration or training. This is an isolated
adapter. Original archives and third-party derived cohorts retain distinct provenance.
"""

from __future__ import annotations

import hashlib
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import stim
from numpy.typing import NDArray

# Catalogue pins verified during intake; hashes identify the derived publication,
# not an independently checked original Google archive. Unknown user tables must
# never inherit this attribution merely because their supplied hash matches.
WILLOW_CIRCUITS = {
    "9cc2b0fedb742c2e7abca54a8a0946695d2b505c40f400ef56cf08d765b8af67": (
        "X",
        10,
        "f162c2caca9d7ddfdf103c60b41d6dfc2ed24abf958f27f7dbefd8741b9ce7e1",
    ),
    "386e4dc3106e56fd96514289e215c204a2b99f8c25e09f0ebc438df5fdc73c71": (
        "X",
        13,
        "55fe7cd0934f594560320745c17d382abfd3f8f51d02a31842150dc5c339b3a1",
    ),
    "fba4d5575c0afa11ce2126acbbe7d3a2546609ecac66195ea2ba696c45ef085e": (
        "Z",
        10,
        "2324ecb77a859d005850341a442396d6c3e9f8a2f12a4928364f27933b8ac696",
    ),
    "6cdd932ba7a44195a8d920db62d9d51cd8381c8a83940d4d1b8d5c36cbcb1f50": (
        "Z",
        13,
        "9c888bbe39c0cc463bc126528d95d118669eb2f782ed39cd05fda9b09faddac9",
    ),
}


def validate_layout_identity(
    circuit: stim.Circuit, description: dict[str, Any], file_sha256: str
) -> str:
    """Require an audited layout instead of labelling an arbitrary circuit d3/Z."""
    from qecgen.circuits import Basis, build_circuit

    known = WILLOW_CIRCUITS.get(file_sha256)
    if known is not None:
        basis, rounds, _ = known
        if (
            description["distance"] != 3
            or description["basis"].upper() != basis
            or description["rounds"] != rounds
            or not description["rotated"]
        ):
            raise ValueError("Circuit metadata disagrees with the catalogue-verified Willow layout")
        return (
            "catalogue-verified Willow circuit; detector/observable mapping audited, "
            "not correction roles"
        )
    canonical, _ = build_circuit(
        description["distance"],
        0.0,
        rounds=description["rounds"],
        basis=Basis(description["basis"]),
        rotated=description["rotated"],
    )
    if circuit.flattened() != canonical.flattened():
        raise ValueError(
            "Unsupported external circuit identity: use a canonical Stim surface-code circuit "
            "or one of the catalogue-verified Willow circuits; dimensions alone are insufficient"
        )
    return "exact canonical Stim surface-code circuit"


@dataclass(frozen=True)
class ImportedCohort:
    detectors: NDArray[np.uint8]
    observables: NDArray[np.uint8]
    ideal: stim.Circuit
    source: dict[str, Any]


@dataclass(frozen=True)
class Split:
    train: NDArray[np.int64]
    validation: NDArray[np.int64]
    test: NDArray[np.int64]


def parse_b8(payload: bytes, width: int) -> NDArray[np.uint8]:
    if width < 1:
        raise ValueError("a hardware cohort must have a positive detector width")
    packed = math.ceil(width / 8)
    if not payload or len(payload) % packed:
        raise ValueError("b8 byte count must contain whole nonempty rows")
    rows = np.frombuffer(payload, dtype=np.uint8).reshape(-1, packed).copy()
    if width % 8 and np.any(rows[:, -1] >> (width % 8)):
        raise ValueError("nonzero padding bits disagree with the declared detector width")
    return rows


def parse_01(payload: bytes, width: int) -> NDArray[np.uint8]:
    lines = payload.decode("ascii").splitlines()
    if not lines or width < 1:
        raise ValueError("actual observable outcomes must be nonempty")
    if any(len(line) != width or set(line) - {"0", "1"} for line in lines):
        raise ValueError("observable rows must contain exactly the declared number of 0/1 bits")
    bits = np.frombuffer("".join(lines).encode("ascii"), dtype=np.uint8) - ord("0")
    return np.packbits(bits.reshape(-1, width), axis=1, bitorder="little")


def load_google_cohort(archive: Path, cohort: str) -> ImportedCohort:
    """Read one exact cohort; no extraction, filename guessing or target substitution."""
    names = {
        "circuit": f"{cohort.rstrip('/')}/circuit_noisy.stim",
        "detectors": f"{cohort.rstrip('/')}/detection_events.b8",
        "observables": f"{cohort.rstrip('/')}/obs_flips_actual.01",
    }
    with zipfile.ZipFile(archive) as source:
        missing = sorted(set(names.values()) - set(source.namelist()))
        if missing:
            raise ValueError(f"cohort requires actual outcomes and exact source files: {missing}")
        for name in names.values():
            if source.getinfo(name).file_size > 512_000_000:
                raise ValueError(f"pilot member exceeds the 512 MB intake limit: {name}")
        payloads = {key: source.read(name) for key, name in names.items()}
    circuit = stim.Circuit(payloads["circuit"].decode("utf-8"))
    detectors = parse_b8(payloads["detectors"], circuit.num_detectors)
    observables = parse_01(payloads["observables"], circuit.num_observables)
    if len(detectors) != len(observables):
        raise ValueError("detector and actual-observable row counts differ")
    ideal = circuit.without_noise()
    if (ideal.num_detectors, ideal.num_observables, ideal.num_measurements) != (
        circuit.num_detectors,
        circuit.num_observables,
        circuit.num_measurements,
    ):
        raise ValueError("removing source noise changed the experiment's measurement convention")
    return ImportedCohort(
        detectors=detectors,
        observables=observables,
        ideal=ideal,
        source={
            "archive": str(archive.resolve()),
            "cohort": cohort,
            "doi": "10.5281/zenodo.6804040",
            "members": {
                key: {"name": names[key], "sha256": hashlib.sha256(value).hexdigest()}
                for key, value in payloads.items()
            },
            "source_noise_weights_used": False,
            "source_decoder_predictions_used": False,
            "ordering": "source row order; acquisition chronology not established",
            "shots": len(detectors),
            "n_detectors": ideal.num_detectors,
            "n_observables": ideal.num_observables,
            "ideal_circuit_sha256": hashlib.sha256(str(ideal).encode()).hexdigest(),
        },
    )


def blocked_split(shots: int, guard: int = 128) -> Split:
    """Reserve test rows before fitting, without claiming unknown rows are chronological.

    The guard is a declared pilot precaution, not an empirically proven independence
    distance. Final study groups and guards must be justified at the design gate.
    """
    if shots < 1 or guard < 0:
        raise ValueError("shots must be positive and guard must be nonnegative")
    first, second = int(shots * 0.6), int(shots * 0.8)
    if first - guard <= 0 or first + guard >= second - guard or second + guard >= shots:
        raise ValueError("not enough rows for three nonempty partitions and guard bands")
    return Split(
        np.arange(first - guard, dtype=np.int64),
        np.arange(first + guard, second - guard, dtype=np.int64),
        np.arange(second + guard, shots, dtype=np.int64),
    )


def load_willow_derived(
    table_path: Path, circuit_path: Path, expected: dict[str, Any]
) -> ImportedCohort:
    """A verified mirror is not an independently verified original hardware archive.

    The publisher's conversion source identifies ``observable`` as actual flips and
    compares its output with Google's shipped outcomes. We verify the pinned mirror's
    bytes, typed columns and identity, never claim to have repeated the original check.
    """
    payloads = {}
    for key, path in (("table_sha256", table_path), ("circuit_sha256", circuit_path)):
        if path.stat().st_size > 512_000_000:
            raise ValueError(
                "One imported cohort file exceeds the 512 MB materialized intake limit"
            )
        payload = path.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected[key]:
            raise ValueError(f"{key} does not match the pinned acquisition")
        payloads[key] = payload
    table = pq.read_table(pa.BufferReader(payloads["table_sha256"]))
    required = {"shot", "detectors", "observable", "distance", "basis", "rounds", "orientation"}
    if not required <= set(table.column_names):
        raise ValueError("mirror is missing actual outcomes or identity columns")
    if not pa.types.is_boolean(table.schema.field("observable").type):
        raise ValueError("actual observable column must be boolean")
    detector_type = table.schema.field("detectors").type
    if not pa.types.is_list(detector_type) or not pa.types.is_boolean(detector_type.value_type):
        raise ValueError("detectors must be a list of booleans")
    if any(table[name].null_count for name in required):
        raise ValueError("null source fields must not be converted to fabricated zeros")
    for key in ("distance", "basis", "rounds", "orientation"):
        if table[key].unique().to_pylist() != [expected[key]]:
            raise ValueError(f"cohort identity disagrees with expected {key}")
    shots = table["shot"].to_numpy()
    if len(shots) == 0 or not np.array_equal(shots, np.arange(len(shots))):
        raise ValueError(
            "shot column must be in original row order with no duplicates or omissions"
        )
    ideal = stim.Circuit(payloads["circuit_sha256"].decode("utf-8"))
    if ideal != ideal.without_noise():
        raise ValueError("mirror ideal circuit contains source-fitted noise")
    if ideal.num_observables != 1:
        raise ValueError("this mirror's one target cannot represent a multi-observable circuit")
    cells = table["detectors"].to_pylist()
    if any(len(row) != ideal.num_detectors or any(value is None for value in row) for row in cells):
        raise ValueError("detector rows must match circuit width and contain no null bits")
    detectors = np.packbits(np.asarray(cells, dtype=bool), axis=1, bitorder="little")
    outcomes = table["observable"].to_numpy().reshape(-1, 1)
    published = WILLOW_CIRCUITS.get(expected["circuit_sha256"])
    verified = published == (expected["basis"], expected["rounds"], expected["table_sha256"])
    verified = verified and expected["distance"] == 3 and expected["orientation"] == "q10_7"
    return ImportedCohort(
        detectors,
        np.packbits(outcomes, axis=1, bitorder="little"),
        ideal,
        {
            **expected,
            "table_path": str(table_path.resolve()),
            "circuit_path": str(circuit_path.resolve()),
            "shots": len(shots),
            "n_detectors": ideal.num_detectors,
            "n_observables": ideal.num_observables,
            "original_doi": "10.5281/zenodo.13273331" if verified else None,
            "mirror_revision": "79ea4cbbc278047c9ce5d4d74d79a0f38aa28c7c" if verified else None,
            "source_kind": "third_party_derived_hardware" if verified else "user_supplied_table",
            "publisher_identity_verified": verified,
            "original_archive_independently_verified": False,
            "ordering": "preserved source row indices; acquisition chronology unverified",
            "source_noise_weights_used": False,
            "source_decoder_predictions_used": False,
        },
    )


def detector_anchors(circuit: stim.Circuit) -> dict[int, list[float]]:
    """Google coordinates list participating measurements, including final data qubits.

    For final-boundary checks, the oldest measurement's location identifies the
    stabilizer, while the latest time identifies the boundary. Taking the first data
    qubit would invent a new stabilizer identity exactly at the final round.
    """
    anchors: dict[int, list[float]] = {}
    coordinates = circuit.get_detector_coordinates()
    for detector in range(circuit.num_detectors):
        values = coordinates.get(detector, [])
        if not values or len(values) % 3 or not all(math.isfinite(c) for c in values):
            raise ValueError("pilot requires finite detector coordinates in x,y,t triples")
        triples = [values[start : start + 3] for start in range(0, len(values), 3)]
        earliest, latest = min(t[2] for t in triples), max(t[2] for t in triples)
        positions = {tuple(t[:2]) for t in triples if t[2] == earliest}
        if len(positions) != 1:
            raise ValueError("cannot unambiguously identify a stabilizer from source coordinates")
        x, y = next(iter(positions))
        anchors[detector] = [x, y, latest]
    return anchors


def detector_sequence(
    bits: NDArray[np.bool_], circuit: stim.Circuit
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Map detector identity explicitly; absent boundary checks are masked, not invented."""
    if bits.ndim != 2 or bits.shape[1] != circuit.num_detectors:
        raise ValueError("detector width does not match the source circuit")
    coordinates = detector_anchors(circuit)
    positions = sorted({tuple(value[:2]) for value in coordinates.values()})
    times = sorted({value[2] for value in coordinates.values()})
    position_ids = {value: index for index, value in enumerate(positions)}
    time_ids = {value: index for index, value in enumerate(times)}
    output = np.zeros((len(bits), len(times), 2 * len(positions)), dtype=np.float32)
    seen: set[tuple[int, int]] = set()
    mapping: list[list[int]] = []
    for detector in range(circuit.num_detectors):
        coordinate = coordinates[detector]
        ti, pi = time_ids[coordinate[2]], position_ids[tuple(coordinate[:2])]
        if (ti, pi) in seen:
            raise ValueError("two detectors share the same spatial and temporal coordinate")
        seen.add((ti, pi))
        output[:, ti, pi] = bits[:, detector]
        output[:, ti, pi + len(positions)] = 1
        mapping.append([ti, pi])
    return output, {
        "positions": [list(value) for value in positions],
        "times": times,
        "detector_to_sequence": mapping,
        "features": "detector bits followed by check-presence masks",
        "coordinate_convention": "oldest measurement stabilizer location and latest boundary time",
        "full_detector_coordinates": {
            str(key): value for key, value in circuit.get_detector_coordinates().items()
        },
    }
