"""Verified sources and row streams for the residual datasets.

Every source is identified by evidence the pipeline recomputes (content hash, circuit
sha256, b8 byte equality), never by a filename, and every generated extension is
admitted only when the seed/chunk contract reproduces the source as a prefix. The tests
build tiny d=3 sources in ``tmp_path`` through the same qecgen writers that produced the
real ones, so nothing here depends on ``data/``.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import stim

from qecgen.configuration import write_configured
from qecgen.dataset import StreamingContentHasher
from qecgen.environments import build_single_environment
from qecgen.exporters.ml_csv import MLCSVExporter, read_manifest_only
from qecgen.residual.config import (
    DecoderKind,
    ResidualConfig,
    SourceKind,
    ZenodoConfig,
    parse_config,
)
from qecgen.residual.decoder import MultiObservableError
from qecgen.residual.sources import (
    ResolvedSource,
    SourceIdentity,
    iter_ml_csv_rows,
    resolve_source,
    source_prefix_hash,
)
from qecgen.sampling import packed_width, unpack_bits

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SI1000_MEMBER = "decoding_results/correlated_matching_decoder_with_si1000_prior/error_model.dem"
RL_MEMBER = "decoding_results/correlated_matching_decoder_with_rl_optimized_prior/error_model.dem"
ROOT = "google_105Q_surface_code_d3_d5_d7"
PREFIX = f"{ROOT}/d3_at_q10_7/Z/r2/"


def _noise_block(name: str) -> dict[str, Any]:
    payload = json.loads((EXAMPLES / name).read_text(encoding="utf-8"))
    noise = payload["noise"]
    assert isinstance(noise, dict)
    return noise


def _collect(rows: Iterator[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    dets, obs = zip(*list(rows), strict=True)
    return np.concatenate(dets), np.concatenate(obs)


# ---------------------------------------------------------------------------
# Legacy sources


def _legacy_source(tmp_path: Path, *, shots: int = 64, chunk_size: int = 32) -> Path:
    dataset = build_single_environment(
        distance=3, p=0.01, shots=shots, seed=1, chunk_size=chunk_size
    )
    path = tmp_path / "legacy.ml.csv"
    MLCSVExporter().write(dataset, path)
    return path


def _legacy_config(
    tmp_path: Path,
    path: Path,
    *,
    mode: str = "source_rows",
    shots: int | None = None,
    chunk_size: int = 32,
    seed: int = 1,
    expected_hash: str | None = None,
) -> ResidualConfig:
    manifest = read_manifest_only(path)
    generation: dict[str, Any] = {"mode": mode, "chunk_size": chunk_size}
    if mode != "source_rows":
        generation.update(shots=shots, seed=seed)
    raw = {
        "version": 1,
        "dataset_name": "legacy_test",
        "output_root": str(tmp_path / "out"),
        "source": {
            "kind": "legacy_ml_csv",
            "path": str(path),
            "expected_content_hash": expected_hash or manifest["content_hash"],
        },
        "generation": generation,
        "decoder": {"kind": "circuit_dem"},
        "splits": {
            "method": "seeded_permutation",
            "seed": 3,
            "fractions": {"train": 0.7, "validation": 0.15, "test": 0.15},
        },
        "pipeline": {"checkpoint_rows": chunk_size, "feature_rows": 8},
        "sanity_model": {"enabled": False},
    }
    return parse_config(raw, tmp_path)


class TestIterMlCsvRows:
    def test_round_trips_and_verifies_the_content_hash(self, tmp_path: Path) -> None:
        path = _legacy_source(tmp_path)
        manifest = read_manifest_only(path)
        expected = MLCSVExporter().read(path)

        chunks = list(iter_ml_csv_rows(path, manifest, 24))

        # 64 rows at 24 per chunk: two full chunks and a ragged 16-row tail.
        assert [c[0].shape[0] for c in chunks] == [24, 24, 16]
        dets = np.concatenate([c[0] for c in chunks])
        obs = np.concatenate([c[1] for c in chunks])
        assert dets.dtype == np.uint8 and obs.dtype == np.uint8
        assert np.array_equal(dets, expected.detectors)
        assert np.array_equal(obs, expected.observables)

    def test_rows_are_packed_little_endian(self, tmp_path: Path) -> None:
        path = _legacy_source(tmp_path)
        manifest = read_manifest_only(path)
        dets, _ = _collect(iter_ml_csv_rows(path, manifest, 64))
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            first = next(reader)
        cells = [first[header.index(f"detector_{i:02d}")] for i in range(manifest["n_detectors"])]
        bits = np.asarray([cell == "1" for cell in cells])
        assert bits.any(), "pick a row with at least one fired detector"
        assert np.array_equal(unpack_bits(dets[:1], manifest["n_detectors"])[0], bits)
        big = np.unpackbits(dets[:1], axis=1, count=manifest["n_detectors"], bitorder="big")[0]
        assert not np.array_equal(big.astype(bool), bits)

    def test_corrupted_cell_fails_the_hash_check_at_end_of_stream(self, tmp_path: Path) -> None:
        path = _legacy_source(tmp_path)
        manifest = read_manifest_only(path)
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        header = lines[0].rstrip("\n").split(",")
        column = header.index("detector_00")
        cells = lines[5].rstrip("\n").split(",")
        cells[column] = "1" if cells[column] == "0" else "0"
        lines[5] = ",".join(cells) + "\n"
        path.write_text("".join(lines), encoding="utf-8")

        stream = iter_ml_csv_rows(path, manifest, 16)
        # Rows stream without complaint; only exhausting the stream compares the digest.
        for _ in range(4):
            next(stream)
        with pytest.raises(ValueError, match="content_hash"):
            next(stream)

    def test_reordered_rows_are_refused(self, tmp_path: Path) -> None:
        path = _legacy_source(tmp_path)
        manifest = read_manifest_only(path)
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        lines[2], lines[3] = lines[3], lines[2]
        path.write_text("".join(lines), encoding="utf-8")
        with pytest.raises(ValueError, match="row"):
            list(iter_ml_csv_rows(path, manifest, 16))

    def test_truncated_table_is_refused(self, tmp_path: Path) -> None:
        path = _legacy_source(tmp_path)
        manifest = read_manifest_only(path)
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text("".join(lines[:-1]), encoding="utf-8")
        with pytest.raises(ValueError, match="63"):
            list(iter_ml_csv_rows(path, manifest, 16))

    def test_manifest_hash_disagreement_is_refused(self, tmp_path: Path) -> None:
        path = _legacy_source(tmp_path)
        manifest = dict(read_manifest_only(path))
        manifest["content_hash"] = "0" * 64
        with pytest.raises(ValueError, match="content_hash"):
            list(iter_ml_csv_rows(path, manifest, 16))


class TestSourcePrefixHash:
    def test_equals_streaming_hasher_over_the_first_rows(self, tmp_path: Path) -> None:
        path = _legacy_source(tmp_path)
        manifest = read_manifest_only(path)
        dets, obs = _collect(iter_ml_csv_rows(path, manifest, 64))
        hasher = StreamingContentHasher()
        hasher.update(dets[:40], obs[:40])
        expected = hasher.hexdigest(40, manifest["n_detectors"], manifest["n_observables"])

        # Chunks of 24 straddle the 40-row boundary: 24 + 16 of the second chunk.
        rows = iter_ml_csv_rows(path, manifest, 24)
        digest = source_prefix_hash(rows, 40, manifest["n_detectors"], manifest["n_observables"])
        assert digest == expected

    def test_short_stream_is_refused(self, tmp_path: Path) -> None:
        path = _legacy_source(tmp_path)
        manifest = read_manifest_only(path)
        with pytest.raises(ValueError, match="64"):
            source_prefix_hash(
                iter_ml_csv_rows(path, manifest, 16),
                100,
                manifest["n_detectors"],
                manifest["n_observables"],
            )


class TestLegacySource:
    def test_resolves_identity_and_decoder(self, tmp_path: Path) -> None:
        path = _legacy_source(tmp_path)
        manifest = read_manifest_only(path)
        config = _legacy_config(tmp_path, path)

        source = resolve_source(config)

        assert isinstance(source, ResolvedSource)
        identity = source.identity
        assert isinstance(identity, SourceIdentity)
        assert identity.kind is SourceKind.LEGACY_ML_CSV
        assert identity.distance == 3 and identity.rounds == 3 and identity.basis == "Z"
        assert identity.rotated is True
        assert identity.noise_model == "stim_uniform_circuit_level"
        assert identity.noise_parameters["p"] == 0.01
        assert identity.n_detectors == manifest["n_detectors"] == source.decoder.n_detectors
        assert identity.n_observables == 1
        assert identity.shots == 64 and identity.seed == 1 and identity.chunk_size == 32
        assert identity.hashes["content_hash"] == manifest["content_hash"]
        assert identity.circuit_sha256 == source.decoder.provenance["circuit_sha256"]
        assert identity.paths["manifest"].endswith("legacy.ml.manifest.json")
        assert source.decoder.kind is DecoderKind.CIRCUIT_DEM
        assert source.decoder.dem_sha256 == identity.hashes["dem_sha256"]
        assert source.circuit_for_coords.num_detectors == identity.n_detectors
        assert identity.dem_available.startswith("exact")
        payload = json.dumps(identity.to_dict(), allow_nan=False)
        assert json.loads(payload)["kind"] == "legacy_ml_csv"

    def test_rebuilt_circuit_matches_the_provenance_text(self, tmp_path: Path) -> None:
        dataset = build_single_environment(distance=3, p=0.01, shots=8, seed=1, chunk_size=8)
        path = tmp_path / "legacy.ml.csv"
        MLCSVExporter().write(dataset, path)
        source = resolve_source(_legacy_config(tmp_path, path))
        assert str(source.decoder.circuit) == dataset.meta.environments[0].circuit

    def test_source_rows_match_the_file(self, tmp_path: Path) -> None:
        path = _legacy_source(tmp_path)
        expected = MLCSVExporter().read(path)
        source = resolve_source(_legacy_config(tmp_path, path))
        dets, obs = _collect(source.iter_source_rows(20))
        assert np.array_equal(dets, expected.detectors)
        assert np.array_equal(obs, expected.observables)

    def test_expected_hash_disagreement_is_refused(self, tmp_path: Path) -> None:
        path = _legacy_source(tmp_path)
        config = _legacy_config(tmp_path, path, expected_hash="a" * 64)
        with pytest.raises(ValueError, match="expected_content_hash"):
            resolve_source(config)

    def test_extension_reproduces_the_source_as_a_prefix(self, tmp_path: Path) -> None:
        path = _legacy_source(tmp_path)
        manifest = read_manifest_only(path)
        config = _legacy_config(tmp_path, path, mode="extend", shots=128, chunk_size=32)

        source = resolve_source(config)
        rows = source.iter_generated_rows(128, 1, 32)
        digest = source_prefix_hash(rows, 64, manifest["n_detectors"], manifest["n_observables"])

        assert digest == manifest["content_hash"]

    def test_inadmissible_chunk_size_is_refused_before_sampling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _legacy_source(tmp_path)
        config = _legacy_config(tmp_path, path, mode="extend", shots=128, chunk_size=16)

        def no_sampling(*_: object, **__: object) -> Iterator[object]:
            raise AssertionError("the sampler was invoked before the prefix rule was applied")

        monkeypatch.setattr("qecgen.residual.sources.iter_chunks", no_sampling)
        with pytest.raises(ValueError, match="prefix"):
            resolve_source(config)

    def test_seed_disagreement_is_refused(self, tmp_path: Path) -> None:
        path = _legacy_source(tmp_path)
        config = _legacy_config(tmp_path, path, mode="extend", shots=128, chunk_size=32, seed=2)
        with pytest.raises(ValueError, match="seed"):
            resolve_source(config)

    def test_fewer_shots_than_the_source_is_refused(self, tmp_path: Path) -> None:
        path = _legacy_source(tmp_path)
        config = _legacy_config(tmp_path, path, mode="extend", shots=32, chunk_size=32)
        with pytest.raises(ValueError, match="prefix"):
            resolve_source(config)


# ---------------------------------------------------------------------------
# Device sources


def _device_raw(
    noise: dict[str, Any], output: Path, *, shots: int, chunk_size: int
) -> dict[str, Any]:
    return {
        "version": 1,
        "mode": "device",
        "output": {"path": str(output), "format": "ml_csv", "structure": "coords"},
        "sampling": {"shots": shots, "seed": 7, "chunk_size": chunk_size},
        "circuit": {"distance": 3, "rounds": 3, "basis": "z", "rotated": True},
        "parameter_provenance": {"kind": "scenario", "description": "test scenario"},
        "noise": noise,
    }


def _device_config(
    tmp_path: Path,
    path: Path,
    *,
    decoder_kind: str,
    mode: str = "source_rows",
    shots: int | None = None,
    chunk_size: int = 16,
    seed: int = 7,
) -> ResidualConfig:
    manifest = read_manifest_only(path)
    generation: dict[str, Any] = {"mode": mode, "chunk_size": chunk_size}
    if mode != "source_rows":
        generation.update(shots=shots, seed=seed)
    raw = {
        "version": 1,
        "dataset_name": "device_test",
        "output_root": str(tmp_path / "out"),
        "source": {
            "kind": "device_ml_csv",
            "path": str(path),
            "expected_content_hash": manifest["content_hash"],
        },
        "generation": generation,
        "decoder": {"kind": decoder_kind},
        "splits": {
            "method": "seeded_permutation",
            "seed": 3,
            "fractions": {"train": 0.7, "validation": 0.15, "test": 0.15},
        },
        "pipeline": {"checkpoint_rows": chunk_size, "feature_rows": 8},
        "sanity_model": {"enabled": False},
    }
    return parse_config(raw, tmp_path)


class TestDeviceSource:
    def test_static_profile_decoder_matches_the_manifest_audit(self, tmp_path: Path) -> None:
        path = tmp_path / "device.ml.csv"
        write_configured(
            _device_raw(_noise_block("device-static.json"), path, shots=32, chunk_size=16), path
        )
        manifest = read_manifest_only(path)
        config = _device_config(tmp_path, path, decoder_kind="static_profile_dem")

        source = resolve_source(config)

        assert source.decoder.kind is DecoderKind.STATIC_PROFILE_DEM
        assert (
            source.decoder.provenance["circuit_sha256"]
            == manifest["generation_audit"]["circuit_sha256"]
        )
        assert source.identity.kind is SourceKind.DEVICE_ML_CSV
        assert source.identity.noise_model == "device_profile"
        assert source.identity.noise_parameters["profile"]["probabilities"]["measurement"] == 0.01
        assert source.identity.seed == 7 and source.identity.chunk_size == 16
        assert source.identity.shots == 32
        assert (
            source.identity.hashes["profile_sha256"]
            == (manifest["generation_audit"]["profile_sha256"])
        )
        assert source.identity.dem_available.startswith("exact")
        expected = MLCSVExporter().read(path)
        dets, obs = _collect(source.iter_source_rows(10))
        assert np.array_equal(dets, expected.detectors)
        assert np.array_equal(obs, expected.observables)

    def test_static_extension_reproduces_the_source_prefix(self, tmp_path: Path) -> None:
        path = tmp_path / "device.ml.csv"
        write_configured(
            _device_raw(_noise_block("device-static.json"), path, shots=32, chunk_size=16), path
        )
        manifest = read_manifest_only(path)
        config = _device_config(
            tmp_path, path, decoder_kind="static_profile_dem", mode="extend", shots=64
        )
        source = resolve_source(config)
        digest = source_prefix_hash(
            source.iter_generated_rows(64, 7, 16),
            32,
            manifest["n_detectors"],
            manifest["n_observables"],
        )
        assert digest == manifest["content_hash"]

    def test_dynamic_profile_gets_a_frozen_reference_and_no_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "dynamic.ml.csv"
        write_configured(
            _device_raw(_noise_block("device-dynamic.json"), path, shots=16, chunk_size=8), path
        )
        config = _device_config(tmp_path, path, decoder_kind="frozen_reference_dem", chunk_size=8)

        source = resolve_source(config)

        assert source.decoder.kind is DecoderKind.FROZEN_REFERENCE_DEM
        assert "stationary point" in source.decoder.provenance["transformation"]
        assert source.identity.dem_available.startswith("reference")
        assert any("bursts" in item for item in source.identity.provenance_limitations)

        extend = _device_config(
            tmp_path,
            path,
            decoder_kind="frozen_reference_dem",
            mode="extend",
            shots=32,
            chunk_size=8,
        )
        with pytest.raises(ValueError, match="dynamic"):
            resolve_source(extend)

    def test_decoder_kind_must_match_the_profile(self, tmp_path: Path) -> None:
        path = tmp_path / "device.ml.csv"
        write_configured(
            _device_raw(_noise_block("device-static.json"), path, shots=16, chunk_size=16), path
        )
        config = _device_config(tmp_path, path, decoder_kind="frozen_reference_dem")
        with pytest.raises(ValueError, match="static"):
            resolve_source(config)


class TestDeviceConfigSource:
    def test_fresh_stream_from_a_version_1_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / "device-dynamic.json"
        raw = _device_raw(
            _noise_block("device-dynamic.json"), tmp_path / "unused.h5", shots=16, chunk_size=8
        )
        raw["output"] = {"path": str(tmp_path / "unused.h5"), "format": "hdf5"}
        config_path.write_text(json.dumps(raw), encoding="utf-8")
        config = parse_config(
            {
                "version": 1,
                "dataset_name": "device_config_test",
                "output_root": str(tmp_path / "out"),
                "source": {"kind": "device_config", "path": str(config_path)},
                "generation": {"mode": "fresh", "shots": 16, "seed": 11, "chunk_size": 8},
                "decoder": {"kind": "frozen_reference_dem"},
                "splits": {
                    "method": "seeded_permutation",
                    "seed": 3,
                    "fractions": {"train": 0.7, "validation": 0.15, "test": 0.15},
                },
                "pipeline": {"checkpoint_rows": 8, "feature_rows": 8},
                "sanity_model": {"enabled": False},
            },
            tmp_path,
        )

        source = resolve_source(config)

        assert source.identity.kind is SourceKind.DEVICE_CONFIG
        assert source.identity.shots == 0
        assert source.decoder.kind is DecoderKind.FROZEN_REFERENCE_DEM
        dets, obs = _collect(source.iter_generated_rows(16, 11, 8))
        assert dets.shape == (16, packed_width(source.decoder.n_detectors))
        assert obs.shape == (16, 1)
        with pytest.raises(ValueError, match="no source rows"):
            list(source.iter_source_rows(8))


# ---------------------------------------------------------------------------
# Hardware (Willow) sources


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


def _willow_fixture(tmp_path: Path, *, observables: int = 1) -> dict[str, Any]:
    """A 6-row synthetic Willow-style mirror: parquet + ideal circuit + archive members."""
    ideal = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=2)
    if observables == 2:
        ideal = ideal + stim.Circuit("OBSERVABLE_INCLUDE(1) rec[-1]")
    noisy = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=2, after_clifford_depolarization=0.01
    )
    n_det = ideal.num_detectors
    rng = np.random.default_rng(5)
    bits = rng.integers(0, 2, size=(6, n_det), dtype=np.uint8).astype(bool)
    bits[0, :] = False
    bits[1, :] = False
    bits[1, 0] = True  # asymmetric byte: little-endian 0x01, big-endian 0x80
    truth = np.array([0, 1, 1, 0, 1, 0], dtype=bool)
    dets = np.packbits(bits, axis=1, bitorder="little")
    obs = np.packbits(truth.reshape(-1, 1), axis=1, bitorder="little")

    circuit_path = tmp_path / "mirror.stim"
    circuit_bytes = str(ideal).encode()
    circuit_path.write_bytes(circuit_bytes)
    table_path = tmp_path / "mirror.parquet"
    pq.write_table(
        pa.table(
            {
                "distance": pa.array([3] * 6, pa.int16()),
                "basis": pa.array(["Z"] * 6, pa.string()),
                "rounds": pa.array([2] * 6, pa.int16()),
                "orientation": pa.array(["q10_7"] * 6, pa.string()),
                "shot": pa.array(list(range(6)), pa.int32()),
                "detectors": pa.array(bits.tolist(), pa.list_(pa.bool_())),
                "observable": pa.array(truth.tolist(), pa.bool_()),
            }
        ),
        table_path,
    )
    data_coords, meas_coords = _qubit_partition(ideal)
    metadata = {
        "basis": "Z",
        "rounds": 2,
        "shots": 6,
        "distance": 3,
        "data_qubit_coords": data_coords,
        "meas_qubit_coords": meas_coords,
    }
    dem = noisy.detector_error_model(decompose_errors=True)
    predicted = np.packbits(
        np.array([0, 1, 0, 0, 1, 0], dtype=bool).reshape(-1, 1), axis=1, bitorder="little"
    )
    files: dict[str, bytes] = {
        "circuit_ideal.stim": circuit_bytes,
        "circuit_noisy_si1000.stim": str(noisy).encode(),
        "detection_events.b8": dets.tobytes(),
        "obs_flips_actual.b8": obs.tobytes(),
        "sweep_bits.b8": bytes(6),
        "metadata.json": json.dumps(metadata).encode(),
        "README/README.md": b"# readme\n",
    }
    for member in (SI1000_MEMBER, RL_MEMBER):
        files[member] = str(dem).encode()
        files[member.replace("error_model.dem", "obs_flips_predicted.b8")] = predicted.tobytes()
    expected = {
        "table_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
        "circuit_sha256": hashlib.sha256(circuit_bytes).hexdigest(),
        "distance": 3,
        "basis": "Z",
        "rounds": 2,
        "orientation": "q10_7",
    }
    return {
        "table": table_path,
        "circuit": circuit_path,
        "expected": expected,
        "files": files,
        "detectors": dets,
        "observables": obs,
        "dem_sha256": hashlib.sha256(str(dem).encode()).hexdigest(),
        "ideal": ideal,
    }


def _fake_fetch(files: dict[str, bytes], calls: list[ZenodoConfig]) -> Any:
    def fetch(zenodo: ZenodoConfig) -> dict[str, Any]:
        calls.append(zenodo)
        members = []
        for name, payload in files.items():
            target = zenodo.cache_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            members.append(
                {
                    "member": (zenodo.cohort_prefix if name != "README/README.md" else f"{ROOT}/")
                    + name,
                    "path": name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "uncompressed_size": len(payload),
                    "status": "fetched",
                }
            )
        return {
            "record": {"id": zenodo.record, "version": "1.0.0"},
            "archive": {"key": zenodo.archive, "md5_status": "not_reverified"},
            "members": members,
        }

    return fetch


def _willow_config(
    tmp_path: Path,
    fixture: dict[str, Any],
    *,
    member: str = SI1000_MEMBER,
    dem_sha256: str | None = None,
    formatted_prefix: dict[str, Any] | None = None,
) -> ResidualConfig:
    source: dict[str, Any] = {
        "kind": "hardware_willow",
        "table": str(fixture["table"]),
        "circuit": str(fixture["circuit"]),
        "expected": fixture["expected"],
        "zenodo": {
            "record": 13273331,
            "archive": "google_105Q_surface_code_d3_d5_d7.zip",
            "archive_md5_published": "21fa6ad35b395d838ebcdbc92e364a12",
            "cohort_prefix": PREFIX,
            "cache_dir": str(tmp_path / "zenodo"),
        },
    }
    if formatted_prefix is not None:
        source["formatted_prefix"] = formatted_prefix
    raw = {
        "version": 1,
        "dataset_name": "willow_test",
        "output_root": str(tmp_path / "out"),
        "source": source,
        "generation": {"mode": "source_rows", "chunk_size": 4},
        "decoder": {
            "kind": "official_dem",
            "member": member,
            "expected_sha256": dem_sha256 or fixture["dem_sha256"],
        },
        "splits": {
            "method": "contiguous_blocks",
            "fractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
        },
        "pipeline": {"checkpoint_rows": 4, "feature_rows": 4},
        "sanity_model": {"enabled": False},
    }
    return parse_config(raw, tmp_path)


class TestWillowSource:
    def test_resolves_with_a_stubbed_zenodo_fetch(self, tmp_path: Path) -> None:
        fixture = _willow_fixture(tmp_path)
        calls: list[ZenodoConfig] = []
        config = _willow_config(tmp_path, fixture)

        source = resolve_source(config, fetch_cohort=_fake_fetch(fixture["files"], calls))

        assert len(calls) == 1 and calls[0].record == 13273331
        assert source.identity.kind is SourceKind.HARDWARE_WILLOW
        assert source.identity.shots == 6
        assert source.identity.seed is None and source.identity.chunk_size is None
        assert source.identity.noise_model == "hardware"
        assert source.identity.n_detectors == fixture["ideal"].num_detectors
        assert source.identity.hashes["table_sha256"] == fixture["expected"]["table_sha256"]
        assert source.identity.hashes["dem_sha256"] == fixture["dem_sha256"]
        assert source.identity.dem_available.startswith("official")
        assert source.decoder.kind is DecoderKind.OFFICIAL_DEM
        assert source.decoder.dem_sha256 == fixture["dem_sha256"]
        assert source.decoder.provenance["third_party_fitting"] is None
        assert source.decoder.provenance["fitted_in_this_pipeline"] is False
        assert source.decoder.provenance["zenodo"]["member"]["path"] == SI1000_MEMBER
        assert "noisy_circuit_dem_cross_reference" in source.decoder.provenance
        assert source.verification["cohort"]["detectors_equal_cohort"] is True
        assert (tmp_path / "zenodo" / "detection_events.b8").exists()

        dets, obs = _collect(source.iter_source_rows(4))
        assert np.array_equal(dets, fixture["detectors"])
        assert np.array_equal(obs, fixture["observables"])
        with pytest.raises(ValueError, match="hardware"):
            next(source.iter_generated_rows(4, 0, 4))

    def test_rl_prior_discloses_third_party_fitting(self, tmp_path: Path) -> None:
        fixture = _willow_fixture(tmp_path)
        config = _willow_config(tmp_path, fixture, member=RL_MEMBER)
        source = resolve_source(config, fetch_cohort=_fake_fetch(fixture["files"], []))
        fitting = source.decoder.provenance["third_party_fitting"]
        assert fitting is not None
        assert fitting["overlap_with_this_cohort"] == "unknown"
        assert "13-cycle" in fitting["data"]

    def test_mismatched_b8_is_refused(self, tmp_path: Path) -> None:
        fixture = _willow_fixture(tmp_path)
        files = dict(fixture["files"])
        corrupted = bytearray(files["detection_events.b8"])
        corrupted[0] ^= 0x01
        files["detection_events.b8"] = bytes(corrupted)
        config = _willow_config(tmp_path, fixture)
        with pytest.raises(ValueError, match="detection_events"):
            resolve_source(config, fetch_cohort=_fake_fetch(files, []))

    def test_wrong_member_hash_is_refused(self, tmp_path: Path) -> None:
        fixture = _willow_fixture(tmp_path)
        config = _willow_config(tmp_path, fixture, dem_sha256="b" * 64)
        with pytest.raises(ValueError, match="expected_sha256"):
            resolve_source(config, fetch_cohort=_fake_fetch(fixture["files"], []))

    def test_two_observables_stop_the_source(self, tmp_path: Path) -> None:
        fixture = _willow_fixture(tmp_path, observables=2)
        config = _willow_config(tmp_path, fixture)
        with pytest.raises(MultiObservableError):
            resolve_source(config, fetch_cohort=_fake_fetch(fixture["files"], []))

    def test_formatted_prefix_is_checked_bit_for_bit(self, tmp_path: Path) -> None:
        fixture = _willow_fixture(tmp_path)
        formatted = tmp_path / "willow-shots.ml.csv"
        write_configured(
            {
                "version": 1,
                "mode": "hardware",
                "output": {"path": str(formatted), "format": "ml_csv", "structure": "coords"},
                "sampling": {"shots": 4, "seed": 0, "chunk_size": 4},
                "circuit": {"distance": 3, "rounds": 2, "basis": "z", "rotated": True},
                "hardware": {
                    "table": str(fixture["table"]),
                    "circuit": str(fixture["circuit"]),
                    "expected": fixture["expected"],
                    "offset": 1,
                },
            },
            formatted,
        )
        manifest = read_manifest_only(formatted)

        good = _willow_config(
            tmp_path,
            fixture,
            formatted_prefix={
                "path": str(formatted),
                "expected_content_hash": manifest["content_hash"],
                "offset": 1,
            },
        )
        source = resolve_source(good, fetch_cohort=_fake_fetch(fixture["files"], []))
        assert source.verification["formatted_prefix"]["rows"] == 4
        assert source.identity.hashes["formatted_prefix_content_hash"] == manifest["content_hash"]

        wrong_offset = _willow_config(
            tmp_path,
            fixture,
            formatted_prefix={
                "path": str(formatted),
                "expected_content_hash": manifest["content_hash"],
                "offset": 0,
            },
        )
        with pytest.raises(ValueError, match="formatted_prefix"):
            resolve_source(wrong_offset, fetch_cohort=_fake_fetch(fixture["files"], []))

        wrong_hash = _willow_config(
            tmp_path,
            fixture,
            formatted_prefix={
                "path": str(formatted),
                "expected_content_hash": "c" * 64,
                "offset": 1,
            },
        )
        with pytest.raises(ValueError, match="content_hash"):
            resolve_source(wrong_hash, fetch_cohort=_fake_fetch(fixture["files"], []))
