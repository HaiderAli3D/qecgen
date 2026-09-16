"""Checkpoint store: crash-safe chunk files, checksums, resume and identity refusal."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from qecgen.residual.checkpoint import (
    CheckpointIdentity,
    CheckpointIdentityError,
    ChunkIntegrityError,
    ChunkRangeError,
    ChunkStore,
)

N_DETECTORS = 24
N_OBSERVABLES = 1


def _identity(**overrides: Any) -> CheckpointIdentity:
    fields: dict[str, Any] = {
        "config_hash": "c" * 64,
        "source_hash": "s" * 64,
        "decoder_dem_sha256": "d" * 64,
        "graph_digest": "g" * 64,
        "schema_version": 1,
        "versions": {"stim": "1.16.0", "pymatching": "2.4.0", "numpy": "2.3.3"},
        "seed": 0,
        "chunk_size": 32,
        "shots": 96,
    }
    fields.update(overrides)
    return CheckpointIdentity(**fields)


def _rows(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Packed rows with a deliberately asymmetric byte so a bit-order flip would show."""
    rng = np.random.default_rng(seed)
    detectors = rng.integers(0, 256, size=(n, N_DETECTORS // 8), dtype=np.uint8)
    detectors[:, 0] = 0b0000_0001
    observables = rng.integers(0, 2, size=(n, 1), dtype=np.uint8)
    return detectors, observables


def _write_three(store: ChunkStore, rows: int = 32) -> None:
    for index in range(3):
        detectors, observables = _rows(rows, seed=index)
        store.write_raw(index, index * rows, detectors, observables)


def _names(root: Path) -> set[str]:
    return {path.name for path in root.iterdir()}


class TestRawChunks:
    def test_three_chunks_survive_reopen_with_matching_hashes(self, tmp_path: Path) -> None:
        store = ChunkStore(tmp_path / "ck", _identity())
        _write_three(store)
        first = store.completed_raw()
        assert sorted(first) == [0, 1, 2]

        reopened = ChunkStore(tmp_path / "ck", _identity())
        second = reopened.completed_raw()
        assert second == first
        for index, record in second.items():
            assert record.index == index
            assert record.row_start == index * 32
            assert record.row_count == 32
            assert len(record.raw_sha256) == 64
            assert record.feat_sha256 is None
            assert record.pm_wrong_count is None

    def test_only_chk_files_and_index_exist(self, tmp_path: Path) -> None:
        """np.savez given a *path* appends .npz and os.replace then targets a file that
        does not exist; the handle form is the only thing that keeps the extension."""
        root = tmp_path / "ck"
        _write_three(ChunkStore(root, _identity()))
        names = _names(root)
        assert names == {
            "raw_chunk_00000.chk",
            "raw_chunk_00001.chk",
            "raw_chunk_00002.chk",
            "checkpoint.json",
        }
        assert not list(root.glob("*.npz"))
        assert not list(root.glob("*.tmp"))

    def test_read_raw_round_trips_exact_bytes(self, tmp_path: Path) -> None:
        store = ChunkStore(tmp_path / "ck", _identity())
        detectors, observables = _rows(16, seed=7)
        store.write_raw(0, 0, detectors, observables)
        got_detectors, got_observables = ChunkStore(tmp_path / "ck", _identity()).read_raw(0)
        assert got_detectors.dtype == np.uint8
        assert got_observables.dtype == np.uint8
        assert np.array_equal(got_detectors, detectors)
        assert np.array_equal(got_observables, observables)

    def test_tampered_chunk_file_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "ck"
        store = ChunkStore(root, _identity())
        _write_three(store)
        path = root / "raw_chunk_00001.chk"
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 0xFF
        path.write_bytes(bytes(data))
        with pytest.raises(ChunkIntegrityError, match=r"raw_chunk_00001.chk"):
            store.read_raw(1)

    def test_missing_chunk_file_on_reopen_is_refused(self, tmp_path: Path) -> None:
        """The index is written after the file, so an indexed chunk with no file is
        external damage, not a crash window; a resume must not pretend it is complete."""
        root = tmp_path / "ck"
        _write_three(ChunkStore(root, _identity()))
        (root / "raw_chunk_00002.chk").unlink()
        with pytest.raises(ChunkIntegrityError, match=r"raw_chunk_00002.chk"):
            ChunkStore(root, _identity())

    def test_unindexed_chunk_file_is_rewritten_not_trusted(self, tmp_path: Path) -> None:
        """A crash between os.replace of the chunk and the index rewrite leaves a file
        the index does not know about; it is simply recomputed."""
        root = tmp_path / "ck"
        store = ChunkStore(root, _identity())
        detectors, observables = _rows(8, seed=1)
        store.write_raw(0, 0, detectors, observables)
        index_path = root / "checkpoint.json"
        saved = index_path.read_bytes()
        store.write_raw(1, 8, detectors, observables)
        index_path.write_bytes(saved)

        reopened = ChunkStore(root, _identity())
        assert sorted(reopened.completed_raw()) == [0]
        reopened.write_raw(1, 8, detectors, observables)
        assert sorted(reopened.completed_raw()) == [0, 1]

    def test_rejects_wrong_dtype_and_row_mismatch(self, tmp_path: Path) -> None:
        store = ChunkStore(tmp_path / "ck", _identity())
        detectors, observables = _rows(8, seed=1)
        with pytest.raises(ValueError, match="uint8"):
            store.write_raw(0, 0, detectors.astype(np.int64), observables)
        with pytest.raises(ValueError, match="rows"):
            store.write_raw(0, 0, detectors, observables[:4])
        with pytest.raises(ValueError, match="rows"):
            store.write_raw(0, 0, detectors[:0], observables[:0])

    def test_packed_width_must_agree_across_chunks(self, tmp_path: Path) -> None:
        store = ChunkStore(tmp_path / "ck", _identity())
        detectors, observables = _rows(8, seed=1)
        store.write_raw(0, 0, detectors, observables)
        wider = np.zeros((8, detectors.shape[1] + 1), dtype=np.uint8)
        with pytest.raises(ValueError, match="width"):
            store.write_raw(1, 8, wider, observables)


class TestRowRanges:
    def test_overlapping_row_start_is_refused(self, tmp_path: Path) -> None:
        store = ChunkStore(tmp_path / "ck", _identity())
        detectors, observables = _rows(32, seed=1)
        store.write_raw(0, 0, detectors, observables)
        with pytest.raises(ChunkRangeError, match="overlap"):
            store.write_raw(1, 16, detectors, observables)
        assert sorted(store.completed_raw()) == [0]
        assert not list((tmp_path / "ck").glob("raw_chunk_00001*"))

    def test_same_index_with_different_range_is_refused(self, tmp_path: Path) -> None:
        store = ChunkStore(tmp_path / "ck", _identity())
        detectors, observables = _rows(32, seed=1)
        store.write_raw(0, 0, detectors, observables)
        with pytest.raises(ChunkRangeError, match="index 0"):
            store.write_raw(0, 32, detectors, observables)

    def test_rewriting_same_index_and_range_drops_stale_features(self, tmp_path: Path) -> None:
        store = ChunkStore(tmp_path / "ck", _identity())
        detectors, observables = _rows(8, seed=1)
        store.write_raw(0, 0, detectors, observables)
        _write_features(store, 0, 8)
        assert sorted(store.completed_features()) == [0]
        store.write_raw(0, 0, detectors, observables)
        assert store.completed_features() == {}
        assert not (tmp_path / "ck" / "feat_chunk_00000.chk").exists()

    def test_contiguous_completed_stops_at_first_gap(self, tmp_path: Path) -> None:
        store = ChunkStore(tmp_path / "ck", _identity())
        detectors, observables = _rows(8, seed=1)
        store.write_raw(0, 0, detectors, observables)
        store.write_raw(1, 8, detectors, observables)
        store.write_raw(3, 24, detectors, observables)
        assert store.contiguous_completed() == (0, 1)
        assert store.contiguous_completed(stage="features") == ()
        _write_features(store, 0, 8)
        assert store.contiguous_completed(stage="features") == (0,)

    def test_contiguous_completed_requires_row_contiguity(self, tmp_path: Path) -> None:
        store = ChunkStore(tmp_path / "ck", _identity())
        detectors, observables = _rows(8, seed=1)
        store.write_raw(0, 0, detectors, observables)
        store.write_raw(1, 16, detectors, observables)
        with pytest.raises(ChunkRangeError, match="contiguous"):
            store.contiguous_completed()


def _write_features(store: ChunkStore, index: int, n: int) -> None:
    rng = np.random.default_rng(index)
    features = rng.random((n, 24))
    pm_guess = rng.integers(0, 2, size=n, dtype=np.uint8)
    truth = rng.integers(0, 2, size=n, dtype=np.uint8)
    pm_wrong = (pm_guess != truth).astype(np.uint8)
    store.write_features(index, features, pm_guess, rng.random(n), truth, pm_wrong)


class TestFeatureChunks:
    def test_features_round_trip_and_count_positives(self, tmp_path: Path) -> None:
        store = ChunkStore(tmp_path / "ck", _identity())
        detectors, observables = _rows(16, seed=3)
        store.write_raw(0, 0, detectors, observables)
        rng = np.random.default_rng(5)
        features = rng.random((16, 24))
        pm_guess = rng.integers(0, 2, size=16, dtype=np.uint8)
        pm_weight = rng.random(16)
        truth = rng.integers(0, 2, size=16, dtype=np.uint8)
        pm_wrong = (pm_guess != truth).astype(np.uint8)
        record = store.write_features(0, features, pm_guess, pm_weight, truth, pm_wrong)
        assert record.pm_wrong_count == int(pm_wrong.sum())
        assert record.feat_sha256 is not None and len(record.feat_sha256) == 64

        reopened = ChunkStore(tmp_path / "ck", _identity())
        assert reopened.completed_features()[0] == record
        got = reopened.read_features(0)
        assert set(got) == {"features", "pm_guess", "pm_weight", "truth", "pm_wrong"}
        assert np.array_equal(got["features"], features)
        assert got["features"].dtype == np.float64
        assert np.array_equal(got["pm_guess"], pm_guess)
        assert np.array_equal(got["pm_weight"], pm_weight)
        assert np.array_equal(got["truth"], truth)
        assert np.array_equal(got["pm_wrong"], pm_wrong)
        assert got["pm_wrong"].dtype == np.uint8
        assert reopened.aggregate() == {
            "raw_rows": 16,
            "feature_rows": 16,
            "pm_wrong_total": int(pm_wrong.sum()),
        }

    def test_features_require_raw_chunk_and_matching_rows(self, tmp_path: Path) -> None:
        store = ChunkStore(tmp_path / "ck", _identity())
        with pytest.raises(ChunkRangeError, match="no raw chunk"):
            _write_features(store, 0, 8)
        detectors, observables = _rows(8, seed=1)
        store.write_raw(0, 0, detectors, observables)
        with pytest.raises(ValueError, match="rows"):
            _write_features(store, 0, 9)

    def test_features_refuse_non_binary_and_inconsistent_pm_wrong(self, tmp_path: Path) -> None:
        store = ChunkStore(tmp_path / "ck", _identity())
        detectors, observables = _rows(4, seed=1)
        store.write_raw(0, 0, detectors, observables)
        features = np.zeros((4, 24))
        ones = np.ones(4, dtype=np.uint8)
        zeros = np.zeros(4, dtype=np.uint8)
        with pytest.raises(ValueError, match="binary"):
            store.write_features(0, features, ones * 2, np.zeros(4), zeros, ones)
        with pytest.raises(ValueError, match="pm_wrong"):
            store.write_features(0, features, ones, np.zeros(4), zeros, zeros)

    def test_tampered_feature_file_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "ck"
        store = ChunkStore(root, _identity())
        detectors, observables = _rows(8, seed=1)
        store.write_raw(0, 0, detectors, observables)
        _write_features(store, 0, 8)
        path = root / "feat_chunk_00000.chk"
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 0xFF
        path.write_bytes(bytes(data))
        with pytest.raises(ChunkIntegrityError, match=r"feat_chunk_00000.chk"):
            store.read_features(0)


class TestIdentity:
    def test_different_decoder_is_refused_naming_the_field(self, tmp_path: Path) -> None:
        _write_three(ChunkStore(tmp_path / "ck", _identity()))
        with pytest.raises(CheckpointIdentityError) as info:
            ChunkStore(tmp_path / "ck", _identity(decoder_dem_sha256="e" * 64))
        assert info.value.fields == ("decoder_dem_sha256",)
        assert "decoder_dem_sha256" in str(info.value)
        assert "config_hash" not in str(info.value)

    def test_every_differing_field_is_named(self, tmp_path: Path) -> None:
        _write_three(ChunkStore(tmp_path / "ck", _identity()))
        with pytest.raises(CheckpointIdentityError) as info:
            ChunkStore(
                tmp_path / "ck",
                _identity(seed=1, versions={"stim": "1.15.0"}, schema_version=2),
            )
        assert info.value.fields == ("schema_version", "versions", "seed")

    def test_identity_survives_json_round_trip(self) -> None:
        identity = _identity(seed=None)
        payload = json.loads(json.dumps(identity.to_dict(), sort_keys=True))
        assert CheckpointIdentity.from_dict(payload) == identity

    def test_index_file_records_identity_and_chunks(self, tmp_path: Path) -> None:
        root = tmp_path / "ck"
        _write_three(ChunkStore(root, _identity()))
        payload = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
        assert payload["format"] == "qecgen-residual-checkpoint"
        assert payload["identity"] == _identity().to_dict()
        assert [chunk["index"] for chunk in payload["chunks"]] == [0, 1, 2]
        assert payload["updated_at"].endswith("+00:00")


class TestAtomicIndex:
    def test_index_never_left_half_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "ck"
        store = ChunkStore(root, _identity())
        detectors, observables = _rows(8, seed=1)
        store.write_raw(0, 0, detectors, observables)
        index_path = root / "checkpoint.json"
        before = index_path.read_bytes()

        real_replace = os.replace

        def failing_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
            if Path(dst).name == "checkpoint.json":
                raise OSError("simulated crash during index commit")
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", failing_replace)
        with pytest.raises(OSError, match="simulated crash"):
            store.write_raw(1, 8, detectors, observables)
        monkeypatch.undo()

        assert index_path.read_bytes() == before
        assert json.loads(before)["chunks"][0]["index"] == 0
        assert not list(root.glob("*.tmp"))
        reopened = ChunkStore(root, _identity())
        assert sorted(reopened.completed_raw()) == [0]

    def test_failed_chunk_write_leaves_no_temp_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "ck"
        store = ChunkStore(root, _identity())
        detectors, observables = _rows(8, seed=1)

        def failing_savez(*args: Any, **kwargs: Any) -> None:
            raise OSError("simulated disk full")

        monkeypatch.setattr("qecgen.residual.checkpoint.np.savez", failing_savez)
        with pytest.raises(OSError, match="disk full"):
            store.write_raw(0, 0, detectors, observables)
        monkeypatch.undo()
        assert _names(root) == set()
        assert store.completed_raw() == {}


class TestDiscard:
    def test_discard_removes_only_owned_files(self, tmp_path: Path) -> None:
        root = tmp_path / "ck"
        store = ChunkStore(root, _identity())
        _write_three(store)
        _write_features(store, 0, 32)
        foreign = root / "inventory.json"
        foreign.write_text("{}", encoding="utf-8")
        store.discard()
        assert _names(root) == {"inventory.json"}
        assert store.completed_raw() == {}
        assert store.contiguous_completed() == ()
        reopened = ChunkStore(root, _identity(decoder_dem_sha256="e" * 64))
        assert reopened.completed_raw() == {}
