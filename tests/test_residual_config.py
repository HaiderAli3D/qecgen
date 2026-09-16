"""Residual configuration: parsing, validation and the deterministic configuration hash."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest

from qecgen.residual.config import (
    ConfigError,
    DecoderKind,
    GenerationMode,
    ResidualConfig,
    SourceKind,
    SplitMethod,
    canonical_dict,
    chunk_sizes,
    config_hash,
    from_resolved_dict,
    load_config,
    parse_config,
    prefix_admissible,
    resolved_dict,
)
from qecgen.sampling import _chunk_sizes

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples" / "residual"
EXAMPLE_NAMES = (
    "indep_d9_r200_p0005",
    "indep_d25_r25_p0005",
    "device_static_d3_r3",
    "willow_d3_z_r10_si1000",
    "willow_d3_z_r10_rlprior",
)
SIMULATED_NAMES = EXAMPLE_NAMES[:3]
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def example_raw(name: str) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((EXAMPLES / f"{name}.json").read_text("utf-8"))
    return payload


def write(tmp_path: Path, raw: dict[str, Any], name: str = "config.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def legacy_raw() -> dict[str, Any]:
    """A minimal legacy config; the smallest shape every rule test can start from."""
    return {
        "version": 1,
        "dataset_name": "tiny_legacy",
        "output_root": "out",
        "source": {
            "kind": "legacy_ml_csv",
            "path": "src/tiny.ml.csv",
            "expected_content_hash": "ab" * 32,
        },
        "generation": {"mode": "extend", "shots": 192, "seed": 3, "chunk_size": 16},
        "decoder": {"kind": "circuit_dem"},
        "splits": {
            "method": "seeded_permutation",
            "seed": 7,
            "fractions": {"train": 0.7, "validation": 0.15, "test": 0.15},
        },
    }


# ---------------------------------------------------------------------------
# The five reviewed configs


class TestExamples:
    @pytest.mark.parametrize("name", EXAMPLE_NAMES)
    def test_round_trip_and_hash_stability(self, name: str) -> None:
        first = load_config(EXAMPLES / f"{name}.json", REPO_ROOT)
        second = load_config(EXAMPLES / f"{name}.json", REPO_ROOT)
        assert first.dataset_name == name
        assert first.version == 1
        digest = config_hash(first)
        assert HEX64.match(digest)
        assert digest == config_hash(second)
        assert first == second

    @pytest.mark.parametrize("name", EXAMPLE_NAMES)
    def test_hash_ignores_output_root(self, name: str, tmp_path: Path) -> None:
        raw = example_raw(name)
        reference = config_hash(load_config(EXAMPLES / f"{name}.json", REPO_ROOT))
        raw["output_root"] = "somewhere/else/entirely"
        moved = load_config(write(tmp_path, raw), REPO_ROOT)
        assert moved.output_root == (REPO_ROOT / "somewhere/else/entirely").resolve()
        assert config_hash(moved) == reference
        raw["output_root"] = str(tmp_path / "abs-root")
        assert config_hash(load_config(write(tmp_path, raw), REPO_ROOT)) == reference

    @pytest.mark.parametrize("name", EXAMPLE_NAMES)
    def test_hash_ignores_repo_root(self, name: str, tmp_path: Path) -> None:
        reference = config_hash(load_config(EXAMPLES / f"{name}.json", REPO_ROOT))
        other_checkout = tmp_path / "checkout"
        (other_checkout / "examples" / "residual").mkdir(parents=True)
        copied = other_checkout / "examples" / "residual" / f"{name}.json"
        copied.write_bytes((EXAMPLES / f"{name}.json").read_bytes())
        relocated = load_config(copied, other_checkout)
        assert config_hash(relocated) == reference
        assert relocated.output_root == (other_checkout / "data" / "residual").resolve()

    @pytest.mark.parametrize("name", SIMULATED_NAMES)
    def test_hash_changes_with_seed(self, name: str, tmp_path: Path) -> None:
        raw = example_raw(name)
        reference = config_hash(load_config(EXAMPLES / f"{name}.json", REPO_ROOT))
        raw["generation"]["seed"] += 1
        assert config_hash(load_config(write(tmp_path, raw), REPO_ROOT)) != reference

    def test_hash_changes_with_decoder_member(self, tmp_path: Path) -> None:
        si1000 = load_config(EXAMPLES / "willow_d3_z_r10_si1000.json", REPO_ROOT)
        rlprior = load_config(EXAMPLES / "willow_d3_z_r10_rlprior.json", REPO_ROOT)
        assert config_hash(si1000) != config_hash(rlprior)
        raw = example_raw("willow_d3_z_r10_si1000")
        raw["dataset_name"] = "willow_d3_z_r10_rlprior"
        raw["additional"] = True
        raw["decoder"]["member"] = rlprior.decoder.member
        raw["decoder"]["expected_sha256"] = rlprior.decoder.expected_sha256
        assert config_hash(load_config(write(tmp_path, raw), REPO_ROOT)) == config_hash(rlprior)

    def test_hash_is_independent_of_key_order_and_whitespace(self, tmp_path: Path) -> None:
        raw = example_raw("device_static_d3_r3")
        reference = config_hash(load_config(EXAMPLES / "device_static_d3_r3.json", REPO_ROOT))
        reordered = dict(reversed(list(raw.items())))
        path = tmp_path / "reordered.json"
        path.write_text(json.dumps(reordered, indent=None, separators=(",", ":")), "utf-8")
        assert config_hash(load_config(path, REPO_ROOT)) == reference

    def test_kinds_and_flags(self) -> None:
        d9 = load_config(EXAMPLES / "indep_d9_r200_p0005.json", REPO_ROOT)
        assert d9.source.kind is SourceKind.LEGACY_ML_CSV
        assert d9.decoder.kind is DecoderKind.CIRCUIT_DEM
        assert d9.generation.mode is GenerationMode.EXTEND
        assert d9.generation.require_source_prefix is True
        assert d9.splits.method is SplitMethod.SEEDED_PERMUTATION
        assert d9.additional is False
        assert d9.pipeline.checkpoint_rows == 16000
        assert d9.pipeline.feature_rows == 2000
        assert d9.sanity_model.enabled is True

        device = load_config(EXAMPLES / "device_static_d3_r3.json", REPO_ROOT)
        assert device.source.kind is SourceKind.DEVICE_ML_CSV
        assert device.decoder.kind is DecoderKind.STATIC_PROFILE_DEM

        rlprior = load_config(EXAMPLES / "willow_d3_z_r10_rlprior.json", REPO_ROOT)
        assert rlprior.additional is True
        assert rlprior.source.kind is SourceKind.HARDWARE_WILLOW
        assert rlprior.decoder.kind is DecoderKind.OFFICIAL_DEM
        assert rlprior.generation.mode is GenerationMode.SOURCE_ROWS
        assert rlprior.generation.shots is None
        assert rlprior.generation.seed is None
        assert rlprior.splits.method is SplitMethod.CONTIGUOUS_BLOCKS
        assert rlprior.splits.seed is None
        assert rlprior.source.zenodo is not None
        assert rlprior.source.zenodo.record == 13273331
        assert rlprior.source.formatted_prefix is not None
        assert rlprior.source.formatted_prefix.offset == 0
        assert rlprior.source.expected is not None
        assert rlprior.source.expected.orientation == "q10_7"

    @pytest.mark.parametrize("name", EXAMPLE_NAMES)
    def test_paths_resolve_under_repo_root(self, name: str) -> None:
        config = load_config(EXAMPLES / f"{name}.json", REPO_ROOT)
        assert config.repo_root == REPO_ROOT.resolve()
        assert config.output_root == (REPO_ROOT / "data" / "residual").resolve()
        for path in _all_paths(config):
            assert path.is_absolute()
            assert path.relative_to(REPO_ROOT.resolve())


def _all_paths(config: ResidualConfig) -> list[Path]:
    paths = [config.output_root]
    source = config.source
    for candidate in (source.path, source.table, source.circuit):
        if candidate is not None:
            paths.append(candidate)
    if source.formatted_prefix is not None:
        paths.append(source.formatted_prefix.path)
    if source.zenodo is not None:
        paths.append(source.zenodo.cache_dir)
    return paths


# ---------------------------------------------------------------------------
# Canonical and resolved forms


class TestSerialisation:
    @pytest.mark.parametrize("name", EXAMPLE_NAMES)
    def test_canonical_is_repo_relative_posix_without_output_root(self, name: str) -> None:
        config = load_config(EXAMPLES / f"{name}.json", REPO_ROOT)
        canonical = canonical_dict(config)
        assert "output_root" not in canonical
        assert "repo_root" not in canonical
        assert "config_path" not in canonical
        text = json.dumps(canonical, sort_keys=True, allow_nan=False)
        assert "\\" not in text
        assert str(REPO_ROOT.resolve()) not in text
        raw = example_raw(name)
        assert canonical["source"]["kind"] == raw["source"]["kind"]
        if "path" in raw["source"]:
            assert canonical["source"]["path"] == raw["source"]["path"]
        else:
            assert canonical["source"]["table"] == raw["source"]["table"]
            assert (
                canonical["source"]["zenodo"]["cache_dir"] == raw["source"]["zenodo"]["cache_dir"]
            )

    @pytest.mark.parametrize("name", EXAMPLE_NAMES)
    def test_resolved_round_trips_and_carries_hash(self, name: str, tmp_path: Path) -> None:
        config = load_config(EXAMPLES / f"{name}.json", REPO_ROOT)
        resolved = resolved_dict(config)
        assert resolved["config_hash"] == config_hash(config)
        assert resolved["output_root"] == config.output_root.as_posix()
        assert resolved["repo_root"] == config.repo_root.as_posix()
        assert resolved["config_path"] == (EXAMPLES / f"{name}.json").resolve().as_posix()
        assert list(resolved) == sorted(resolved)
        path = tmp_path / "resolved.json"
        path.write_text(json.dumps(resolved, sort_keys=True, allow_nan=False), "utf-8")
        again = from_resolved_dict(json.loads(path.read_text("utf-8")))
        assert again == config
        assert config_hash(again) == config_hash(config)
        assert resolved_dict(again) == resolved

    def test_resolved_refuses_tampered_hash(self) -> None:
        config = load_config(EXAMPLES / "indep_d9_r200_p0005.json", REPO_ROOT)
        resolved = resolved_dict(config)
        resolved["generation"]["seed"] = 99
        with pytest.raises(ConfigError, match="config_hash"):
            from_resolved_dict(resolved)

    def test_resolved_without_a_hash_is_refused(self) -> None:
        """A resolved config with no hash is not "unverified", it is a file this module
        did not write; accepting it would let a hand edit skip the hash check."""
        config = load_config(EXAMPLES / "indep_d9_r200_p0005.json", REPO_ROOT)
        resolved = resolved_dict(config)
        del resolved["config_hash"]
        with pytest.raises(ConfigError, match="config_hash is required"):
            from_resolved_dict(resolved)
        resolved = resolved_dict(config)
        resolved["config_hash"] = None
        with pytest.raises(ConfigError, match="config_hash"):
            from_resolved_dict(resolved)

    def test_absolute_paths_are_kept(self, tmp_path: Path) -> None:
        raw = legacy_raw()
        elsewhere = (tmp_path / "elsewhere" / "tiny.ml.csv").resolve()
        raw["source"]["path"] = str(elsewhere)
        config = parse_config(raw, REPO_ROOT)
        assert config.source.path == elsewhere
        assert config.output_root == (REPO_ROOT / "out").resolve()
        # Outside the checkout there is no repo-relative form; the absolute POSIX path is
        # hashed instead, so two checkouts pointing at one external file still agree.
        assert canonical_dict(config)["source"]["path"] == elsewhere.as_posix()

    def test_defaults_are_filled_in(self) -> None:
        config = parse_config(legacy_raw(), REPO_ROOT)
        assert config.additional is False
        assert config.pipeline.checkpoint_rows == 10_000
        assert config.pipeline.feature_rows == 2_000
        assert config.pipeline.spot_check_rows == 8
        assert config.sanity_model.enabled is True
        assert config.sanity_model.seed == 0
        assert config.decoder.enable_correlations is False
        assert config.generation.require_source_prefix is True
        assert config.config_path is None
        canonical = canonical_dict(config)
        assert canonical["pipeline"] == {
            "checkpoint_rows": 10_000,
            "feature_rows": 2_000,
            "spot_check_rows": 8,
        }
        assert canonical["additional"] is False


# ---------------------------------------------------------------------------
# Refusals


def _expect(raw: dict[str, Any], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        parse_config(raw, REPO_ROOT)


class TestRefusals:
    def test_unknown_top_level_key(self) -> None:
        raw = legacy_raw()
        raw["extra"] = 1
        _expect(raw, "Unknown configuration fields.*extra")

    def test_unknown_nested_key(self) -> None:
        raw = legacy_raw()
        raw["generation"]["workers"] = 4
        _expect(raw, "Unknown generation fields.*workers")
        raw = legacy_raw()
        raw["splits"]["fractions"]["holdout"] = 0.0
        _expect(raw, "holdout")

    def test_version_must_be_one(self) -> None:
        raw = legacy_raw()
        raw["version"] = 2
        _expect(raw, "version")
        raw["version"] = True
        _expect(raw, "version")

    def test_fractions_must_sum_to_one(self) -> None:
        raw = legacy_raw()
        raw["splits"]["fractions"] = {"train": 0.7, "validation": 0.2, "test": 0.2}
        _expect(raw, "sum to 1")
        raw["splits"]["fractions"] = {"train": 0.7, "validation": 0.3, "test": 0.0}
        _expect(raw, "fractions.test")
        raw["splits"]["fractions"] = {"train": 0.7, "validation": 0.15, "test": 0.15 + 1e-12}
        parse_config(raw, REPO_ROOT)

    def test_split_seed_per_method(self) -> None:
        raw = legacy_raw()
        del raw["splits"]["seed"]
        _expect(raw, "splits.seed")
        raw = legacy_raw()
        raw["splits"]["method"] = "contiguous_blocks"
        _expect(raw, "splits.seed")

    def test_enable_correlations_refused(self) -> None:
        raw = legacy_raw()
        raw["decoder"]["enable_correlations"] = True
        _expect(raw, "separate")

    def test_official_dem_requires_member_and_sha(self) -> None:
        raw = example_raw("willow_d3_z_r10_si1000")
        del raw["decoder"]["member"]
        _expect(raw, "decoder.member")
        raw = example_raw("willow_d3_z_r10_si1000")
        del raw["decoder"]["expected_sha256"]
        _expect(raw, "decoder.expected_sha256")
        raw = example_raw("willow_d3_z_r10_si1000")
        raw["decoder"]["expected_sha256"] = "XYZ"
        _expect(raw, "decoder.expected_sha256")
        raw = legacy_raw()
        raw["decoder"]["member"] = "foo.dem"
        _expect(raw, "decoder.member")

    def test_decoder_kind_must_match_source_kind(self) -> None:
        raw = legacy_raw()
        raw["decoder"]["kind"] = "static_profile_dem"
        _expect(raw, "decoder.kind")
        raw = example_raw("device_static_d3_r3")
        raw["decoder"]["kind"] = "circuit_dem"
        _expect(raw, "decoder.kind")

    def test_generation_mode_rules(self) -> None:
        raw = legacy_raw()
        raw["generation"]["mode"] = "sideways"
        _expect(raw, "generation.mode")
        raw = legacy_raw()
        raw["generation"]["shots"] = 0
        _expect(raw, "generation.shots")
        raw = legacy_raw()
        del raw["generation"]["seed"]
        _expect(raw, "generation.seed")
        raw = legacy_raw()
        raw["generation"]["chunk_size"] = 0
        _expect(raw, "generation.chunk_size")
        raw = legacy_raw()
        raw["generation"] = {"mode": "source_rows", "chunk_size": 16, "seed": 1}
        _expect(raw, "generation.seed")
        raw = legacy_raw()
        raw["generation"] = {"mode": "source_rows", "chunk_size": 16}
        config = parse_config(raw, REPO_ROOT)
        assert config.generation.require_source_prefix is False
        raw = example_raw("willow_d3_z_r10_si1000")
        raw["generation"] = {"mode": "extend", "shots": 100000, "seed": 1, "chunk_size": 1000}
        _expect(raw, "hardware_willow")

    def test_checkpoint_rows_must_be_multiple_of_chunk_size(self) -> None:
        raw = legacy_raw()
        raw["pipeline"] = {"checkpoint_rows": 40}
        _expect(raw, "checkpoint_rows")
        raw["pipeline"] = {"checkpoint_rows": 8}
        _expect(raw, "checkpoint_rows")
        raw["pipeline"] = {"checkpoint_rows": 64, "feature_rows": 16, "spot_check_rows": 5}
        config = parse_config(raw, REPO_ROOT)
        assert config.pipeline.checkpoint_rows == 64
        raw["pipeline"] = {"spot_check_rows": 4}
        _expect(raw, "spot_check_rows")

    def test_dataset_name_is_a_safe_directory_name(self) -> None:
        raw = legacy_raw()
        raw["dataset_name"] = "has space"
        _expect(raw, "dataset_name")
        raw["dataset_name"] = "../escape"
        _expect(raw, "dataset_name")
        raw["dataset_name"] = ""
        _expect(raw, "dataset_name")

    def test_source_kind_fields(self) -> None:
        raw = legacy_raw()
        raw["source"]["kind"] = "hologram"
        _expect(raw, "source.kind")
        raw = legacy_raw()
        raw["source"]["table"] = "x.parquet"
        _expect(raw, "source.table")
        raw = legacy_raw()
        raw["source"]["expected_content_hash"] = "short"
        _expect(raw, "expected_content_hash")
        raw = example_raw("willow_d3_z_r10_si1000")
        del raw["source"]["zenodo"]
        _expect(raw, "source.zenodo")
        raw = example_raw("willow_d3_z_r10_si1000")
        raw["source"]["formatted_prefix"]["offset"] = -1
        _expect(raw, "offset")
        raw = example_raw("willow_d3_z_r10_si1000")
        raw["source"]["expected"]["basis"] = "Q"
        _expect(raw, "basis")

    def test_missing_required_blocks(self) -> None:
        for key in ("dataset_name", "output_root", "source", "generation", "decoder", "splits"):
            raw = legacy_raw()
            del raw[key]
            _expect(raw, key)

    def test_top_level_must_be_an_object(self, tmp_path: Path) -> None:
        path = tmp_path / "list.json"
        path.write_text("[1, 2]", "utf-8")
        with pytest.raises(ConfigError, match="object"):
            load_config(path, REPO_ROOT)

    def test_bool_is_not_an_integer(self) -> None:
        raw = legacy_raw()
        raw["generation"]["seed"] = True
        _expect(raw, "generation.seed")

    def test_trailing_newline_is_refused(self) -> None:
        """``re.match`` with ``$`` accepts a trailing newline; a digest or a directory
        name with one would hash and resolve differently from what the file shows."""
        raw = legacy_raw()
        raw["dataset_name"] = "tiny_legacy\n"
        _expect(raw, "dataset_name")
        raw = legacy_raw()
        raw["source"]["expected_content_hash"] = "ab" * 32 + "\n"
        _expect(raw, "expected_content_hash")
        raw = example_raw("willow_d3_z_r10_si1000")
        raw["decoder"]["expected_sha256"] = raw["decoder"]["expected_sha256"] + "\n"
        _expect(raw, "decoder.expected_sha256")

    def test_require_source_prefix_errors_name_the_block(self) -> None:
        raw = legacy_raw()
        raw["generation"]["require_source_prefix"] = 1
        _expect(raw, r"generation\.require_source_prefix must be a JSON boolean")
        raw = legacy_raw()
        raw["generation"] = {"mode": "source_rows", "chunk_size": 16, "require_source_prefix": "no"}
        _expect(raw, r"generation\.require_source_prefix must be a JSON boolean")
        raw = legacy_raw()
        raw["generation"] = {
            "mode": "fresh",
            "shots": 192,
            "seed": 3,
            "chunk_size": 16,
            "require_source_prefix": 0,
        }
        _expect(raw, r"generation\.require_source_prefix must be a JSON boolean")

    def test_parse_does_not_mutate_input(self) -> None:
        raw = legacy_raw()
        snapshot = copy.deepcopy(raw)
        parse_config(raw, REPO_ROOT)
        assert raw == snapshot


# ---------------------------------------------------------------------------
# Seed/chunk contract helpers


class TestChunkContract:
    @pytest.mark.parametrize(
        ("shots", "chunk"),
        [(0, 1), (1, 1), (5, 2), (16000, 100000), (304000, 16000), (300000, 2000), (7, 7), (8, 7)],
    )
    def test_chunk_sizes_matches_sampling(self, shots: int, chunk: int) -> None:
        assert list(chunk_sizes(shots, chunk)) == list(_chunk_sizes(shots, chunk))

    def test_chunk_sizes_validation(self) -> None:
        with pytest.raises(ValueError, match="shots"):
            chunk_sizes(-1, 4)
        with pytest.raises(ValueError, match="chunk_size"):
            chunk_sizes(4, 0)

    def test_prefix_admissible(self) -> None:
        # The d9/d25 sources recorded chunk_size 100000 but made a single sample(16000)
        # call; a new run at chunk 16000 reproduces that call, one at chunk 8000 does not.
        assert prefix_admissible(16000, 100000, 304000, 16000)
        assert not prefix_admissible(16000, 100000, 304000, 8000)
        # Device: one sample(2000) call is the prefix of 150 such calls.
        assert prefix_admissible(2000, 2000, 300000, 2000)
        # A source made of [32, 32] is a prefix of [32]*4 and not of [16]*8 or [64, 64].
        assert prefix_admissible(64, 32, 128, 32)
        assert not prefix_admissible(64, 32, 128, 16)
        assert not prefix_admissible(64, 32, 128, 64)
        # Shrinking the run below the source is never an extension.
        assert not prefix_admissible(64, 32, 32, 32)
        # A ragged final source chunk cannot be a prefix of anything longer.
        assert not prefix_admissible(50, 32, 96, 32)
        assert prefix_admissible(50, 32, 50, 32)
