"""Residual dataset configuration: parsed once, refused loudly, hashed deterministically.

A residual dataset is identified by its *resolved* configuration, so the hash computed here
is what every checkpoint, decoder-metadata file and validation report is keyed on. Two
traps shape the module:

* **The hash must survive a move.** ``output_root`` is where artifacts land, not what they
  are, and an absolute path differs between checkouts and machines. The hashed form
  (:func:`canonical_dict`) therefore drops ``output_root``, ``repo_root`` and
  ``config_path`` and writes every remaining path repo-relative with POSIX separators; the
  published form (:func:`resolved_dict`) keeps the absolute paths a reader needs and carries
  the hash alongside so it can be recomputed and checked.
* **A silently ignored key is a silently wrong dataset.** A misspelled ``seed`` or a
  ``chunk_size`` in the wrong block would fall back to a default and produce a well-formed
  dataset that cannot be reproduced from the file it claims to extend. Every object refuses
  unknown keys (mirroring ``qecgen.noise._keys``) and every scalar is type-checked with
  ``type(value) is int`` so that JSON ``true`` never passes as ``1``.

The module also states the seed/chunk contract that makes an *extension* of an existing
qecgen dataset legitimate: :func:`chunk_sizes` is the sequence of ``sample()`` call sizes
a run makes, and :func:`prefix_admissible` says whether a source's sequence is a prefix of
a new run's. That rule, not ``source_shots % chunk_size``, is what decides reproducibility,
because the sampler's stream is consumed call by call (``CLAUDE.md``, "chunk_size is part
of the reproducibility contract").
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

CONFIG_VERSION = 1
DEFAULT_CHECKPOINT_ROWS = 10_000
DEFAULT_FEATURE_ROWS = 2_000
DEFAULT_SPOT_CHECK_ROWS = 8
MIN_SPOT_CHECK_ROWS = 5
"""The brief requires at least five independently recomputed rows per configuration."""

SPLIT_NAMES: tuple[str, ...] = ("calibration", "train", "validation", "test")
"""Every split a fraction may name, in the order blocks are assigned."""

_FRACTION_TOLERANCE = 1e-9
# Always applied with fullmatch: ``$`` matches before a trailing newline, so ``match``
# accepted "…\n" as a digest and as a directory name.
_DATASET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_BASES = frozenset({"X", "Z"})


class ConfigError(ValueError):
    """A configuration that must not produce any output."""


class SourceKind(enum.StrEnum):
    LEGACY_ML_CSV = "legacy_ml_csv"
    DEVICE_ML_CSV = "device_ml_csv"
    DEVICE_CONFIG = "device_config"
    HARDWARE_WILLOW = "hardware_willow"


class DecoderKind(enum.StrEnum):
    CIRCUIT_DEM = "circuit_dem"
    STATIC_PROFILE_DEM = "static_profile_dem"
    FROZEN_REFERENCE_DEM = "frozen_reference_dem"
    OFFICIAL_DEM = "official_dem"


class SplitMethod(enum.StrEnum):
    SEEDED_PERMUTATION = "seeded_permutation"
    CONTIGUOUS_BLOCKS = "contiguous_blocks"


class GenerationMode(enum.StrEnum):
    EXTEND = "extend"
    SOURCE_ROWS = "source_rows"
    FRESH = "fresh"


# Which decoder constructions a source kind can honestly support. A legacy stream has an
# exact circuit DEM and nothing else; a device profile has a static DEM or a frozen
# reference; Willow has only Google's shipped models. Any other pairing would be a
# decoder built from something the source never was.
_DECODERS_FOR_SOURCE: Mapping[SourceKind, frozenset[DecoderKind]] = MappingProxyType(
    {
        SourceKind.LEGACY_ML_CSV: frozenset({DecoderKind.CIRCUIT_DEM}),
        SourceKind.DEVICE_ML_CSV: frozenset(
            {DecoderKind.STATIC_PROFILE_DEM, DecoderKind.FROZEN_REFERENCE_DEM}
        ),
        SourceKind.DEVICE_CONFIG: frozenset(
            {DecoderKind.STATIC_PROFILE_DEM, DecoderKind.FROZEN_REFERENCE_DEM}
        ),
        SourceKind.HARDWARE_WILLOW: frozenset({DecoderKind.OFFICIAL_DEM}),
    }
)
_MODES_FOR_SOURCE: Mapping[SourceKind, frozenset[GenerationMode]] = MappingProxyType(
    {
        SourceKind.LEGACY_ML_CSV: frozenset(
            {GenerationMode.EXTEND, GenerationMode.SOURCE_ROWS, GenerationMode.FRESH}
        ),
        SourceKind.DEVICE_ML_CSV: frozenset(
            {GenerationMode.EXTEND, GenerationMode.SOURCE_ROWS, GenerationMode.FRESH}
        ),
        # A bare version-1 config has no rows: nothing to extend, nothing to decode as-is.
        SourceKind.DEVICE_CONFIG: frozenset({GenerationMode.FRESH}),
        # Hardware rows are never extended or resampled (brief: no synthetic Willow shots).
        SourceKind.HARDWARE_WILLOW: frozenset({GenerationMode.SOURCE_ROWS}),
    }
)


# ---------------------------------------------------------------------------
# Dataclasses (fields mirror the JSON; paths absolute; every rule already applied)


@dataclass(frozen=True)
class WillowExpected:
    table_sha256: str
    circuit_sha256: str
    distance: int
    basis: str
    rounds: int
    orientation: str


@dataclass(frozen=True)
class FormattedPrefix:
    path: Path
    expected_content_hash: str
    offset: int = 0


@dataclass(frozen=True)
class ZenodoConfig:
    record: int
    archive: str
    archive_md5_published: str
    cohort_prefix: str
    cache_dir: Path


@dataclass(frozen=True)
class SourceConfig:
    kind: SourceKind
    path: Path | None = None
    expected_content_hash: str | None = None
    table: Path | None = None
    circuit: Path | None = None
    expected: WillowExpected | None = None
    formatted_prefix: FormattedPrefix | None = None
    zenodo: ZenodoConfig | None = None


@dataclass(frozen=True)
class GenerationConfig:
    mode: GenerationMode
    chunk_size: int
    shots: int | None = None
    seed: int | None = None
    require_source_prefix: bool = False


@dataclass(frozen=True)
class DecoderConfig:
    kind: DecoderKind
    enable_correlations: bool = False
    member: str | None = None
    expected_sha256: str | None = None


@dataclass(frozen=True)
class SplitsConfig:
    method: SplitMethod
    fractions: Mapping[str, float]
    seed: int | None = None


@dataclass(frozen=True)
class PipelineConfig:
    checkpoint_rows: int = DEFAULT_CHECKPOINT_ROWS
    feature_rows: int = DEFAULT_FEATURE_ROWS
    spot_check_rows: int = DEFAULT_SPOT_CHECK_ROWS


@dataclass(frozen=True)
class SanityModelConfig:
    enabled: bool = True
    seed: int = 0


@dataclass(frozen=True)
class ResidualConfig:
    """One reviewed dataset configuration with every default resolved.

    ``repo_root`` and ``config_path`` are provenance, not identity: they are excluded from
    :func:`canonical_dict` so the hash is the same in every checkout, and kept in
    :func:`resolved_dict` so a reader can see where the absolute paths came from.
    """

    version: int
    dataset_name: str
    output_root: Path
    additional: bool
    source: SourceConfig
    generation: GenerationConfig
    decoder: DecoderConfig
    splits: SplitsConfig
    pipeline: PipelineConfig
    sanity_model: SanityModelConfig
    repo_root: Path
    config_path: Path | None = None


# ---------------------------------------------------------------------------
# Scalar validators. `type(value) is int` rather than isinstance: JSON `true` is an int
# subclass and would otherwise read as seed 1.


def _object(value: Any, where: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{where} must be a JSON object with string keys")
    if unknown := set(value) - allowed:
        raise ConfigError(f"Unknown {where} fields: {sorted(unknown)}")
    return value


def _require(obj: Mapping[str, Any], names: tuple[str, ...], where: str) -> None:
    if missing := [f"{where}.{name}" for name in names if name not in obj]:
        raise ConfigError(f"Missing required fields: {missing}")


def _forbid(obj: Mapping[str, Any], names: tuple[str, ...], where: str, reason: str) -> None:
    if present := [name for name in names if name in obj]:
        raise ConfigError(f"{where}.{present[0]} is not allowed {reason}")


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{where} must be an integer >= {minimum}, got {value!r}")
    return value


def _boolean(value: Any, where: str) -> bool:
    if type(value) is not bool:
        raise ConfigError(f"{where} must be a JSON boolean, got {value!r}")
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a nonempty string")
    return value


def _hex64(value: Any, where: str) -> str:
    text = _text(value, where)
    if not _HEX64.fullmatch(text):
        raise ConfigError(f"{where} must be a lowercase 64-character hex digest")
    return text


def _hex32(value: Any, where: str) -> str:
    text = _text(value, where)
    if not re.fullmatch(r"[0-9a-f]{32}", text):
        raise ConfigError(f"{where} must be a lowercase 32-character hex digest (MD5)")
    return text


def _fraction(value: Any, where: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
        raise ConfigError(f"{where} must be a number in (0, 1], got {value!r}")
    return float(value)


def _enum[E: enum.StrEnum](kind: type[E], value: Any, where: str) -> E:
    text = _text(value, where)
    try:
        return kind(text)
    except ValueError:
        choices = ", ".join(member.value for member in kind)
        raise ConfigError(f"{where} must be one of {choices}, got {text!r}") from None


def _path(value: Any, where: str, repo_root: Path) -> Path:
    """Resolve a config path: relative to the repo root, absolute kept as given.

    ``resolve()`` rather than ``absolute()`` so ``..`` segments and Windows drive-letter
    case collapse to one spelling; otherwise two spellings of one file hash differently.
    """
    text = _text(value, where)
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def _relative(path: Path, repo_root: Path) -> str:
    """The hashed spelling of a path.

    Repo-relative POSIX when the file lives in the checkout, which is every reviewed
    config. A path outside the checkout has no such form, so its absolute POSIX spelling
    is hashed instead — two checkouts on one machine still agree, but that hash is not
    portable across machines, which is the price of pointing outside the tree.
    """
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


# ---------------------------------------------------------------------------
# Block parsers


def _parse_source(raw: Any, repo_root: Path) -> SourceConfig:
    obj = _object(
        raw,
        "source",
        {
            "kind",
            "path",
            "expected_content_hash",
            "table",
            "circuit",
            "expected",
            "formatted_prefix",
            "zenodo",
        },
    )
    _require(obj, ("kind",), "source")
    kind = _enum(SourceKind, obj["kind"], "source.kind")
    file_keys = ("path", "expected_content_hash")
    hardware_keys = ("table", "circuit", "expected", "formatted_prefix", "zenodo")
    if kind is SourceKind.HARDWARE_WILLOW:
        _forbid(obj, file_keys, "source", f"for kind {kind.value}")
        _require(obj, ("table", "circuit", "expected", "zenodo"), "source")
        expected = _object(
            obj["expected"],
            "source.expected",
            {"table_sha256", "circuit_sha256", "distance", "basis", "rounds", "orientation"},
        )
        _require(
            expected,
            ("table_sha256", "circuit_sha256", "distance", "basis", "rounds", "orientation"),
            "source.expected",
        )
        basis = _text(expected["basis"], "source.expected.basis")
        if basis not in _BASES:
            raise ConfigError(f"source.expected.basis must be X or Z, got {basis!r}")
        willow = WillowExpected(
            table_sha256=_hex64(expected["table_sha256"], "source.expected.table_sha256"),
            circuit_sha256=_hex64(expected["circuit_sha256"], "source.expected.circuit_sha256"),
            distance=_integer(expected["distance"], "source.expected.distance", minimum=3),
            basis=basis,
            rounds=_integer(expected["rounds"], "source.expected.rounds", minimum=1),
            orientation=_text(expected["orientation"], "source.expected.orientation"),
        )
        prefix: FormattedPrefix | None = None
        if "formatted_prefix" in obj:
            raw_prefix = _object(
                obj["formatted_prefix"],
                "source.formatted_prefix",
                {"path", "expected_content_hash", "offset"},
            )
            _require(raw_prefix, ("path", "expected_content_hash"), "source.formatted_prefix")
            prefix = FormattedPrefix(
                path=_path(raw_prefix["path"], "source.formatted_prefix.path", repo_root),
                expected_content_hash=_hex64(
                    raw_prefix["expected_content_hash"],
                    "source.formatted_prefix.expected_content_hash",
                ),
                offset=_integer(raw_prefix.get("offset", 0), "source.formatted_prefix.offset"),
            )
        raw_zenodo = _object(
            obj["zenodo"],
            "source.zenodo",
            {"record", "archive", "archive_md5_published", "cohort_prefix", "cache_dir"},
        )
        _require(
            raw_zenodo,
            ("record", "archive", "archive_md5_published", "cohort_prefix", "cache_dir"),
            "source.zenodo",
        )
        zenodo = ZenodoConfig(
            record=_integer(raw_zenodo["record"], "source.zenodo.record", minimum=1),
            archive=_text(raw_zenodo["archive"], "source.zenodo.archive"),
            archive_md5_published=_hex32(
                raw_zenodo["archive_md5_published"], "source.zenodo.archive_md5_published"
            ),
            cohort_prefix=_text(raw_zenodo["cohort_prefix"], "source.zenodo.cohort_prefix"),
            cache_dir=_path(raw_zenodo["cache_dir"], "source.zenodo.cache_dir", repo_root),
        )
        return SourceConfig(
            kind=kind,
            table=_path(obj["table"], "source.table", repo_root),
            circuit=_path(obj["circuit"], "source.circuit", repo_root),
            expected=willow,
            formatted_prefix=prefix,
            zenodo=zenodo,
        )

    _forbid(obj, hardware_keys, "source", f"for kind {kind.value}")
    _require(obj, ("path",), "source")
    if kind is SourceKind.DEVICE_CONFIG:
        # A version-1 config file has no rows and therefore no content hash to expect.
        _forbid(obj, ("expected_content_hash",), "source", f"for kind {kind.value}")
        return SourceConfig(kind=kind, path=_path(obj["path"], "source.path", repo_root))
    _require(obj, ("expected_content_hash",), "source")
    return SourceConfig(
        kind=kind,
        path=_path(obj["path"], "source.path", repo_root),
        expected_content_hash=_hex64(obj["expected_content_hash"], "source.expected_content_hash"),
    )


def _parse_generation(raw: Any, source_kind: SourceKind) -> GenerationConfig:
    obj = _object(
        raw, "generation", {"mode", "shots", "seed", "chunk_size", "require_source_prefix"}
    )
    _require(obj, ("mode", "chunk_size"), "generation")
    mode = _enum(GenerationMode, obj["mode"], "generation.mode")
    if mode not in _MODES_FOR_SOURCE[source_kind]:
        allowed = ", ".join(sorted(m.value for m in _MODES_FOR_SOURCE[source_kind]))
        raise ConfigError(
            f"generation.mode {mode.value!r} is not available for source kind "
            f"{source_kind.value!r} (allowed: {allowed})"
        )
    chunk_size = _integer(obj["chunk_size"], "generation.chunk_size", minimum=1)
    if mode is GenerationMode.SOURCE_ROWS:
        _forbid(obj, ("shots", "seed"), "generation", "in source_rows mode (no sampling)")
        prefix = _boolean(
            obj.get("require_source_prefix", False), "generation.require_source_prefix"
        )
        if prefix:
            raise ConfigError("generation.require_source_prefix has no meaning in source_rows mode")
        return GenerationConfig(mode=mode, chunk_size=chunk_size)
    _require(obj, ("shots", "seed"), "generation")
    shots = _integer(obj["shots"], "generation.shots", minimum=1)
    seed = _integer(obj["seed"], "generation.seed")
    if mode is GenerationMode.FRESH:
        prefix = _boolean(
            obj.get("require_source_prefix", False), "generation.require_source_prefix"
        )
        if prefix:
            raise ConfigError(
                "generation.require_source_prefix has no meaning in fresh mode: "
                "a fresh stream extends nothing"
            )
        return GenerationConfig(mode=mode, chunk_size=chunk_size, shots=shots, seed=seed)
    prefix = _boolean(obj.get("require_source_prefix", True), "generation.require_source_prefix")
    return GenerationConfig(
        mode=mode, chunk_size=chunk_size, shots=shots, seed=seed, require_source_prefix=prefix
    )


def _parse_decoder(raw: Any, source_kind: SourceKind) -> DecoderConfig:
    obj = _object(raw, "decoder", {"kind", "enable_correlations", "member", "expected_sha256"})
    _require(obj, ("kind",), "decoder")
    kind = _enum(DecoderKind, obj["kind"], "decoder.kind")
    if kind not in _DECODERS_FOR_SOURCE[source_kind]:
        allowed = ", ".join(sorted(k.value for k in _DECODERS_FOR_SOURCE[source_kind]))
        raise ConfigError(
            f"decoder.kind {kind.value!r} cannot be built from source kind "
            f"{source_kind.value!r} (allowed: {allowed})"
        )
    if _boolean(obj.get("enable_correlations", False), "decoder.enable_correlations"):
        raise ConfigError(
            "decoder.enable_correlations must be false: schema v1 records standard matching "
            "only, so results stay comparable with the previous PyMatching baseline. "
            "Correlated matching is a separate named decoder configuration and never "
            "shares a dataset with standard matching."
        )
    if kind is DecoderKind.OFFICIAL_DEM:
        _require(obj, ("member", "expected_sha256"), "decoder")
        return DecoderConfig(
            kind=kind,
            member=_text(obj["member"], "decoder.member"),
            expected_sha256=_hex64(obj["expected_sha256"], "decoder.expected_sha256"),
        )
    _forbid(obj, ("member", "expected_sha256"), "decoder", f"for kind {kind.value}")
    return DecoderConfig(kind=kind)


def _parse_splits(raw: Any) -> SplitsConfig:
    obj = _object(raw, "splits", {"method", "seed", "fractions"})
    _require(obj, ("method", "fractions"), "splits")
    method = _enum(SplitMethod, obj["method"], "splits.method")
    raw_fractions = _object(obj["fractions"], "splits.fractions", set(SPLIT_NAMES))
    if not raw_fractions:
        raise ConfigError("splits.fractions must name at least one split")
    fractions = {
        name: _fraction(raw_fractions[name], f"splits.fractions.{name}")
        for name in SPLIT_NAMES
        if name in raw_fractions
    }
    total = sum(fractions.values())
    if abs(total - 1.0) > _FRACTION_TOLERANCE:
        raise ConfigError(f"splits.fractions must sum to 1 (got {total!r})")
    if method is SplitMethod.SEEDED_PERMUTATION:
        _require(obj, ("seed",), "splits")
        seed: int | None = _integer(obj["seed"], "splits.seed")
    else:
        _forbid(obj, ("seed",), "splits", "for contiguous_blocks (nothing is permuted)")
        seed = None
    return SplitsConfig(method=method, fractions=MappingProxyType(fractions), seed=seed)


def _parse_pipeline(raw: Any, chunk_size: int) -> PipelineConfig:
    obj = _object(raw, "pipeline", {"checkpoint_rows", "feature_rows", "spot_check_rows"})
    checkpoint_rows = _integer(
        obj.get("checkpoint_rows", DEFAULT_CHECKPOINT_ROWS), "pipeline.checkpoint_rows", minimum=1
    )
    # A checkpoint holds whole sample() calls. If it did not, one call's rows would
    # straddle two checkpoint files, and resuming would have to replay half a call —
    # which the stream cannot do without re-sampling everything before it.
    if checkpoint_rows % chunk_size != 0:
        raise ConfigError(
            f"pipeline.checkpoint_rows ({checkpoint_rows}) must be a positive multiple of "
            f"generation.chunk_size ({chunk_size}) so every checkpoint holds whole sample() calls"
        )
    return PipelineConfig(
        checkpoint_rows=checkpoint_rows,
        feature_rows=_integer(
            obj.get("feature_rows", DEFAULT_FEATURE_ROWS), "pipeline.feature_rows", minimum=1
        ),
        spot_check_rows=_integer(
            obj.get("spot_check_rows", DEFAULT_SPOT_CHECK_ROWS),
            "pipeline.spot_check_rows",
            minimum=MIN_SPOT_CHECK_ROWS,
        ),
    )


def _parse_sanity(raw: Any) -> SanityModelConfig:
    obj = _object(raw, "sanity_model", {"enabled", "seed"})
    return SanityModelConfig(
        enabled=_boolean(obj.get("enabled", True), "sanity_model.enabled"),
        seed=_integer(obj.get("seed", 0), "sanity_model.seed"),
    )


# ---------------------------------------------------------------------------
# Public API


def parse_config(
    raw: Mapping[str, Any], repo_root: Path, *, config_path: Path | None = None
) -> ResidualConfig:
    """Validate a decoded JSON object and resolve every default and path.

    Relative paths resolve under ``repo_root`` (the checkout, not the config file's
    directory) so a reviewed config reads the same wherever it is copied, and so the hashed
    repo-relative spelling is exactly what the file said.
    """
    obj = _object(
        raw,
        "configuration",
        {
            "version",
            "dataset_name",
            "output_root",
            "additional",
            "source",
            "generation",
            "decoder",
            "splits",
            "pipeline",
            "sanity_model",
        },
    )
    _require(
        obj,
        ("version", "dataset_name", "output_root", "source", "generation", "decoder", "splits"),
        "configuration",
    )
    version = _integer(obj["version"], "version", minimum=1)
    if version != CONFIG_VERSION:
        raise ConfigError(f"version must be {CONFIG_VERSION}, got {version}")
    dataset_name = _text(obj["dataset_name"], "dataset_name")
    if not _DATASET_NAME.fullmatch(dataset_name):
        # It becomes a directory name and every artifact's file prefix.
        raise ConfigError(
            "dataset_name must match [A-Za-z0-9][A-Za-z0-9_-]* (it names a directory), "
            f"got {dataset_name!r}"
        )
    root = repo_root.resolve()
    source = _parse_source(obj["source"], root)
    generation = _parse_generation(obj["generation"], source.kind)
    return ResidualConfig(
        version=version,
        dataset_name=dataset_name,
        output_root=_path(obj["output_root"], "output_root", root),
        additional=_boolean(obj.get("additional", False), "additional"),
        source=source,
        generation=generation,
        decoder=_parse_decoder(obj["decoder"], source.kind),
        splits=_parse_splits(obj["splits"]),
        pipeline=_parse_pipeline(obj.get("pipeline", {}), generation.chunk_size),
        sanity_model=_parse_sanity(obj.get("sanity_model", {})),
        repo_root=root,
        config_path=None if config_path is None else config_path.resolve(),
    )


def load_config(path: Path, repo_root: Path) -> ResidualConfig:
    """Read one reviewed JSON config. Existence of the files it names is *not* checked
    here: a missing Willow table must surface as a *blocked* dataset from the pipeline,
    after inventory, not as a config error that stops ``build-all`` before it starts."""
    try:
        raw = json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a JSON object at the top level")
    return parse_config(raw, repo_root, config_path=path)


def _dump(config: ResidualConfig, spell: Any) -> dict[str, Any]:
    """The JSON form shared by both serialisations; ``spell`` decides how paths are written."""
    source = config.source
    source_dict: dict[str, Any] = {"kind": source.kind.value}
    if source.path is not None:
        source_dict["path"] = spell(source.path)
    if source.expected_content_hash is not None:
        source_dict["expected_content_hash"] = source.expected_content_hash
    if source.table is not None:
        source_dict["table"] = spell(source.table)
    if source.circuit is not None:
        source_dict["circuit"] = spell(source.circuit)
    if source.expected is not None:
        source_dict["expected"] = {
            "table_sha256": source.expected.table_sha256,
            "circuit_sha256": source.expected.circuit_sha256,
            "distance": source.expected.distance,
            "basis": source.expected.basis,
            "rounds": source.expected.rounds,
            "orientation": source.expected.orientation,
        }
    if source.formatted_prefix is not None:
        source_dict["formatted_prefix"] = {
            "path": spell(source.formatted_prefix.path),
            "expected_content_hash": source.formatted_prefix.expected_content_hash,
            "offset": source.formatted_prefix.offset,
        }
    if source.zenodo is not None:
        source_dict["zenodo"] = {
            "record": source.zenodo.record,
            "archive": source.zenodo.archive,
            "archive_md5_published": source.zenodo.archive_md5_published,
            "cohort_prefix": source.zenodo.cohort_prefix,
            "cache_dir": spell(source.zenodo.cache_dir),
        }
    generation: dict[str, Any] = {
        "mode": config.generation.mode.value,
        "chunk_size": config.generation.chunk_size,
        "require_source_prefix": config.generation.require_source_prefix,
    }
    if config.generation.shots is not None:
        generation["shots"] = config.generation.shots
    if config.generation.seed is not None:
        generation["seed"] = config.generation.seed
    decoder: dict[str, Any] = {
        "kind": config.decoder.kind.value,
        "enable_correlations": config.decoder.enable_correlations,
    }
    if config.decoder.member is not None:
        decoder["member"] = config.decoder.member
    if config.decoder.expected_sha256 is not None:
        decoder["expected_sha256"] = config.decoder.expected_sha256
    splits: dict[str, Any] = {
        "method": config.splits.method.value,
        "fractions": dict(config.splits.fractions),
    }
    if config.splits.seed is not None:
        splits["seed"] = config.splits.seed
    return {
        "version": config.version,
        "dataset_name": config.dataset_name,
        "additional": config.additional,
        "source": source_dict,
        "generation": generation,
        "decoder": decoder,
        "splits": splits,
        "pipeline": {
            "checkpoint_rows": config.pipeline.checkpoint_rows,
            "feature_rows": config.pipeline.feature_rows,
            "spot_check_rows": config.pipeline.spot_check_rows,
        },
        "sanity_model": {
            "enabled": config.sanity_model.enabled,
            "seed": config.sanity_model.seed,
        },
    }


def _sorted(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    return result


def canonical_dict(config: ResidualConfig) -> dict[str, Any]:
    """The hashed form: repo-relative POSIX paths, no output root, no checkout identity."""
    return _sorted(_dump(config, lambda path: _relative(path, config.repo_root)))


def resolved_dict(config: ResidualConfig) -> dict[str, Any]:
    """The published form (``<name>_resolved_config.json``): absolute POSIX paths, the
    output root, the checkout they were resolved against and the hash itself, so
    :func:`from_resolved_dict` can rebuild the config and re-check the hash."""
    payload = _dump(config, lambda path: path.as_posix())
    payload["output_root"] = config.output_root.as_posix()
    payload["repo_root"] = config.repo_root.as_posix()
    payload["config_path"] = None if config.config_path is None else config.config_path.as_posix()
    payload["config_hash"] = config_hash(config)
    return _sorted(payload)


def from_resolved_dict(payload: Mapping[str, Any]) -> ResidualConfig:
    """Rebuild a config from :func:`resolved_dict` output and re-verify its hash.

    The absolute paths pass through :func:`parse_config` unchanged, and ``repo_root`` makes
    them relative again for hashing — which is how a validation run recomputes the hash of
    a published dataset from its own metadata rather than trusting the recorded value.
    The hash is mandatory: a resolved config without one is not "unverified", it is a
    file this module did not write, and accepting it would let a hand-edited config pass
    the very check that exists to catch hand edits.
    """
    obj = dict(payload)
    repo_root = _path(obj.pop("repo_root", None), "repo_root", Path.cwd())
    raw_config_path = obj.pop("config_path", None)
    if "config_hash" not in obj:
        raise ConfigError("config_hash is required in a resolved configuration")
    recorded = obj.pop("config_hash")
    config_path = None if raw_config_path is None else Path(_text(raw_config_path, "config_path"))
    config = parse_config(obj, repo_root, config_path=config_path)
    if _hex64(recorded, "config_hash") != config_hash(config):
        raise ConfigError(
            f"config_hash {recorded} does not match the configuration it accompanies "
            f"(recomputed {config_hash(config)})"
        )
    return config


def config_hash(config: ResidualConfig) -> str:
    """SHA-256 of the canonical JSON: identical across output roots and checkouts."""
    text = json.dumps(
        canonical_dict(config), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_sizes(shots: int, chunk_size: int) -> tuple[int, ...]:
    """The ``sample()`` call sizes a run of ``shots`` at ``chunk_size`` makes.

    Restated here rather than imported from ``qecgen.sampling._chunk_sizes`` because that
    name is private and this module is the contract's public statement; the test suite pins
    the two together so they cannot drift.
    """
    if shots < 0:
        raise ValueError(f"shots must be >= 0, got {shots}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    full, rest = divmod(shots, chunk_size)
    return (chunk_size,) * full + ((rest,) if rest else ())


def prefix_admissible(source_shots: int, source_chunk: int, new_shots: int, new_chunk: int) -> bool:
    """Whether a new run reproduces the source's rows as its prefix.

    The sampler is consumed one ``sample(n)`` call at a time, so equality of the first
    ``source_shots`` rows requires the source's whole call-size sequence to be the start of
    the new one. ``source_shots % new_chunk == 0`` is *not* the rule: the d9 source recorded
    ``chunk_size 100000`` yet made one ``sample(16000)`` call, which only a new run at chunk
    16000 repeats.
    """
    source = chunk_sizes(source_shots, source_chunk)
    new = chunk_sizes(new_shots, new_chunk)
    return len(source) <= len(new) and new[: len(source)] == source
