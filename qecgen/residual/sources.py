"""Verified sources and row streams for the residual datasets.

A residual dataset extends, or re-decodes, rows that already exist somewhere else: a
legacy ``.ml.csv``, a device-profile ``.ml.csv``, or the Willow mirror. Every one of those
files is identified here by evidence the pipeline recomputes, never by its name — a
``data/`` directory holds several same-size look-alikes whose only difference is the
content hash. Three traps shape the module:

* **The table is never materialised.** The d=9 source is 16,000 rows of 16,000 one-column
  bits; :func:`iter_ml_csv_rows` packs each row as it is read and folds the packed bytes
  into a :class:`qecgen.dataset.StreamingContentHasher`, so the manifest's
  ``content_hash`` is verified at the end of the stream against what was actually
  consumed. A reader that stops early gets rows but no verdict, which is why the prefix
  comparison in :func:`source_prefix_hash` is its own digest rather than a flag.
* **An extension is admissible by the call-size rule, and it is refused before any
  sampling.** The sampler stream is consumed one ``sample(n)`` call at a time, so rows
  ``0..source_shots`` come out identical only when the source's call-size sequence is a
  prefix of the new run's (:func:`qecgen.residual.config.prefix_admissible`) *and* the seed
  is the same. :func:`resolve_source` applies that rule while resolving, so a config that
  cannot reproduce its source never spends a second sampling; the content-hash equality is
  still the ground truth and the pipeline re-checks it on the generated rows.
* **The decoder must describe the circuit the rows came from.** A static device profile's
  DEM is derived from the rebuilt noisy circuit, and its sha256 is asserted equal to the
  manifest's ``generation_audit.circuit_sha256``; the legacy rebuild cross-checks its
  channel vector against the manifest the way ``qa.benchmark_dataset`` does. A Willow
  cohort is verified byte-for-byte against the Zenodo members before Google's shipped DEM
  is accepted, and the network step is injectable so that verification is testable offline.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import stim

from qecgen.configuration import ideal_circuit, read_config
from qecgen.dataset import (
    SHOT_COLUMN,
    Contract,
    DatasetMeta,
    StreamingContentHasher,
    require_legacy_environment,
)
from qecgen.environments import DriftAxis, build_environment
from qecgen.exporters.bit_columns import (
    bits_from_cells,
    block_slices,
    require_row_in_order,
)
from qecgen.exporters.ml_csv import (
    SIDECAR_FORMAT,
    SIDECAR_VERSION,
    MLCSVExporter,
    _blocks,
    _header_row,
    read_manifest_only,
    read_provenance_only,
)
from qecgen.hardware import ImportedCohort, load_willow_derived
from qecgen.noise import NoiseProfile
from qecgen.residual.config import (
    DecoderKind,
    GenerationMode,
    ResidualConfig,
    SourceKind,
    WillowExpected,
    ZenodoConfig,
    prefix_admissible,
)
from qecgen.residual.decoder import (
    DecoderModel,
    MultiObservableError,
    from_circuit,
    from_frozen_reference,
    from_official_dem,
    from_static_profile,
)
from qecgen.residual.graph import time_slices
from qecgen.residual.zenodo import fetch_cohort, verify_cohort
from qecgen.sampling import iter_chunks, iter_profile_chunks, packed_width

__all__ = [
    "FetchCohort",
    "ResolvedSource",
    "RowChunk",
    "SourceIdentity",
    "default_fetch_cohort",
    "iter_ml_csv_rows",
    "resolve_source",
    "source_prefix_hash",
]

RowChunk = tuple[np.ndarray, np.ndarray]
"""``(detectors, observables)``: uint8, little-endian bit-packed, one row per shot."""

FetchCohort = Callable[[ZenodoConfig], dict[str, Any]]
"""The Zenodo step of a Willow resolution: fetch the cohort members into
``zenodo.cache_dir`` and return the receipts. Injectable so tests stay offline."""

CONTENT_HASH_ALGORITHM = "blake2b-256"

_RL_PRIOR_FITTING: dict[str, Any] = {
    "method": (
        "reinforcement-learning optimisation of the matching prior (Sivak et al., "
        "arXiv:2406.02700), performed by Google, not by this pipeline"
    ),
    "objective": "logical error rate of the decoder on same-device data",
    "data": (
        "optimised jointly for all distance-3 and distance-5 patches using the 13-cycle "
        "calibration data (archive README)"
    ),
    "overlap_with_this_cohort": "unknown",
    "logical_outcomes_entered_the_weights": True,
}
"""Disclosure carried by every RL-prior decoder. The phrase "no data-fitted weights" is
never used for it: logical outcomes of same-device data entered these weights, and the
overlap with the 10-cycle cohort decoded here is not documented by the publisher."""


def _third_party_fitting(member: str) -> dict[str, Any] | None:
    """What the publisher fitted into the shipped DEM, or ``None`` when nothing was.

    An unrecognised member is refused rather than recorded as "not fitted": a decoder
    whose provenance quietly defaults to the clean case is exactly the over-claim this
    field exists to prevent.
    """
    if "correlated_matching_decoder_with_rl_optimized_prior/" in member:
        return dict(_RL_PRIOR_FITTING)
    if "correlated_matching_decoder_with_si1000_prior/" in member:
        return None
    raise ValueError(
        f"no fitting disclosure is recorded for decoder member {member!r}; add one to "
        "qecgen.residual.sources before decoding with it"
    )


# ---------------------------------------------------------------------------
# Identity and resolved source


@dataclass(frozen=True)
class SourceIdentity:
    """Everything the brief asks to be recorded for a selected source.

    ``circuit_sha256`` follows the ``generation_audit.circuit_sha256`` convention
    (sha256 of ``str(circuit)``), so a device source can be checked against its own
    manifest; the sha256 of a circuit *file*, which differs for the Willow mirror, lives
    in ``hashes`` under its own key.
    """

    kind: SourceKind
    paths: dict[str, str]
    hashes: dict[str, str]
    distance: int
    rounds: int
    basis: str
    rotated: bool
    noise_model: str
    noise_parameters: dict[str, Any]
    n_detectors: int
    n_observables: int
    shots: int
    seed: int | None
    chunk_size: int | None
    circuit_sha256: str
    dem_available: str
    provenance_limitations: list[str]

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["kind"] = self.kind.value
        json.dumps(payload, allow_nan=False)
        return payload


@dataclass(frozen=True, eq=False)
class ResolvedSource:
    """A verified source, its frozen decoder, and the two row streams it can provide.

    ``eq=False`` for the same reason as :class:`DecoderModel`: identity is the recorded
    hashes, not the equality of matcher objects.
    """

    identity: SourceIdentity
    decoder: DecoderModel
    circuit_for_coords: stim.Circuit
    """The circuit whose detector coordinates define the time slices (``decoder.circuit``)."""

    config: ResidualConfig
    verification: dict[str, Any]
    """Cross-check reports gathered while resolving (cohort verification, prefix
    comparison, time-slice sizes); JSON-serialisable, for the inventory record."""

    _source_rows: Callable[[int], Iterator[RowChunk]] = field(repr=False)
    _generated_rows: Callable[[int, int, int], Iterator[RowChunk]] | None = field(repr=False)

    def iter_source_rows(self, chunk_rows: int) -> Iterator[RowChunk]:
        """The rows that already exist, in source order, ``chunk_rows`` at a time."""
        if chunk_rows < 1:
            raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")
        return self._source_rows(chunk_rows)

    def iter_generated_rows(self, shots: int, seed: int, chunk_size: int) -> Iterator[RowChunk]:
        """A seeded sampling stream (simulated kinds only).

        Hardware rows are never resampled: there is no generating model to sample from,
        and a synthetic Willow row presented beside a measured one would be a fabrication.
        """
        if self._generated_rows is None:
            raise ValueError(
                f"{self.identity.kind.value} is a hardware source; its rows cannot be "
                "generated, only read"
            )
        if shots < 0 or chunk_size < 1 or seed < 0:
            raise ValueError("shots must be >= 0, chunk_size >= 1 and seed >= 0")
        return self._generated_rows(shots, seed, chunk_size)


# ---------------------------------------------------------------------------
# Streaming ml_csv reader and prefix digest


def _load_sidecar(path: Path) -> dict[str, Any]:
    sidecar_path = MLCSVExporter().companions(path)[0]
    if not sidecar_path.is_file():
        raise ValueError(f"{path} has no manifest sidecar {sidecar_path.name}; not a qecgen table")
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != SIDECAR_FORMAT:
        raise ValueError(f"{sidecar_path} is not a {SIDECAR_FORMAT} sidecar")
    if payload.get("version") != SIDECAR_VERSION:
        raise ValueError(f"{sidecar_path} declares sidecar version {payload.get('version')!r}")
    columns = payload.get("columns")
    if not isinstance(columns, dict):
        raise ValueError(f"{sidecar_path} carries no columns block")
    return columns


def iter_ml_csv_rows(path: Path, manifest: dict[str, Any], chunk_rows: int) -> Iterator[RowChunk]:
    """Stream an ``.ml.csv`` table as packed chunks and verify its content hash at the end.

    Each row is packed the moment it is parsed, so the working set is one chunk of packed
    bytes rather than the unpacked table. The digest is compared to ``manifest``'s
    ``content_hash`` only once the file is exhausted — the same digest the exporter wrote
    from the packed arrays — so a corrupted cell, a truncated file or a wrong manifest
    surfaces as a refusal *after* the last chunk; a caller that needs a verdict must drain
    the stream. Multi-environment and Contract B tables are refused: their content hash
    folds in arrays this reader does not produce, so it could never verify them.
    """
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")
    n_detectors = int(manifest["n_detectors"])
    n_observables = int(manifest["n_observables"])
    declared_rows = int(manifest["shots"])
    expected_hash = manifest.get("content_hash")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"{path}: manifest carries no 64-hex content_hash to verify against")
    algorithm = manifest.get("content_hash_algorithm", CONTENT_HASH_ALGORITHM)
    if algorithm != CONTENT_HASH_ALGORITHM:
        raise ValueError(
            f"{path}: content_hash_algorithm {algorithm!r} is not {CONTENT_HASH_ALGORITHM}; "
            "this reader can only reproduce the BLAKE2b digest"
        )
    if manifest.get("bit_order", "little") != "little":
        raise ValueError(f"{path}: bit_order {manifest.get('bit_order')!r} is not little")

    columns = _load_sidecar(path)
    blocks = _blocks(columns)
    if blocks["environment"]:
        raise ValueError(
            f"{path}: a multi-environment table folds environment ids into its content "
            "hash; residual sources are single-environment"
        )
    if blocks["mechanism"]:
        raise ValueError(
            f"{path}: a Contract B table folds mechanism labels into its content hash; "
            "residual sources are Contract A"
        )
    if len(blocks["feature"]) != n_detectors or len(blocks["target"]) != n_observables:
        raise ValueError(
            f"{path}: the sidecar names {len(blocks['feature'])} detector and "
            f"{len(blocks['target'])} observable columns but the manifest declares "
            f"{n_detectors} and {n_observables}"
        )
    expected_header = _header_row(columns)
    at = block_slices({block: len(names) for block, names in blocks.items()})
    feature_at, target_at = at["feature"], at["target"]

    hasher = StreamingContentHasher()
    rows = 0
    det_rows: list[np.ndarray] = []
    obs_rows: list[np.ndarray] = []

    def flush() -> RowChunk:
        dets = np.stack(det_rows).astype(np.uint8, copy=False)
        obs = np.stack(obs_rows).astype(np.uint8, copy=False)
        det_rows.clear()
        obs_rows.clear()
        hasher.update(dets, obs)
        return dets, obs

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path} is empty")
        if header != expected_header:
            raise ValueError(f"{path}: the header row disagrees with its sidecar")
        for row in reader:
            if not row:
                continue
            where = f"{path}:{reader.line_num}"
            if row[0].startswith("#"):
                raise ValueError(f"{where}: this format has no comment lines")
            if len(row) != len(expected_header):
                raise ValueError(
                    f"{where}: the row has {len(row)} fields but the header declares "
                    f"{len(expected_header)}"
                )
            require_row_in_order(row[0], rows, where, SHOT_COLUMN)
            det_rows.append(
                np.packbits(bits_from_cells(row[feature_at], where, "detector"), bitorder="little")
            )
            obs_rows.append(
                np.packbits(bits_from_cells(row[target_at], where, "observable"), bitorder="little")
            )
            rows += 1
            if len(det_rows) == chunk_rows:
                yield flush()
    if det_rows:
        yield flush()
    if rows != declared_rows:
        raise ValueError(
            f"{path}: the table holds {rows} rows but the manifest declares {declared_rows}"
        )
    digest = hasher.hexdigest(rows, n_detectors, n_observables)
    if digest != expected_hash:
        raise ValueError(
            f"{path}: recomputed content_hash {digest} does not equal the manifest's "
            f"{expected_hash}; the table does not carry the rows the manifest describes"
        )


def source_prefix_hash(
    rows: Iterable[RowChunk], n_rows: int, n_detectors: int, n_observables: int
) -> str:
    """The ``content_hash`` of the first ``n_rows`` rows of a stream.

    Stops consuming as soon as ``n_rows`` are seen — a generated stream of 304,000 rows is
    not sampled to compare its first 16,000 — and refuses a stream that ends first, since
    a digest over fewer rows than asked for can never equal the source's.
    """
    if n_rows < 0:
        raise ValueError(f"n_rows must be >= 0, got {n_rows}")
    hasher = StreamingContentHasher()
    seen = 0
    if n_rows > 0:
        for dets, obs in rows:
            if dets.shape[0] != obs.shape[0]:
                raise ValueError("detector and observable chunks disagree in row count")
            take = min(int(dets.shape[0]), n_rows - seen)
            if take > 0:
                hasher.update(dets[:take], obs[:take])
                seen += take
            if seen >= n_rows:
                break
    if seen != n_rows:
        raise ValueError(
            f"the stream ended after {seen} rows; {n_rows} were needed for the prefix digest"
        )
    return hasher.hexdigest(n_rows, n_detectors, n_observables)


# ---------------------------------------------------------------------------
# Resolution


def default_fetch_cohort(zenodo: ZenodoConfig) -> dict[str, Any]:
    """The real Zenodo fetch, with the published archive MD5 pinned from the config."""
    return fetch_cohort(
        zenodo.record,
        zenodo.archive,
        zenodo.cohort_prefix,
        zenodo.cache_dir,
        archive_md5_published=zenodo.archive_md5_published,
    )


def resolve_source(
    config: ResidualConfig, *, fetch_cohort: FetchCohort | None = None
) -> ResolvedSource:
    """Identify, verify and freeze the source a configuration names.

    Refuses before any sampling or staging: a wrong content hash, a decoder kind that
    does not fit the profile, an inadmissible extension or an unverifiable Willow cohort
    all raise here, so a build that cannot be reproduced never creates an output directory.
    """
    match config.source.kind:
        case SourceKind.LEGACY_ML_CSV:
            source = _resolve_legacy(config)
        case SourceKind.DEVICE_ML_CSV:
            source = _resolve_device_ml_csv(config)
        case SourceKind.DEVICE_CONFIG:
            source = _resolve_device_config(config)
        case SourceKind.HARDWARE_WILLOW:
            source = _resolve_willow(config, fetch_cohort or default_fetch_cohort)
    _check_extension(config, source)
    slices = time_slices(source.circuit_for_coords)
    source.verification["time_slices"] = {
        "n_slices": slices.n_slices,
        "sizes": [int(s) for s in slices.sizes],
    }
    return source


def _check_extension(config: ResidualConfig, source: ResolvedSource) -> None:
    """Apply the call-size-sequence rule (Decision 6) before a single shot is sampled."""
    generation = config.generation
    if generation.mode is not GenerationMode.EXTEND:
        return
    identity = source.identity
    if source.decoder.kind is DecoderKind.FROZEN_REFERENCE_DEM:
        raise ValueError(
            "extend mode is refused for a dynamic profile: its rows depend on hidden "
            "drift/burst/leakage state, so no prefix rule makes a longer run reproduce them"
        )
    if identity.shots < 1 or identity.seed is None or identity.chunk_size is None:
        raise ValueError(
            f"{identity.kind.value} records no seeded sample stream to extend "
            f"(shots={identity.shots}, seed={identity.seed}, chunk_size={identity.chunk_size})"
        )
    if not generation.require_source_prefix:
        return
    if generation.shots is None or generation.seed is None:
        raise ValueError("extend mode requires generation.shots and generation.seed")
    if generation.seed != identity.seed:
        raise ValueError(
            f"generation.seed {generation.seed} differs from the source seed {identity.seed}; "
            "a different seed cannot reproduce the source as a prefix"
        )
    if not prefix_admissible(
        identity.shots, identity.chunk_size, generation.shots, generation.chunk_size
    ):
        raise ValueError(
            f"extension is not prefix-admissible: the source's sample() call sizes "
            f"(shots={identity.shots}, chunk_size={identity.chunk_size}) are not a prefix of "
            f"the new run's (shots={generation.shots}, chunk_size={generation.chunk_size}); "
            "no sampling was performed"
        )
    source.verification["extension"] = {
        "prefix_admissible": True,
        "source_shots": identity.shots,
        "source_chunk_size": identity.chunk_size,
        "new_shots": generation.shots,
        "new_chunk_size": generation.chunk_size,
        "seed": generation.seed,
        "content_hash_to_reproduce": identity.hashes["content_hash"],
    }


def _read_meta(config: ResidualConfig) -> tuple[Path, dict[str, Any], DatasetMeta]:
    path = config.source.path
    if path is None:  # pragma: no cover - config refuses file kinds without a path
        raise ValueError("source.path is required")
    if not path.is_file():
        raise FileNotFoundError(f"source table {path} does not exist")
    manifest = read_manifest_only(path)
    meta = DatasetMeta.from_json_dict(manifest)
    if meta.n_observables != 1:
        raise MultiObservableError(meta.n_observables, str(path))
    if len(meta.environments) != 1:
        raise ValueError(
            f"{path} holds {len(meta.environments)} environments; residual sources are "
            "single-environment"
        )
    if meta.contract is not Contract.LOGICAL_FRAME:
        raise ValueError(f"{path} is a {meta.contract} table; residual sources are Contract A")
    if meta.content_hash != config.source.expected_content_hash:
        raise ValueError(
            f"{path}: manifest content_hash {meta.content_hash} does not equal the configured "
            f"expected_content_hash {config.source.expected_content_hash}; this is not the "
            "file the configuration was reviewed against"
        )
    return path, manifest, meta


def _sidecar_paths(path: Path) -> dict[str, str]:
    companions = MLCSVExporter().companions(path)
    paths = {"source": str(path), "manifest": str(companions[0])}
    for key, companion in zip(("structure", "provenance"), companions[1:], strict=True):
        if companion.is_file():
            paths[key] = str(companion)
    return paths


def _chunks_as_rows(chunks: Iterable[Any]) -> Iterator[RowChunk]:
    for chunk in chunks:
        dets = np.asarray(chunk.detectors, dtype=np.uint8)
        obs = np.asarray(chunk.observables, dtype=np.uint8)
        yield dets, obs


def _resolve_legacy(config: ResidualConfig) -> ResolvedSource:
    path, manifest, meta = _read_meta(config)
    env = meta.environments[0]
    noise_model, p, channels = require_legacy_environment(env)
    build = build_environment(
        environment_id=env.environment_id,
        distance=meta.distance,
        base_p=p,
        axis=DriftAxis(env.axis),
        axis_value=env.axis_value,
        shots=env.shots,
        noise_model=noise_model,
        rounds=meta.rounds,
        basis=meta.basis,
        rotated=meta.rotated,
    )
    # The same guard qa.benchmark_dataset applies: a rebuild is only faithful while the
    # axis leaves the channel vector as recorded, and a future axis that breaks that
    # must surface as a refusal rather than a decoder for the wrong circuit.
    _, _, rebuilt_channels = require_legacy_environment(build.spec)
    if rebuilt_channels != channels:
        raise ValueError(
            f"{path}: the environment rebuilt from its manifest parameters has channels "
            f"{rebuilt_channels.as_dict()}, but the file records {channels.as_dict()}"
        )
    if build.circuit.num_detectors != meta.n_detectors:
        raise ValueError(
            f"{path}: rebuilt circuit has {build.circuit.num_detectors} detectors, manifest "
            f"declares {meta.n_detectors}"
        )
    provenance_sidecar = read_provenance_only(path)
    recorded_circuit = None
    if provenance_sidecar is not None:
        environments = provenance_sidecar.get("environments", [])
        if environments and environments[0].get("circuit"):
            recorded_circuit = str(environments[0]["circuit"])
            if recorded_circuit != str(build.circuit):
                raise ValueError(
                    f"{path}: the provenance sidecar's circuit text differs from the rebuilt "
                    "circuit; the shots did not come from the circuit this would decode"
                )
    decoder = from_circuit(
        build.circuit,
        DecoderKind.CIRCUIT_DEM,
        {
            "source_path": str(path),
            "source_content_hash": meta.content_hash,
            "rebuild": "qecgen.environments.build_environment from the manifest environment",
            "channels_cross_checked": True,
            "provenance_circuit_text_checked": recorded_circuit is not None,
        },
    )
    hashes = {
        "content_hash": str(meta.content_hash),
        "content_hash_algorithm": meta.content_hash_algorithm,
        "dem_sha256": decoder.dem_sha256,
        "dem_blake2b128": decoder.dem_blake2b128,
    }
    if meta.structure_dem_sha is not None:
        hashes["manifest_structure_dem_sha"] = meta.structure_dem_sha
    identity = SourceIdentity(
        kind=SourceKind.LEGACY_ML_CSV,
        paths=_sidecar_paths(path),
        hashes=hashes,
        distance=meta.distance,
        rounds=meta.rounds,
        basis=str(meta.basis).upper(),
        rotated=meta.rotated,
        noise_model=str(noise_model),
        noise_parameters={
            "p": p,
            "effective_p": build.spec.p,
            "channels": channels.as_dict(),
            "axis": env.axis,
            "axis_value": env.axis_value,
        },
        n_detectors=meta.n_detectors,
        n_observables=meta.n_observables,
        shots=meta.shots,
        seed=meta.seed,
        chunk_size=meta.chunk_size,
        circuit_sha256=str(decoder.provenance["circuit_sha256"]),
        dem_available="exact: decomposed DEM of the rebuilt legacy noisy circuit",
        provenance_limitations=[
            "circuit rebuilt from manifest parameters through environments.build_environment; "
            "channel vector cross-checked against the manifest"
            + (
                ", circuit text cross-checked against the provenance sidecar"
                if recorded_circuit
                else ""
            ),
            f"manifest version {manifest.get('manifest_version', 1)}; generated_at "
            f"{meta.generated_at or 'unrecorded'}; git_commit {meta.git_commit or 'unrecorded'}",
        ],
    )
    circuit = build.circuit

    def source_rows(chunk_rows: int) -> Iterator[RowChunk]:
        return iter_ml_csv_rows(path, manifest, chunk_rows)

    def generated_rows(shots: int, seed: int, chunk_size: int) -> Iterator[RowChunk]:
        return _chunks_as_rows(iter_chunks(circuit, shots, seed, chunk_size))

    return ResolvedSource(
        identity=identity,
        decoder=decoder,
        circuit_for_coords=decoder.circuit,
        config=config,
        verification={"channels_equal_manifest": True},
        _source_rows=source_rows,
        _generated_rows=generated_rows,
    )


def _device_decoder(
    config: ResidualConfig, cfg: dict[str, Any], caller_provenance: dict[str, Any]
) -> tuple[stim.Circuit, NoiseProfile, DecoderModel]:
    """Build the device decoder the configured kind asks for, refusing a mismatch.

    A static profile decoded with a "frozen reference" would carry a reference-model
    disclaimer for a DEM that is in fact exact; a dynamic profile decoded as
    ``static_profile_dem`` would claim exactness it does not have. Both are refused.
    """
    if cfg.get("mode") != "device":
        raise ValueError(f"generation config mode {cfg.get('mode')!r} is not 'device'")
    if cfg["sampling"].get("emit_mechanisms", False):
        raise ValueError(
            "a device source sampled with emit_mechanisms uses the DEM sampler stream, "
            "which iter_profile_chunks cannot reproduce; residual sources are Contract A"
        )
    ideal = ideal_circuit(cfg)
    profile = NoiseProfile.from_dict(cfg["noise"])
    if ideal.num_observables != 1:
        raise MultiObservableError(ideal.num_observables, "device ideal circuit")
    kind = config.decoder.kind
    if kind is DecoderKind.STATIC_PROFILE_DEM:
        if profile.dynamic:
            raise ValueError(
                "decoder.kind static_profile_dem needs a static profile, but this profile "
                "is dynamic; use frozen_reference_dem, which records its transformation"
            )
        decoder = from_static_profile(ideal, profile, caller_provenance)
    elif kind is DecoderKind.FROZEN_REFERENCE_DEM:
        if not profile.dynamic:
            raise ValueError(
                "decoder.kind frozen_reference_dem needs a dynamic profile, but this profile "
                "is static and has an exact DEM; use static_profile_dem"
            )
        decoder = from_frozen_reference(ideal, profile, caller_provenance)
    else:  # pragma: no cover - config restricts device decoder kinds
        raise ValueError(f"decoder kind {kind.value} cannot be built from a device profile")
    return ideal, profile, decoder


def _device_identity_parts(
    profile: NoiseProfile, decoder: DecoderModel
) -> tuple[str, list[str], str]:
    if profile.dynamic:
        return (
            "reference: frozen reference DEM built from the drift stationary point; "
            "bursts and leakage omitted; not an exact DEM of the dynamic process",
            [
                "dynamic profile: rows depend on hidden drift/burst/leakage state that no "
                "static DEM describes",
                "frozen reference transformation: " + str(decoder.provenance["transformation"]),
                "extension (extend mode) is refused for dynamic profiles",
            ],
            "frozen_profile_sha256",
        )
    return (
        "exact: decomposed DEM of build_noisy_circuit(ideal_circuit(config), profile)",
        ["scenario profile parameters, not measured hardware rates (parameter_provenance)"],
        "profile_sha256",
    )


def _resolve_device_ml_csv(config: ResidualConfig) -> ResolvedSource:
    path, manifest, meta = _read_meta(config)
    cfg = meta.generation_config
    if cfg is None:
        raise ValueError(
            f"{path}: manifest version 1 carries no generation_config; a device source needs "
            "the version-2 manifest that records its profile"
        )
    audit = meta.generation_audit or {}
    ideal, profile, decoder = _device_decoder(
        config,
        cfg,
        {"source_path": str(path), "source_content_hash": meta.content_hash},
    )
    if ideal.num_detectors != meta.n_detectors:
        raise ValueError(
            f"{path}: rebuilt ideal circuit has {ideal.num_detectors} detectors, manifest "
            f"declares {meta.n_detectors}"
        )
    # The one check that ties the decoder to the rows: the circuit whose DEM this is must
    # be the circuit the sampler ran. For a static profile that is the noisy circuit; for
    # a dynamic profile the sampler ran the ideal circuit plus injected state, so the
    # audit records the ideal's hash.
    audited_circuit = audit.get("circuit_sha256")
    rebuilt_key = "circuit_sha256" if not profile.dynamic else "ideal_circuit_sha256"
    if audited_circuit != decoder.provenance[rebuilt_key]:
        raise ValueError(
            f"{path}: generation_audit.circuit_sha256 {audited_circuit!r} does not equal the "
            f"rebuilt circuit's {decoder.provenance[rebuilt_key]}; the shots did not come "
            "from the circuit this decoder describes"
        )
    profile_key = "profile_sha256" if not profile.dynamic else "dynamic_profile_sha256"
    audited_profile = audit.get("profile_sha256")
    if audited_profile is not None and audited_profile != decoder.provenance[profile_key]:
        raise ValueError(
            f"{path}: generation_audit.profile_sha256 {audited_profile!r} does not equal the "
            f"profile rebuilt from generation_config ({decoder.provenance[profile_key]})"
        )
    dem_available, limitations, _ = _device_identity_parts(profile, decoder)
    hashes = {
        "content_hash": str(meta.content_hash),
        "content_hash_algorithm": meta.content_hash_algorithm,
        "dem_sha256": decoder.dem_sha256,
        "dem_blake2b128": decoder.dem_blake2b128,
        "profile_sha256": str(decoder.provenance[profile_key]),
        "ideal_circuit_sha256": str(decoder.provenance["ideal_circuit_sha256"]),
    }
    if profile.dynamic:
        hashes["frozen_profile_sha256"] = str(decoder.provenance["frozen_profile_sha256"])
    if "config_sha256" in audit:
        hashes["generation_config_sha256"] = str(audit["config_sha256"])
    identity = SourceIdentity(
        kind=SourceKind.DEVICE_ML_CSV,
        paths=_sidecar_paths(path),
        hashes=hashes,
        distance=meta.distance,
        rounds=meta.rounds,
        basis=str(meta.basis).upper(),
        rotated=meta.rotated,
        noise_model="device_profile",
        noise_parameters={
            "profile": profile.to_dict(),
            "parameter_provenance": cfg.get("parameter_provenance"),
            "dynamic": profile.dynamic,
        },
        n_detectors=meta.n_detectors,
        n_observables=meta.n_observables,
        shots=meta.shots,
        seed=meta.seed,
        chunk_size=meta.chunk_size,
        circuit_sha256=str(decoder.provenance["circuit_sha256"]),
        dem_available=dem_available,
        provenance_limitations=[
            *limitations,
            f"manifest version {manifest.get('manifest_version')}; generated_at "
            f"{meta.generated_at or 'unrecorded'}; git_commit {meta.git_commit or 'unrecorded'}",
        ],
    )

    def source_rows(chunk_rows: int) -> Iterator[RowChunk]:
        return iter_ml_csv_rows(path, manifest, chunk_rows)

    def generated_rows(shots: int, seed: int, chunk_size: int) -> Iterator[RowChunk]:
        return _chunks_as_rows(iter_profile_chunks(ideal, profile, shots, seed, chunk_size))

    return ResolvedSource(
        identity=identity,
        decoder=decoder,
        circuit_for_coords=decoder.circuit,
        config=config,
        verification={"audit_circuit_sha256_equal": True, "profile_dynamic": profile.dynamic},
        _source_rows=source_rows,
        _generated_rows=generated_rows,
    )


def _resolve_device_config(config: ResidualConfig) -> ResolvedSource:
    """A bare version-1 config: a stream to sample, no rows to read."""
    path = config.source.path
    if path is None:  # pragma: no cover - config refuses file kinds without a path
        raise ValueError("source.path is required")
    if not path.is_file():
        raise FileNotFoundError(f"device config {path} does not exist")
    cfg = read_config(path)
    ideal, profile, decoder = _device_decoder(config, cfg, {"source_config_path": str(path)})
    dem_available, limitations, _ = _device_identity_parts(profile, decoder)
    profile_key = "profile_sha256" if not profile.dynamic else "dynamic_profile_sha256"
    description = cfg["circuit"]
    identity = SourceIdentity(
        kind=SourceKind.DEVICE_CONFIG,
        paths={"config": str(path)},
        hashes={
            "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "dem_sha256": decoder.dem_sha256,
            "dem_blake2b128": decoder.dem_blake2b128,
            "profile_sha256": str(decoder.provenance[profile_key]),
            "ideal_circuit_sha256": str(decoder.provenance["ideal_circuit_sha256"]),
        },
        distance=int(description["distance"]),
        rounds=int(description["rounds"]),
        basis=str(description["basis"]).upper(),
        rotated=bool(description["rotated"]),
        noise_model="device_profile",
        noise_parameters={
            "profile": profile.to_dict(),
            "parameter_provenance": cfg.get("parameter_provenance"),
            "dynamic": profile.dynamic,
        },
        n_detectors=ideal.num_detectors,
        n_observables=ideal.num_observables,
        shots=0,
        seed=None,
        chunk_size=None,
        circuit_sha256=str(decoder.provenance["circuit_sha256"]),
        dem_available=dem_available,
        provenance_limitations=[*limitations, "no existing rows: every row is freshly sampled"],
    )

    def source_rows(_chunk_rows: int) -> Iterator[RowChunk]:
        raise ValueError(f"device_config {path} has no source rows; sample a fresh stream instead")

    def generated_rows(shots: int, seed: int, chunk_size: int) -> Iterator[RowChunk]:
        return _chunks_as_rows(iter_profile_chunks(ideal, profile, shots, seed, chunk_size))

    return ResolvedSource(
        identity=identity,
        decoder=decoder,
        circuit_for_coords=decoder.circuit,
        config=config,
        verification={"profile_dynamic": profile.dynamic},
        _source_rows=source_rows,
        _generated_rows=generated_rows,
    )


def _expected_dict(expected: WillowExpected) -> dict[str, Any]:
    return {
        "table_sha256": expected.table_sha256,
        "circuit_sha256": expected.circuit_sha256,
        "distance": expected.distance,
        "basis": expected.basis,
        "rounds": expected.rounds,
        "orientation": expected.orientation,
    }


def _check_formatted_prefix(config: ResidualConfig, cohort: ImportedCohort) -> dict[str, Any]:
    """The formatted 2,000-row file must be exactly the cohort rows it claims to be.

    Verified bit for bit, not by row count: the formatted file and the cohort could share
    a shape while carrying different rows, and the recorded offset is checked against the
    manifest before the bits so a wrong offset is reported as such.
    """
    prefix = config.source.formatted_prefix
    if prefix is None:
        return {"checked": False, "reason": "no formatted_prefix configured"}
    if not prefix.path.is_file():
        raise FileNotFoundError(f"formatted_prefix table {prefix.path} does not exist")
    manifest = read_manifest_only(prefix.path)
    meta = DatasetMeta.from_json_dict(manifest)
    if meta.content_hash != prefix.expected_content_hash:
        raise ValueError(
            f"formatted_prefix {prefix.path}: manifest content_hash {meta.content_hash} does "
            f"not equal the configured expected_content_hash {prefix.expected_content_hash}"
        )
    if meta.n_detectors != cohort.ideal.num_detectors or meta.n_observables != 1:
        raise ValueError(
            f"formatted_prefix {prefix.path}: widths ({meta.n_detectors}, {meta.n_observables}) "
            f"differ from the cohort's ({cohort.ideal.num_detectors}, 1)"
        )
    generation = meta.generation_config or {}
    if generation:
        if generation.get("mode") != "hardware":
            raise ValueError(
                f"formatted_prefix {prefix.path} was generated in mode "
                f"{generation.get('mode')!r}, not from hardware rows"
            )
        recorded_offset = generation.get("hardware", {}).get("offset")
        if recorded_offset is not None and int(recorded_offset) != prefix.offset:
            raise ValueError(
                f"formatted_prefix {prefix.path} records source offset {recorded_offset}, the "
                f"configuration says {prefix.offset}"
            )
    stop = prefix.offset + meta.shots
    if stop > len(cohort.detectors):
        raise ValueError(
            f"formatted_prefix rows {prefix.offset}..{stop} exceed the cohort's "
            f"{len(cohort.detectors)} rows"
        )
    dets, obs = zip(*list(iter_ml_csv_rows(prefix.path, manifest, 2000)), strict=True)
    file_dets = np.concatenate(dets)
    file_obs = np.concatenate(obs)
    cohort_dets = cohort.detectors[prefix.offset : stop]
    cohort_obs = cohort.observables[prefix.offset : stop]
    if not np.array_equal(file_dets, cohort_dets) or not np.array_equal(file_obs, cohort_obs):
        differing = int(
            ((file_dets != cohort_dets).any(axis=1) | (file_obs != cohort_obs).any(axis=1)).sum()
        )
        raise ValueError(
            f"formatted_prefix {prefix.path} rows differ from cohort rows "
            f"{prefix.offset}..{stop} ({differing} row(s) differ); it is not the documented prefix"
        )
    return {
        "checked": True,
        "path": str(prefix.path),
        "content_hash": str(meta.content_hash),
        "offset": prefix.offset,
        "rows": meta.shots,
        "bit_for_bit_equal": True,
    }


def _resolve_willow(config: ResidualConfig, fetch: FetchCohort) -> ResolvedSource:
    source = config.source
    expected = source.expected
    zenodo = source.zenodo
    if source.table is None or source.circuit is None or expected is None or zenodo is None:
        # The config parser requires all four for this kind; kept as a guard for mypy.
        raise ValueError("hardware_willow needs table, circuit, expected and zenodo")
    if config.decoder.member is None or config.decoder.expected_sha256 is None:
        raise ValueError("official_dem needs decoder.member and decoder.expected_sha256")
    for name, path in (("table", source.table), ("circuit", source.circuit)):
        if not path.is_file():
            raise FileNotFoundError(f"Willow {name} {path} does not exist")
    # The observable count is checked before the mirror loader runs so that a
    # multi-observable circuit stops the source with the schema proposal, not a generic
    # mirror error.
    circuit_bytes = source.circuit.read_bytes()
    circuit_file_sha = hashlib.sha256(circuit_bytes).hexdigest()
    if circuit_file_sha != expected.circuit_sha256:
        raise ValueError(
            f"{source.circuit}: sha256 {circuit_file_sha} does not equal expected.circuit_sha256"
        )
    parsed = stim.Circuit(circuit_bytes.decode("utf-8"))
    if parsed.num_observables != 1:
        raise MultiObservableError(parsed.num_observables, str(source.circuit))
    expected_dict = _expected_dict(expected)
    cohort = load_willow_derived(source.table, source.circuit, expected_dict)
    prefix_report = _check_formatted_prefix(config, cohort)

    receipts = fetch(zenodo)
    cohort_report = verify_cohort(zenodo.cache_dir, cohort, expected.circuit_sha256, expected_dict)

    member = config.decoder.member
    member_path = zenodo.cache_dir / member
    if not member_path.is_file():
        raise FileNotFoundError(f"decoder member {member} is not in the cache {zenodo.cache_dir}")
    member_bytes = member_path.read_bytes()
    member_sha = hashlib.sha256(member_bytes).hexdigest()
    if member_sha != config.decoder.expected_sha256:
        raise ValueError(
            f"{member}: sha256 {member_sha} does not equal decoder.expected_sha256 "
            f"{config.decoder.expected_sha256}"
        )
    entries = [m for m in receipts.get("members", []) if m.get("path") == member]
    if len(entries) != 1:
        raise ValueError(f"receipts list {len(entries)} entries for member {member}; expected one")
    entry = dict(entries[0])
    if entry.get("sha256") != member_sha:
        raise ValueError(f"receipt sha256 for {member} disagrees with the cached bytes")

    noisy_bytes = (zenodo.cache_dir / "circuit_noisy_si1000.stim").read_bytes()
    noisy_dem = stim.Circuit(noisy_bytes.decode("utf-8")).detector_error_model(
        decompose_errors=True
    )
    official = stim.DetectorErrorModel(member_bytes.decode("utf-8"))
    cross_reference = {
        "noisy_circuit": "circuit_noisy_si1000.stim",
        "noisy_circuit_sha256": hashlib.sha256(noisy_bytes).hexdigest(),
        "method": "circuit.detector_error_model(decompose_errors=True)",
        "dem_sha256": hashlib.sha256(str(noisy_dem).encode("utf-8")).hexdigest(),
        "num_errors": noisy_dem.num_errors,
        "official_num_errors": official.num_errors,
        "approx_equals_official": bool(official.approx_equals(noisy_dem, atol=1e-9)),
    }
    fitting = _third_party_fitting(member)
    decoder = from_official_dem(
        member_bytes,
        cohort.ideal,
        {
            "zenodo": {
                "record": receipts.get("record"),
                "archive": receipts.get("archive"),
                "cohort_prefix": zenodo.cohort_prefix,
                "member": entry,
            },
            "cohort_source": dict(cohort.source),
            "cohort_verification": cohort_report,
            "third_party_fitting": fitting,
            "matching_mode_note": (
                "standard matching on Google's shipped prior: the DEM comes from a "
                "correlated-matching pathway but is decoded here with enable_correlations=False"
            ),
            "noisy_circuit_dem_cross_reference": cross_reference,
        },
    )
    limitations = [
        str(cohort.source.get("ordering", "acquisition chronology unverified")),
        "original_archive_independently_verified=False (qecgen.hardware.load_willow_derived): "
        "the mirror is verified against the archive members Zenodo serves, member CRC32 and "
        "sha256 recorded; the 5.7 GB archive-level MD5 is not re-verified",
        "the shipped DEM is a correlated-matching prior decoded here with standard matching",
    ]
    if not cross_reference["approx_equals_official"]:
        limitations.append(
            "the shipped error_model.dem is not the decomposed DEM of circuit_noisy_si1000.stim "
            f"({official.num_errors} vs {noisy_dem.num_errors} errors); its construction is "
            "the publisher's and is used verbatim"
        )
    if fitting is not None:
        limitations.append(
            "third-party fitting: " + str(fitting["method"]) + "; overlap with this cohort unknown"
        )
    paths = {
        "table": str(source.table),
        "circuit": str(source.circuit),
        "zenodo_cache_dir": str(zenodo.cache_dir),
        "decoder_member": str(member_path),
    }
    hashes = {
        "table_sha256": expected.table_sha256,
        "circuit_file_sha256": circuit_file_sha,
        "dem_sha256": decoder.dem_sha256,
        "dem_blake2b128": decoder.dem_blake2b128,
        "noisy_circuit_sha256": str(cross_reference["noisy_circuit_sha256"]),
    }
    if prefix_report.get("checked"):
        paths["formatted_prefix"] = str(prefix_report["path"])
        hashes["formatted_prefix_content_hash"] = str(prefix_report["content_hash"])
    identity = SourceIdentity(
        kind=SourceKind.HARDWARE_WILLOW,
        paths=paths,
        hashes=hashes,
        distance=expected.distance,
        rounds=expected.rounds,
        basis=expected.basis,
        rotated=True,
        noise_model="hardware",
        noise_parameters={
            "note": "measured device data; no generating noise model is known",
            "orientation": expected.orientation,
            "source_kind": cohort.source.get("source_kind"),
            "publisher_identity_verified": cohort.source.get("publisher_identity_verified"),
        },
        n_detectors=cohort.ideal.num_detectors,
        n_observables=cohort.ideal.num_observables,
        shots=int(cohort.detectors.shape[0]),
        seed=None,
        chunk_size=None,
        circuit_sha256=str(decoder.provenance["circuit_sha256"]),
        dem_available=(
            f"official: Google's shipped {member} (Zenodo record {zenodo.record}), used verbatim"
        ),
        provenance_limitations=limitations,
    )
    detectors = np.asarray(cohort.detectors, dtype=np.uint8)
    observables = np.asarray(cohort.observables, dtype=np.uint8)
    if detectors.shape[1] != packed_width(cohort.ideal.num_detectors):
        raise ValueError("cohort detector width disagrees with the circuit")  # pragma: no cover

    def source_rows(chunk_rows: int) -> Iterator[RowChunk]:
        for start in range(0, detectors.shape[0], chunk_rows):
            stop = min(start + chunk_rows, detectors.shape[0])
            yield detectors[start:stop].copy(), observables[start:stop].copy()

    return ResolvedSource(
        identity=identity,
        decoder=decoder,
        circuit_for_coords=decoder.circuit,
        config=config,
        verification={
            "cohort": cohort_report,
            "formatted_prefix": prefix_report,
            "receipts": receipts,
            "noisy_circuit_dem_cross_reference": cross_reference,
        },
        _source_rows=source_rows,
        _generated_rows=None,
    )
