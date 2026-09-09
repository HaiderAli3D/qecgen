"""Versioned experiment configuration and bounded-memory generation.

The old CLI remains an exact preset. Device profiles and imported hardware have
different metadata identities: a missing calibration is never represented as p=0.
Paths are resolved against the working directory and content-checked before use.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import stim
from scipy.sparse import csc_matrix

from qecgen.circuits import Basis, NoiseModel, build_circuit, default_rounds
from qecgen.dataset import (
    Contract,
    DatasetMeta,
    EnvironmentModel,
    EnvironmentSpec,
    InMemoryDataset,
    StreamingContentHasher,
    StructureLevel,
    dem_digest,
)
from qecgen.dem import DemStructure, parse_dem
from qecgen.environments import DriftAxis, build_environment
from qecgen.exporters import get_exporter, infer_format
from qecgen.exporters.hdf5 import StreamingHDF5Writer
from qecgen.hardware import load_willow_derived, validate_layout_identity
from qecgen.noise import NoiseProfile, build_noisy_circuit, profile_audit
from qecgen.sampling import DEFAULT_CHUNK_SIZE, ShotChunk, iter_chunks, iter_profile_chunks

_ENGINE_SOURCES = {
    name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    for name in (
        "configuration.py",
        "noise.py",
        "sampling.py",
        "hardware.py",
        "circuits.py",
        "environments.py",
        "dataset.py",
        "dem.py",
    )
}


def _object(value: Any, name: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object")
    if unknown := set(value) - allowed:
        raise ValueError(f"Unknown {name} fields: {sorted(unknown)}")
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a JSON boolean")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha(value: Any, name: str) -> str:
    result = _text(value, name)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return result


def normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Resolve every default and refuse ignored fields before any output is opened."""
    config = copy.deepcopy(raw)
    _object(
        config,
        "configuration",
        {
            "version",
            "mode",
            "output",
            "sampling",
            "circuit",
            "legacy",
            "noise",
            "parameter_provenance",
            "hardware",
        },
    )
    if type(config.get("version")) is not int or config["version"] != 1:
        raise ValueError("Configuration version must be integer 1")
    mode = _text(config.get("mode"), "mode")
    if mode not in {"legacy", "device", "hardware"}:
        raise ValueError("mode must be legacy, device or hardware")
    active = {"legacy": "legacy", "device": "noise", "hardware": "hardware"}[mode]
    for key in {"legacy", "noise", "hardware"} - {active}:
        if key in config:
            raise ValueError(f"{key} cannot be used in {mode} mode")
    output = _object(config.setdefault("output", {}), "output", {"path", "format", "structure"})
    output["path"] = str(Path(_text(output.get("path"), "output.path")).resolve())
    output["format"] = _text(output.get("format", "hdf5"), "output.format")
    exporter = get_exporter(output["format"])
    output["structure"] = str(StructureLevel(output.get("structure", "none")))
    try:
        inferred = infer_format(Path(output["path"]))
    except ValueError:
        inferred = exporter.format_name
    if inferred != exporter.format_name:
        raise ValueError("output.path extension disagrees with output.format")
    # Exporters such as NumPy otherwise silently append an extension.
    if not output["path"].endswith(exporter.extension):
        raise ValueError(f"output.path must end in {exporter.extension}")
    sampling = _object(
        config.setdefault("sampling", {}),
        "sampling",
        {
            "shots",
            "seed",
            "chunk_size",
            "emit_mechanisms",
        },
    )
    sampling["shots"] = _integer(sampling.get("shots"), "sampling.shots", 1)
    sampling["seed"] = _integer(sampling.get("seed"), "sampling.seed")
    if sampling["seed"] >= 2**64:
        raise ValueError("sampling.seed must be below 2**64")
    sampling["chunk_size"] = _integer(
        sampling.get("chunk_size", DEFAULT_CHUNK_SIZE), "sampling.chunk_size", 1
    )
    sampling["emit_mechanisms"] = _boolean(
        sampling.get("emit_mechanisms", False), "sampling.emit_mechanisms"
    )
    circuit = _object(
        config.setdefault("circuit", {}),
        "circuit",
        {
            "distance",
            "rounds",
            "basis",
            "rotated",
            "stim_file",
            "sha256",
        },
    )
    circuit["distance"] = _integer(circuit.get("distance"), "circuit.distance", 2)
    circuit["basis"] = str(Basis(circuit.get("basis", "z")))
    circuit["rotated"] = _boolean(circuit.get("rotated", True), "circuit.rotated")
    if "stim_file" in circuit:
        if mode != "device":
            raise ValueError("circuit.stim_file is only supported in device mode")
        circuit["stim_file"] = str(Path(_text(circuit["stim_file"], "stim_file")).resolve())
        circuit["sha256"] = _sha(circuit.get("sha256"), "circuit.sha256")
    elif "sha256" in circuit:
        raise ValueError("circuit.sha256 requires circuit.stim_file")
    if mode == "legacy":
        legacy = _object(config.get("legacy"), "legacy", {"noise_model", "p"})
        model = NoiseModel(legacy.get("noise_model", "stim_uniform_circuit_level"))
        probability = legacy.get("p")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError("legacy.p must be a finite probability")
        if not math.isfinite(probability):
            raise ValueError("legacy.p must be a finite probability")
        if not 0 <= probability <= 1:
            raise ValueError("legacy.p must lie in [0, 1]")
        legacy.update(noise_model=str(model), p=float(probability))
        rounds = circuit.get("rounds")
        if rounds is not None:
            _integer(rounds, "circuit.rounds", 1)
        circuit["rounds"] = default_rounds(model, circuit["distance"], rounds)
    else:
        circuit["rounds"] = _integer(
            circuit.get("rounds", circuit["distance"]), "circuit.rounds", 1
        )
    if mode == "device":
        profile = NoiseProfile.from_dict(config.get("noise", {}))
        config["noise"] = profile.to_dict()
        if profile.dynamic and sampling["emit_mechanisms"]:
            raise ValueError("Dynamic profiles have no independent DEM mechanism targets")
        if profile.dynamic and output["structure"] in {"dem", "full"}:
            raise ValueError("Dynamic profiles support structure none/coords; no exact static DEM")
        provenance = _object(
            config.get("parameter_provenance"),
            "parameter_provenance",
            {
                "kind",
                "description",
                "source",
                "fit_partition",
            },
        )
        if _text(provenance.get("kind"), "parameter_provenance.kind") not in {
            "scenario",
            "fitted",
            "measured",
        }:
            raise ValueError("parameter_provenance.kind must be scenario, fitted or measured")
        _text(provenance.get("description"), "parameter_provenance.description")
        if provenance["kind"] != "scenario":
            _text(provenance.get("source"), "parameter_provenance.source")
        if provenance["kind"] == "fitted" and provenance.get("fit_partition") != "train":
            raise ValueError("Fitted profiles must explicitly identify fit_partition='train'")
    elif "parameter_provenance" in config:
        raise ValueError("parameter_provenance applies to device profiles only")
    if mode == "hardware":
        hardware = _object(
            config.get("hardware"),
            "hardware",
            {
                "table",
                "circuit",
                "expected",
                "offset",
            },
        )
        for key in ("table", "circuit"):
            hardware[key] = str(Path(_text(hardware.get(key), f"hardware.{key}")).resolve())
        hardware["offset"] = _integer(hardware.get("offset", 0), "hardware.offset")
        expected = _object(
            hardware.get("expected"),
            "hardware.expected",
            {
                "table_sha256",
                "circuit_sha256",
                "distance",
                "basis",
                "rounds",
                "orientation",
            },
        )
        for key in ("table_sha256", "circuit_sha256"):
            expected[key] = _sha(expected.get(key), key)
        for key in ("distance", "rounds"):
            if _integer(expected.get(key), f"expected.{key}", 1) != circuit[key]:
                raise ValueError(f"hardware expected.{key} disagrees with circuit")
        if expected.get("basis") != circuit["basis"].upper() or not circuit["rotated"]:
            raise ValueError("Hardware basis/layout disagrees with the declared circuit")
        _text(expected.get("orientation"), "hardware.expected.orientation")
        if sampling["emit_mechanisms"] or output["structure"] in {"dem", "full"}:
            raise ValueError("Hardware import has actual outcomes, but no ground-truth DEM")
    input_paths = []
    if "stim_file" in circuit:
        input_paths.append(circuit["stim_file"])
    if mode == "hardware":
        input_paths.extend(config["hardware"][key] for key in ("table", "circuit"))
    output_path = Path(output["path"])
    output_set = {output_path, *exporter.companions(output_path)}
    if output_set.intersection(Path(value) for value in input_paths):
        raise ValueError("Output or its companion files must not overwrite an input source")
    json.dumps(config, allow_nan=False)
    return config


def read_config(path: Path) -> dict[str, Any]:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate configuration field: {key}")
            result[key] = value
        return result

    config = normalize_config(
        json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    )
    output = Path(config["output"]["path"])
    if path.resolve() in {output, *get_exporter(config["output"]["format"]).companions(output)}:
        raise ValueError("Output or its companion files must not overwrite the configuration file")
    return config


def expand_sweep(
    raw: dict[str, Any], field: str, values: list[float | bool]
) -> list[dict[str, Any]]:
    """Produce independent, replayable configs by changing one mechanism parameter.

    No fitted coefficient is inferred from a covariate. Sweeping humidity without
    a response model therefore deliberately changes metadata, not the physics.
    Named outputs and child seeds are fixed by the list order, not worker order.
    """
    base = normalize_config(raw)
    if not field.startswith("noise.") and field != "legacy.p":
        raise ValueError(
            "Sweep a noise parameter or legacy.p, not dataset identity or source paths"
        )
    if not values or len(values) > 1000:
        raise ValueError("A sweep requires 1 to 1000 explicit scalar values")
    if any(type(value) not in {int, float, bool} for value in values):
        raise ValueError("Sweep values must be numeric or boolean")
    seeds = np.random.SeedSequence(base["sampling"]["seed"]).spawn(len(values))
    path = Path(base["output"]["path"])
    extension = get_exporter(base["output"]["format"]).extension
    stem = path.name[: -len(extension)]
    configs = []
    for index, (value, seed) in enumerate(zip(values, seeds, strict=True)):
        config = copy.deepcopy(base)
        target: Any = config
        parts = field.split(".")
        try:
            for part in parts[:-1]:
                target = target[int(part)] if isinstance(target, list) else target[part]
            if isinstance(target, list):
                target[int(parts[-1])] = value
            else:
                target[parts[-1]] = value
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise ValueError(f"Sweep field is not a valid profile path: {field}") from exc
        config["sampling"]["seed"] = int(seed.generate_state(1, dtype=np.uint64)[0])
        config["output"]["path"] = str(path.with_name(f"{stem}-{index:03d}{extension}"))
        configs.append(normalize_config(config))
    return configs


def _checked_circuit(path: str, expected: str) -> stim.Circuit:
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Circuit input does not match the recorded SHA-256")
    circuit = stim.Circuit(payload.decode("utf-8"))
    if circuit != circuit.without_noise():
        raise ValueError("External ideal circuits must not contain source-fitted noise")
    if circuit.num_detectors < 1 or circuit.num_observables != 1:
        raise ValueError("Expected detectors and one logical observable")
    return circuit


def ideal_circuit(config: dict[str, Any]) -> stim.Circuit:
    """Rebuild the stated layout; a custom circuit must match its source hash."""
    description = config["circuit"]
    if config["mode"] == "hardware":
        source = config["hardware"]
        circuit = _checked_circuit(source["circuit"], source["expected"]["circuit_sha256"])
        validate_layout_identity(circuit, description, source["expected"]["circuit_sha256"])
        return circuit
    if "stim_file" in description:
        circuit = _checked_circuit(description["stim_file"], description["sha256"])
        validate_layout_identity(circuit, description, description["sha256"])
        return circuit
    circuit, _ = build_circuit(
        description["distance"],
        0.0,
        rounds=description["rounds"],
        basis=Basis(description["basis"]),
        rotated=description["rotated"],
    )
    return circuit


def _coords_only(circuit: stim.Circuit) -> DemStructure:
    coords = circuit.get_detector_coordinates()
    width = max((len(values) for values in coords.values()), default=0)
    matrix = np.full((circuit.num_detectors, width), np.nan)
    for index, values in coords.items():
        matrix[index, : len(values)] = values
    return DemStructure(
        h=csc_matrix((circuit.num_detectors, 0), dtype=np.uint8),
        l=csc_matrix((circuit.num_observables, 0), dtype=np.uint8),
        priors=np.empty(0),
        components=(),
        detector_coords=matrix,
        n_detectors=circuit.num_detectors,
        n_observables=circuit.num_observables,
        n_mechanisms=0,
        coord_dim=width,
        has_matrices=False,
    )


@dataclass
class PreparedGeneration:
    meta: DatasetMeta
    structure: DemStructure | None
    chunks: Iterator[ShotChunk]


def prepare_generation(raw: dict[str, Any]) -> PreparedGeneration:
    config = normalize_config(raw)
    mode, sampling = config["mode"], config["sampling"]
    shots, seed, chunk = sampling["shots"], sampling["seed"], sampling["chunk_size"]
    emit = sampling["emit_mechanisms"]
    description = config["circuit"]
    level = StructureLevel(config["output"]["structure"])
    structure: DemStructure | None = None
    dem: stim.DetectorErrorModel | None = None
    audit: dict[str, Any] = {
        "has_dynamic": False,
        "engine_source_sha256": dict(_ENGINE_SOURCES),
        "source_hash_capture": "configuration-module import; restart processes after source edits",
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    if mode == "legacy":
        build = build_environment(
            0,
            description["distance"],
            config["legacy"]["p"],
            DriftAxis.P,
            config["legacy"]["p"],
            shots,
            NoiseModel(config["legacy"]["noise_model"]),
            description["rounds"],
            Basis(description["basis"]),
            description["rotated"],
        )
        circuit, dem, environment = build.circuit, build.dem, build.spec
        chunks = iter_chunks(circuit, shots, seed, chunk, emit, dem if emit else None)
        audit.update(backend="legacy_stim", preset=config["legacy"]["noise_model"])
    elif mode == "device":
        ideal = ideal_circuit(config)
        profile = NoiseProfile.from_dict(config["noise"])
        audit.update(profile_audit(ideal, profile, seed, shots, chunk))
        audit["has_dynamic"] = profile.dynamic
        circuit = ideal if profile.dynamic else build_noisy_circuit(ideal, profile)
        if not profile.dynamic and (emit or level in {StructureLevel.DEM, StructureLevel.FULL}):
            dem = circuit.detector_error_model(decompose_errors=True)
        environment = EnvironmentSpec(
            0,
            None,
            EnvironmentModel.DEVICE_PROFILE,
            None,
            str(circuit),
            str(dem) if dem is not None else "",
            shots,
            axis="profile",
        )
        chunks = (
            iter_chunks(circuit, shots, seed, chunk, True, dem)
            if emit
            else iter_profile_chunks(ideal, profile, shots, seed, chunk)
        )
        if emit:
            audit.update(
                backend="dem_sampler",
                contract="B",
                contains_mechanism_targets=True,
                rng_streams=["stim_dem"],
                rng_derivation="direct sampling.seed passed to Stim DEM sampler",
            )
    else:
        source = config["hardware"]
        cohort = load_willow_derived(
            Path(source["table"]), Path(source["circuit"]), source["expected"]
        )
        offset = source["offset"]
        if offset + shots > len(cohort.detectors):
            raise ValueError("Requested hardware slice exceeds available source rows")
        circuit = cohort.ideal
        layout_validation = validate_layout_identity(
            circuit, description, source["expected"]["circuit_sha256"]
        )
        environment = EnvironmentSpec(
            0,
            None,
            EnvironmentModel.HARDWARE,
            None,
            str(circuit),
            "",
            shots,
            axis="source_rows",
        )
        audit.update(
            source=cohort.source,
            backend="hardware_import",
            sampling_seed_used=False,
            source_rows={"offset": offset, "count": shots},
            layout_validation=layout_validation,
            source_licence=(
                "Google CC-BY-4.0; mirror Apache-2.0 declaration"
                if cohort.source["publisher_identity_verified"]
                else "user-supplied; not verified"
            ),
        )
        chunks = (
            ShotChunk(
                cohort.detectors[start : min(start + chunk, offset + shots)],
                cohort.observables[start : min(start + chunk, offset + shots)],
                None,
            )
            for start in range(offset, offset + shots, chunk)
        )
    if level is StructureLevel.COORDS:
        structure = _coords_only(circuit)
    elif level in {StructureLevel.DEM, StructureLevel.FULL}:
        if dem is None:
            raise ValueError("No exact DEM is available for this generation mode")
        structure = parse_dem(dem, circuit)
    audit["circuit_sha256"] = hashlib.sha256(str(circuit).encode()).hexdigest()
    audit["config_sha256"] = hashlib.sha256(
        json.dumps(config, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    meta = DatasetMeta(
        distance=description["distance"],
        rounds=description["rounds"],
        basis=Basis(description["basis"]),
        rotated=description["rotated"],
        shots=shots,
        seed=seed,
        chunk_size=chunk,
        n_detectors=circuit.num_detectors,
        n_observables=circuit.num_observables,
        environments=(environment,),
        contract=Contract.DEM_MECHANISM if emit else Contract.LOGICAL_FRAME,
        structure_level=level,
        drift_axis=environment.axis,
        structure_source_environment_id=0 if structure is not None else None,
        structure_dem_sha=dem_digest(str(dem))
        if dem is not None and structure is not None
        else None,
        n_mechanisms=dem.num_errors if emit and dem is not None else None,
        mechanism_source_environment_id=0 if emit else None,
        generation_config=config,
        generation_audit=audit,
    )
    return PreparedGeneration(meta, structure, chunks)


def write_configured(
    config: dict[str, Any],
    path: Path,
    progress: Callable[[int], None] | None = None,
    on_phase: Callable[[str], None] | None = None,
) -> DatasetMeta:
    """Write into a caller-owned staging directory, never directly to the public path."""
    if on_phase is not None:
        on_phase("preparing source and model")
    prepared = prepare_generation(config)
    meta = prepared.meta
    if on_phase is not None:
        on_phase("sampling" if config["mode"] != "hardware" else "importing")
    hasher = StreamingContentHasher(meta.contract is Contract.DEM_MECHANISM)
    writer = (
        StreamingHDF5Writer(path, meta.n_detectors, meta.n_observables)
        if config["output"]["format"] == "hdf5"
        else None
    )
    detector_chunks, observable_chunks, mechanism_chunks = [], [], []
    written = 0
    try:
        for batch in prepared.chunks:
            hasher.update(batch.detectors, batch.observables, batch.mechanisms)
            written += batch.n_shots
            if writer is not None:
                writer.append(batch.detectors, batch.observables, None, batch.mechanisms)
            else:
                detector_chunks.append(batch.detectors)
                observable_chunks.append(batch.observables)
                if batch.mechanisms is not None:
                    mechanism_chunks.append(batch.mechanisms)
            if progress is not None:
                progress(batch.n_shots)
        if written != meta.shots:
            raise RuntimeError("Sampler returned a different shot count from its configuration")
        if on_phase is not None:
            on_phase("writing")
        meta = replace(
            meta,
            content_hash=hasher.hexdigest(
                written,
                meta.n_detectors,
                meta.n_observables,
                meta.n_mechanisms,
            ),
        )
        if writer is not None:
            writer.close(meta, prepared.structure)
        else:
            dataset = InMemoryDataset(
                np.concatenate(detector_chunks),
                np.concatenate(observable_chunks),
                meta,
                mechanisms=np.concatenate(mechanism_chunks) if mechanism_chunks else None,
                structure=prepared.structure,
            )
            get_exporter(config["output"]["format"]).write(dataset, path, meta.structure_level)
    except BaseException:
        if writer is not None:
            writer.abort()
        raise
    return meta
