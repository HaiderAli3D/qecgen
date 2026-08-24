"""The canonical in-memory model, its manifest, and streaming read/write protocols.

Three distinct objects, because one class cannot both hold everything in memory and
stream: :class:`InMemoryDataset`, :class:`DatasetReader`, :class:`StreamingDatasetWriter`.

There is deliberately **no top-level ``p``, ``circuit`` or ``dem``**. Those are
per-environment properties held in :class:`EnvironmentSpec`. Promoting them to the
dataset level is wrong the moment there is more than one environment, and a
single-environment dataset is simply a list of length one.
"""

from __future__ import annotations

import datetime as dt
import enum
import functools
import hashlib
import importlib.metadata as md
import json
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from qecgen import __version__
from qecgen.circuits import Basis, ChannelVector, NoiseModel
from qecgen.dem import DemStructure
from qecgen.sampling import packed_width

__all__ = [
    "CSV_SPELLING",
    "DETECTOR_PREFIX",
    "ENVIRONMENT_COLUMN",
    "MECHANISM_PREFIX",
    "OBSERVABLE_PREFIX",
    "SHOT_COLUMN",
    "TARGET_COLUMN",
    "ColumnSpelling",
    "Contract",
    "DatasetMeta",
    "DatasetReader",
    "DriftCondition",
    "EnvironmentSpec",
    "InMemoryDataset",
    "StreamingContentHasher",
    "StreamingDatasetWriter",
    "StructureLevel",
    "content_hash",
    "dem_digest",
    "git_commit",
    "library_versions",
    "target_columns",
]


class Contract(enum.StrEnum):
    """Which prediction target a file carries. See ``DATA_CONTRACT.md``."""

    LOGICAL_FRAME = "logical_frame"
    """Contract A: detection events -> logical observable flips. The default."""

    DEM_MECHANISM = "dem_mechanism"
    """Contract B: additionally records which abstract DEM mechanisms fired.

    Not physical Pauli faults. Contract C (physical corrections) is not implemented
    and is underdetermined as specified.
    """


class StructureLevel(enum.StrEnum):
    """How much structural information to export alongside the shots."""

    NONE = "none"
    """Shots and manifest only."""

    COORDS = "coords"
    """Adds detector coordinates (x, y, t): graph structure without the error model."""

    DEM = "dem"
    """Adds H, L, priors and component structure."""

    FULL = "full"
    """Everything, including per-environment circuit and DEM text."""


class DriftCondition(enum.StrEnum):
    """Where a test file's exported structure was derived from.

    This distinction is the whole point of the drift study and is never inferred.
    """

    ORACLE_CALIBRATED = "oracle_calibrated"
    """Structure derived from the test file's own environment.

    Hands the decoder the true test-time noise distribution. Legitimate as a ceiling
    measurement, but invalidates any generalisation claim.
    """

    FROZEN_PRIOR = "frozen_prior"
    """Structure derived from the nominated training environment.

    The decoder must genuinely generalise to unseen noise.
    """

    NOT_APPLICABLE = "not_applicable"
    """Single-environment datasets that are not part of a drift study."""


SHOT_COLUMN = "shot"
ENVIRONMENT_COLUMN = "environment_id"
DETECTOR_PREFIX = "det_"
OBSERVABLE_PREFIX = "obs_"
MECHANISM_PREFIX = "mech_"
"""Column names and per-bit prefixes for the formats that expand bits into columns.

They live here rather than in ``exporters/csv_table.py``, where they were written,
because :meth:`DatasetMeta.schema_block` now names the same columns. A second copy of
``"det_"`` would let the manifest promise a column the writer does not emit, and the
manifest is the half a reader cannot check against the file.

These are **column** names. They are deliberately not the array names that
:func:`content_hash` folds into its digest -- see the comment there.
"""


TARGET_COLUMN = "target"
"""What the target column is called, in every tabular format.

The data consumer this is for asks for "Primary Key, Target, Variable_1, ..." -- so the
column that holds the answer says so in its own name rather than requiring the manifest to
be read first. It is deliberately *not* spelled per-format the way the bit prefixes are:
`det_0` versus `detector_00` is a difference in how an index is rendered, while this is one
column with one job.
"""


def target_columns(n_observables: int) -> list[str]:
    """The target column names for a file with ``n_observables`` observables.

    A single observable -- which is every surface-code memory experiment -- gets the bare
    name ``target``. More than one is indexed, because there is then no single answer
    column and a bare name would have to mean one of them arbitrarily.

    One implementation, used by both tabular formats and by
    :meth:`DatasetMeta.schema_block`, so the manifest cannot name a column the writer does
    not emit.
    """
    if n_observables == 1:
        return [TARGET_COLUMN]
    return [f"{TARGET_COLUMN}_{i}" for i in range(n_observables)]


@dataclass(frozen=True, slots=True)
class ColumnSpelling:
    """How one tabular format spells its per-bit column names.

    ``csv`` writes ``det_0`` unpadded. ``ml_csv`` writes ``detector_00``, zero-padded so
    that a column sort -- and every column sort hiding inside a join, a concat or a
    feature-store schema -- cannot put ``detector_10`` before ``detector_2`` and permute
    the feature matrix silently.

    One manifest field cannot honestly name both spellings, and a manifest naming the
    wrong one would promise columns the file does not contain: exactly the over-claim
    :meth:`DatasetMeta.schema_block` exists to prevent. So the format writing a file
    states its own spelling *in that file*, and a consumer resolves names from the
    spelling it actually finds rather than from a convention it has to know.

    The pad width is derived per array from that array's own width, so it is a property
    of the file rather than a constant a consumer could hardcode -- ``detector_00`` at
    rotated d=3 (24 detectors) but ``detector_000`` at d=7. That is exactly why the names
    are published rather than described.
    """

    detector: str = DETECTOR_PREFIX
    observable: str = OBSERVABLE_PREFIX
    mechanism: str = MECHANISM_PREFIX
    pad: bool = False

    def pad_width(self, width: int) -> int:
        """Digits needed for the largest index of an array this wide. 1 when unpadded."""
        if not self.pad:
            return 1
        return len(str(max(width - 1, 0)))

    def columns(self, prefix: str, width: int) -> list[str]:
        """The column names this spelling gives an array of ``width`` bits."""
        digits = self.pad_width(width)
        return [f"{prefix}{i:0{digits}d}" for i in range(width)]


CSV_SPELLING = ColumnSpelling()
"""The canonical spelling: ``det_0``, unpadded. What ``csv`` writes."""


@dataclass(frozen=True, slots=True)
class EnvironmentSpec:
    """One noise environment. A dataset holds one or more of these."""

    environment_id: int
    p: float
    noise_model: NoiseModel
    channels: ChannelVector
    circuit: str
    """Full Stim circuit text, sufficient to regenerate this environment exactly."""
    dem: str
    """Full decomposed detector error model text."""
    shots: int
    axis: str = "p"
    """Which drift axis this environment varies. ``"p"`` for plain rate sweeps."""
    axis_value: float = 0.0
    """This environment's position on ``axis``."""

    def to_json_dict(self) -> dict[str, Any]:
        """Serialise the environment's **parameters** only.

        Circuit and DEM text are deliberately excluded and live in the separate
        provenance block (:meth:`provenance_dict`). Under ``FROZEN_PRIOR`` a test file's
        own DEM text is exactly the information the condition exists to withhold, so it
        must never appear in the decoder-visible manifest.

        The parameters here are sufficient to regenerate the environment: distance,
        rounds, basis and rotated live on the dataset manifest, and the channel vector,
        axis and axis value live here.
        """
        return {
            "environment_id": self.environment_id,
            "p": self.p,
            "noise_model": str(self.noise_model),
            "channels": self.channels.as_dict(),
            "shots": self.shots,
            "axis": self.axis,
            "axis_value": self.axis_value,
        }

    def provenance_dict(self) -> dict[str, Any]:
        """Serialise the circuit and DEM text for the provenance block.

        **Not decoder-visible.** See ``DATA_CONTRACT.md``.
        """
        return {
            "environment_id": self.environment_id,
            "circuit": self.circuit,
            "dem": self.dem,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> EnvironmentSpec:
        """Rebuild from :meth:`to_json_dict`.

        The channels dict must be complete. ``ChannelVector(**partial)`` fills absent
        keys from the dataclass defaults — all 0.0 — so a corrupt or foreign manifest
        missing a channel would silently read back as (partially) noiseless instead of
        failing, against this module's strict-parsing rule.
        """
        channels = dict(data["channels"])
        missing = sorted({f.name for f in fields(ChannelVector)} - set(channels))
        if missing:
            raise ValueError(
                f"manifest channels dict is missing {missing}; a missing channel "
                "would silently default to 0.0"
            )
        return cls(
            environment_id=int(data["environment_id"]),
            p=float(data["p"]),
            noise_model=NoiseModel(data["noise_model"]),
            channels=ChannelVector(**channels),
            circuit=str(data.get("circuit", "")),
            dem=str(data.get("dem", "")),
            shots=int(data["shots"]),
            axis=str(data.get("axis", "p")),
            axis_value=float(data.get("axis_value", 0.0)),
        )


def _strict_bool(value: Any, field_name: str) -> bool:
    """Accept only a real boolean, never a truthy string.

    ``bool("false")`` is ``True``. Reading a manifest that carries the *string*
    ``"false"`` for ``rotated`` would silently switch the code layout, so this refuses
    rather than guesses.
    """
    if isinstance(value, bool):
        return value
    raise ValueError(
        f"manifest field {field_name!r} must be a JSON boolean, got {type(value).__name__} "
        f"{value!r}; string values are refused because bool('false') is True"
    )


def _schema_projection(block: dict[str, Any]) -> dict[str, Any]:
    """The load-bearing half of a schema block, for comparison.

    Only the claims a consumer acts on: which arrays are inputs, which are targets, and
    each entry's role, presence and width. Prose (``meaning``, ``caveat``, ``note``) is
    excluded so a future writer improving a sentence is not refused as corrupt, while a
    writer that moved a target is.
    """
    roles = block.get("roles") or {}
    return {
        "features": block.get("features"),
        "targets": block.get("targets"),
        "primary_target": block.get("primary_target"),
        "roles": {
            name: (entry.get("role"), entry.get("present"), entry.get("width"))
            for name, entry in sorted(roles.items())
        },
    }


def _require_schema_agreement(stored: Any, derived: dict[str, Any]) -> None:
    """Refuse a manifest whose stored schema block contradicts the derived one.

    Absent is fine: every manifest written before this block existed lacks it, and it is
    recomputed from ``contract`` and the width fields on read, so nothing is lost.

    Present and different is refused rather than resolved. The stored block and the
    fields it derives from are two descriptions of one dataset, and preferring either is
    a guess about which of them is the corrupt half -- the same reasoning
    ``csv_table._require_column_agreement`` applies to a header row that disagrees with
    its manifest.
    """
    if stored is None:
        return
    if not isinstance(stored, dict):
        raise ValueError(
            f"manifest field 'schema' must be a JSON object, got {type(stored).__name__}"
        )
    if _schema_projection(stored) != _schema_projection(derived):
        raise ValueError(
            "manifest 'schema' block disagrees with the fields it is derived from "
            "(contract, n_detectors, n_observables, n_mechanisms, structure_level). "
            f"Stored: {_schema_projection(stored)}. Derived: {_schema_projection(derived)}. "
            "Refused rather than resolved: preferring either would guess which half is "
            "corrupt."
        )


def dem_digest(dem_text: str) -> str:
    """Short digest of a DEM's text, used to identify structure provenance."""
    return hashlib.blake2b(dem_text.encode("utf-8"), digest_size=16).hexdigest()


def library_versions() -> dict[str, str]:
    """Versions of every library whose behaviour affects the generated bytes."""
    out: dict[str, str] = {"qecgen": __version__}
    for pkg in ("stim", "sinter", "pymatching", "numpy", "scipy", "h5py", "pyarrow"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:  # pragma: no cover - all are hard deps
            out[pkg] = "not-installed"
    return out


@functools.lru_cache(maxsize=8)
def _git_commit_cached(cwd: str | None) -> str | None:
    """One ``git rev-parse`` per working directory per process.

    Every :class:`DatasetMeta` construction asks for the commit, and a drift study builds
    one manifest per file, so an uncached call means a process spawn per output file for
    a value that cannot change mid-run. Caching also makes every file produced by one
    invocation agree on the commit, which is the honest answer when they were all
    generated together.
    """
    # No pipes, deliberately. With `capture_output=True`, `subprocess.run` reads through
    # reader threads and joins them *after* a timeout kills the child -- and git spawns
    # helpers that inherit the pipe handles, so those threads can wait on an EOF that
    # never comes and the `timeout` argument stops bounding anything. Observed here as a
    # generation run wedged forever inside DatasetMeta.__init__. Writing to a real file
    # means there are no reader threads to join and the timeout does what it says.
    # stdin is /dev/null so git can never sit waiting on a prompt either.
    try:
        with tempfile.TemporaryFile() as sink:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=sink,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
            if result.returncode != 0:
                return None
            sink.seek(0)
            return sink.read().decode("utf-8", errors="replace").strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def git_commit(cwd: Path | None = None) -> str | None:
    """Current git commit, or None when not in a repository.

    Returning None is a real outcome, not an error: this tool is usable outside a
    checkout, and the manifest records the absence rather than inventing a value.

    Resolved once per working directory per process; see :func:`_git_commit_cached`.
    """
    return _git_commit_cached(str(cwd) if cwd is not None else None)


def content_hash(
    detectors: np.ndarray,
    observables: np.ndarray,
    environment_ids: np.ndarray | None = None,
    mechanisms: np.ndarray | None = None,
) -> str:
    """Deterministic hash of array *contents*, excluding all manifest metadata.

    Determinism is content-based, not file-based. Manifests contain timestamps and git
    commits, so byte-identical files are not a meaningful reproducibility test. This
    hash covers dtype, shape and bytes of each array and nothing else.

    Reproducible only under an identical Stim version, machine and **chunk size**;
    changing the chunk size changes the sample stream.
    """
    hasher = hashlib.blake2b(digest_size=32)
    # These four strings are the digest's alphabet, not a naming registry. They are fed
    # into the hash itself, so rewiring them to any shared constant -- including
    # DatasetMeta.schema_block's role names, which spell this one "environment_id" --
    # silently changes every content_hash ever computed. Keep them literal here.
    named: list[tuple[str, np.ndarray | None]] = [
        ("detectors", detectors),
        ("observables", observables),
        ("environment_ids", environment_ids),
        ("mechanisms", mechanisms),
    ]
    for name, array in named:
        hasher.update(name.encode("utf-8"))
        if array is None:
            hasher.update(b"\x00none")
            continue
        contiguous = np.ascontiguousarray(array)
        hasher.update(str(contiguous.dtype).encode("utf-8"))
        hasher.update(str(contiguous.shape).encode("utf-8"))
        # The payload is folded in as its own digest rather than raw bytes so the
        # streaming writer can accumulate it chunk by chunk and still arrive at the
        # same value, despite only learning the final shape at the end.
        hasher.update(hashlib.blake2b(contiguous.tobytes(), digest_size=32).digest())
    return hasher.hexdigest()


class StreamingContentHasher:
    """Accumulates a :func:`content_hash` over chunks, without holding them.

    Produces exactly the digest :func:`content_hash` would give for the concatenated
    arrays, so a streamed file and a materialised one are interchangeable.
    """

    def __init__(self, with_mechanisms: bool = False) -> None:
        self._with_mechanisms = with_mechanisms
        self._payload = {
            name: hashlib.blake2b(digest_size=32)
            for name in ("detectors", "observables", "mechanisms")
        }
        self._widths: dict[str, int] = {}
        self._dtypes: dict[str, str] = {}

    def update(
        self,
        detectors: np.ndarray,
        observables: np.ndarray,
        mechanisms: np.ndarray | None = None,
    ) -> None:
        """Fold one chunk into the running digest."""
        for name, array in (
            ("detectors", detectors),
            ("observables", observables),
            ("mechanisms", mechanisms),
        ):
            if array is None:
                continue
            contiguous = np.ascontiguousarray(array)
            self._payload[name].update(contiguous.tobytes())
            self._widths[name] = int(contiguous.shape[1])
            self._dtypes[name] = str(contiguous.dtype)

    def hexdigest(
        self,
        rows: int,
        n_detectors: int,
        n_observables: int,
        n_mechanisms: int | None = None,
    ) -> str:
        """Finalise. ``rows`` is the total shot count actually written.

        Widths fall back to the true packed widths rather than to 0: with zero chunks
        nothing was remembered from ``update``, and hashing shape ``(0, 0)`` instead of
        the materialised path's ``(0, packed_width)`` broke the documented streamed/
        materialised digest parity exactly at the degenerate case. For any non-empty
        stream the two sources agree — stim's bit-packed arrays are ``packed_width``
        wide by construction.
        """
        fallback_widths = {
            "detectors": packed_width(n_detectors),
            "observables": packed_width(n_observables),
            "mechanisms": packed_width(n_mechanisms) if n_mechanisms is not None else 0,
        }
        hasher = hashlib.blake2b(digest_size=32)
        # Must stay byte-identical to content_hash's `named` list, and literal for the
        # same reason: these strings are hashed, not displayed.
        order = ["detectors", "observables", "environment_ids", "mechanisms"]
        for name in order:
            hasher.update(name.encode("utf-8"))
            if name == "environment_ids" or (name == "mechanisms" and not self._with_mechanisms):
                hasher.update(b"\x00none")
                continue
            hasher.update(self._dtypes.get(name, "uint8").encode("utf-8"))
            width = self._widths.get(name, fallback_widths[name])
            hasher.update(str((rows, width)).encode("utf-8"))
            hasher.update(self._payload[name].digest())
        return hasher.hexdigest()


@dataclass(frozen=True)
class DatasetMeta:
    """Everything needed to regenerate a file exactly, minus the shots themselves."""

    distance: int
    rounds: int
    basis: Basis
    rotated: bool
    shots: int
    seed: int
    chunk_size: int
    n_detectors: int
    n_observables: int
    environments: tuple[EnvironmentSpec, ...]

    contract: Contract = Contract.LOGICAL_FRAME
    structure_level: StructureLevel = StructureLevel.NONE
    drift_condition: DriftCondition = DriftCondition.NOT_APPLICABLE
    drift_axis: str = "p"
    structure_source_environment_id: int | None = None
    """Which environment the exported structure came from.

    Under ``FROZEN_PRIOR`` this is the training environment and will differ from the
    environments present in the file. Recording it makes the condition auditable rather
    than a claim.
    """

    bit_order: str = "little"
    """Always ``"little"``. Recorded explicitly because NumPy's default is the opposite."""

    n_mechanisms: int | None = None
    """Mechanism count of the DEM the Contract B labels were **sampled from**.

    Set whenever mechanisms are emitted, independent of ``structure_level``. Taking it
    from the exported structure instead would mislabel column widths under
    ``FROZEN_PRIOR``, where the shipped structure and the sampling DEM are different
    objects.
    """

    mechanism_source_environment_id: int | None = None
    """Which environment's DEM defines the meaning of each mechanism column."""

    structure_dem_sha: str | None = None
    """Digest of the DEM text the exported structure was derived from.

    Lets a reviewer confirm which environment supplied the structure without the file
    having to carry the DEM text itself. The name says "sha" but the digest is BLAKE2b —
    see ``structure_dem_algorithm``; the key itself is kept for compatibility with
    already-written manifests.
    """

    structure_dem_algorithm: str = "blake2b-128"
    """Algorithm behind ``structure_dem_sha``, named so a checker can reproduce it.

    The same trap ``content_hash_algorithm`` exists for: a field whose name implies
    SHA while the digest is BLAKE2b fails any independent verification attempted with
    the advertised algorithm.
    """

    bias_scope: str | None = None
    """Which channels an ``xz_bias`` rewrite touched, when that axis is in use."""

    shuffle_seed: int | None = None
    content_hash: str | None = None
    content_hash_algorithm: str = "blake2b-256"
    """Named explicitly so an external checker can reproduce it.

    This is BLAKE2b with a 32-byte digest, **not** SHA-256. The field was previously
    called ``content_sha256``, which was simply wrong and would fail any independent
    verification attempted with the advertised algorithm.
    """

    versions: dict[str, str] = field(default_factory=library_versions)
    git_commit: str | None = field(default_factory=git_commit)
    generated_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    )
    notes: str = (
        "Syndrome-to-logical-frame dataset. Detection events -> logical observable flips. "
        "Contains no physical Pauli fault labels; see DATA_CONTRACT.md."
    )

    def schema_block(self, spelling: ColumnSpelling = CSV_SPELLING) -> dict[str, Any]:
        """Which arrays are inputs, which are targets, and what everything else is for.

        This exists because a consumer opened a generated file and could not tell which
        column was the label. Nothing in the file said so: the target was inferable only
        by reading ``contract`` and then ``DATA_CONTRACT.md``. That is a convention, and
        a convention is exactly what a data pipeline cannot read.

        **Computed here, never stored as a field.** ``jsonl`` and ``parquet`` serialise
        ``dataclasses.replace(meta, structure_level=recorded_structure_level(...))``, so
        a stored block would go on claiming ``provenance: always`` on a file whose level
        had just been downgraded from ``full`` to ``dem`` -- an over-claim in the one
        field a reader cannot check against the file, which is the drift
        ``recorded_structure_level`` exists to prevent. Deriving it at serialisation makes
        that impossible rather than merely tested.

        **The column spelling comes from the format, not from here.** ``csv`` writes
        ``det_0`` unpadded; ``ml_csv`` writes ``detector_00``. A single hardcoded prefix
        would make one of those two manifests promise columns its own file does not
        contain, so the writing format passes its :class:`ColumnSpelling` and the block
        publishes ``csv_prefix`` plus ``csv_pad_width`` for the file in hand. Resolve a
        column as ``f"{csv_prefix}{i:0{csv_pad_width}d}"``; at pad width 1 that is plain
        decimal, so one expression serves both spellings.

        A role whose names are not an index under a prefix publishes ``csv_names``
        instead, and that list wins where it is present. The target uses it: with one
        observable the column is the bare name ``target``, which no prefix-plus-index
        rule can express.

        There is no ``physical_faults`` entry, not even an absent one -- an ``"absent"``
        entry reads as "coming soon". Contract C is refused, not pending, and ``note``
        says so in prose.
        """
        mechanisms_present = self.contract is Contract.DEM_MECHANISM
        has_structure = self.structure_level is not StructureLevel.NONE
        has_provenance = self.structure_level is StructureLevel.FULL
        roles: dict[str, Any] = {
            SHOT_COLUMN: {
                "role": "row_index",
                "present": "csv_only",
                "meaning": (
                    "Row number. It must equal the row's position in the file: sorting "
                    "the table severs a shot's detectors from its observables and leaves "
                    "a file that still parses."
                ),
            },
            ENVIRONMENT_COLUMN: {
                "role": "grouping_key",
                "present": "if_pooled",
                "meaning": (
                    "Which noise environment this shot came from. Metadata for grouping "
                    "and splitting, never an input: a model handed it can read the noise "
                    "level off the row instead of learning the physics."
                ),
                "note": (
                    "Look for the column or array itself. A file pooling a single "
                    "environment carries it too, so the environment count in this "
                    "manifest does not decide its presence."
                ),
            },
            "detectors": {
                "role": "feature",
                "present": "always",
                "width": self.n_detectors,
                "csv_prefix": spelling.detector,
                "csv_pad_width": spelling.pad_width(self.n_detectors),
                "meaning": (
                    "One bit per detector: did this parity check disagree with its "
                    "previous value on this shot. This is the entire input a decoder is "
                    "allowed to see."
                ),
            },
            "observables": {
                "role": "target",
                "present": "always",
                "width": self.n_observables,
                "csv_names": target_columns(self.n_observables),
                "target_of": str(Contract.LOGICAL_FRAME),
                "meaning": (
                    "One bit per logical observable: did the encoded logical qubit end up "
                    "flipped. This is the value a decoder must predict."
                ),
            },
            "mechanisms": {
                "role": "target",
                "present": "always" if mechanisms_present else "absent",
                "width": self.n_mechanisms,
                "csv_prefix": spelling.mechanism,
                "csv_pad_width": spelling.pad_width(self.n_mechanisms or 0),
                "target_of": str(Contract.DEM_MECHANISM),
                "meaning": (
                    "Which abstract mechanisms of the decomposed detector error model "
                    "fired on this shot. Written only under --emit-mechanisms."
                ),
                "caveat": (
                    "NOT physical Pauli faults. Many physically distinct gate-level "
                    "faults with identical detector signatures collapse into one "
                    "mechanism, and the index is an artifact of DEM construction order: "
                    "not portable across noise models, distances or Stim versions."
                ),
            },
            "dem": {
                "role": "side_information",
                "present": "always" if has_structure else "absent",
                "meaning": (
                    "Decoder calibration data: H, L, priors, decomposed components and "
                    "detector coordinates. A property of the noise model, not of any "
                    "shot, so it has no row correspondence and is not a feature. Under "
                    "frozen_prior it describes the nominated training environment, not "
                    "this file's own."
                ),
            },
            "provenance": {
                "role": "never_read",
                "present": "always" if has_provenance else "absent",
                "meaning": (
                    "Circuit and error-model text. A decoder, or any harness feeding one, "
                    "must never read this. Under frozen_prior it holds the test "
                    "environment's own DEM, which is precisely the distribution the "
                    "condition exists to withhold."
                ),
            },
        }
        return {
            "schema_version": 1,
            "features": ["detectors"],
            "targets": ["observables", "mechanisms"] if mechanisms_present else ["observables"],
            "primary_target": "observables",
            "roles": roles,
            "note": (
                "Detection events in, logical observable flips out. No column in this or "
                "any qecgen file is a physical Pauli fault label: that target (Contract C) "
                "is not implemented and is underdetermined as specified. See "
                "DATA_CONTRACT.md."
            ),
        }

    def to_json_dict(self, spelling: ColumnSpelling = CSV_SPELLING) -> dict[str, Any]:
        """Serialise the **decoder-visible** manifest to plain JSON-compatible types.

        Never contains circuit or DEM text. See :meth:`provenance_dict`.
        """
        return {
            "distance": self.distance,
            "rounds": self.rounds,
            "basis": str(self.basis),
            "rotated": self.rotated,
            "shots": self.shots,
            "seed": self.seed,
            "chunk_size": self.chunk_size,
            "n_detectors": self.n_detectors,
            "n_observables": self.n_observables,
            "n_mechanisms": self.n_mechanisms,
            "mechanism_source_environment_id": self.mechanism_source_environment_id,
            "contract": str(self.contract),
            "structure_level": str(self.structure_level),
            "drift_condition": str(self.drift_condition),
            "drift_axis": self.drift_axis,
            "structure_source_environment_id": self.structure_source_environment_id,
            "structure_dem_sha": self.structure_dem_sha,
            "structure_dem_algorithm": self.structure_dem_algorithm,
            "bias_scope": self.bias_scope,
            "bit_order": self.bit_order,
            "shuffle_seed": self.shuffle_seed,
            "content_hash": self.content_hash,
            "content_hash_algorithm": self.content_hash_algorithm,
            "versions": self.versions,
            "git_commit": self.git_commit,
            "generated_at": self.generated_at,
            "notes": self.notes,
            "schema": self.schema_block(spelling),
            "environments": [e.to_json_dict() for e in self.environments],
        }

    def provenance_dict(self) -> dict[str, Any]:
        """Serialise the circuit and DEM text for every environment.

        **This block is not decoder-visible.** It is written only at
        ``--structure full``, and exporters must store it physically apart from the
        manifest (an HDF5 ``/provenance`` group, a distinct key elsewhere) so that
        reading the manifest cannot expose it.

        Under ``FROZEN_PRIOR`` this contains the *test* environment's DEM, which is
        precisely the distribution the condition withholds from the decoder.
        """
        return {
            "warning": (
                "PROVENANCE ONLY. A decoder must not read this block. Under "
                "frozen_prior it contains the test environment's own DEM, which "
                "reveals the true test-time noise distribution."
            ),
            "environments": [e.provenance_dict() for e in self.environments],
        }

    def to_json(self, spelling: ColumnSpelling = CSV_SPELLING) -> str:
        """Serialise to a JSON string, as stored in NPZ / Parquet / HDF5 attributes.

        ``allow_nan=False`` so a non-finite value can never be written as the bare
        token ``NaN``, which is not valid JSON and which strict parsers reject.
        """
        return json.dumps(self.to_json_dict(spelling), sort_keys=True, allow_nan=False)

    def provenance_json(self) -> str:
        """Serialise the provenance block to a JSON string."""
        return json.dumps(self.provenance_dict(), sort_keys=True, allow_nan=False)

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> DatasetMeta:
        """Rebuild a manifest from :meth:`to_json_dict`.

        Parsing is strict. ``bool("false")`` is ``True`` in Python, so a hand-edited or
        foreign manifest carrying the string ``"false"`` would silently flip the code
        layout and every downstream result with it.
        """
        meta = cls(
            distance=int(data["distance"]),
            rounds=int(data["rounds"]),
            basis=Basis(data["basis"]),
            rotated=_strict_bool(data["rotated"], "rotated"),
            shots=int(data["shots"]),
            seed=int(data["seed"]),
            chunk_size=int(data["chunk_size"]),
            n_detectors=int(data["n_detectors"]),
            n_observables=int(data["n_observables"]),
            environments=tuple(
                EnvironmentSpec.from_json_dict(e) for e in data.get("environments", [])
            ),
            contract=Contract(data["contract"]),
            structure_level=StructureLevel(data["structure_level"]),
            drift_condition=DriftCondition(data["drift_condition"]),
            drift_axis=str(data.get("drift_axis", "p")),
            structure_source_environment_id=data.get("structure_source_environment_id"),
            structure_dem_sha=data.get("structure_dem_sha"),
            structure_dem_algorithm=str(data.get("structure_dem_algorithm", "blake2b-128")),
            bias_scope=data.get("bias_scope"),
            bit_order=str(data["bit_order"]),
            n_mechanisms=data.get("n_mechanisms"),
            mechanism_source_environment_id=data.get("mechanism_source_environment_id"),
            shuffle_seed=data.get("shuffle_seed"),
            content_hash=data.get("content_hash"),
            content_hash_algorithm=str(data.get("content_hash_algorithm", "blake2b-256")),
            versions=dict(data.get("versions", {})),
            git_commit=data.get("git_commit"),
            generated_at=str(data.get("generated_at", "")),
            notes=str(data.get("notes", "")),
        )
        _require_schema_agreement(data.get("schema"), meta.schema_block())
        return meta

    @classmethod
    def from_json(cls, text: str) -> DatasetMeta:
        """Rebuild a manifest from a JSON string."""
        return cls.from_json_dict(json.loads(text))


@dataclass(frozen=True)
class InMemoryDataset:
    """A dataset small enough to hold entirely in memory."""

    detectors: np.ndarray
    """uint8, ``(shots, ceil(n_detectors/8))``, little-endian bit-packed."""

    observables: np.ndarray
    """uint8, ``(shots, ceil(n_observables/8))``, little-endian bit-packed."""

    meta: DatasetMeta

    environment_ids: np.ndarray | None = None
    """int32 ``(shots,)``, one per shot. None for single-environment datasets."""

    mechanisms: np.ndarray | None = None
    """Contract B labels, or None."""

    structure: DemStructure | None = None

    def __post_init__(self) -> None:
        n = self.detectors.shape[0]
        for name in ("observables", "environment_ids", "mechanisms"):
            array = getattr(self, name)
            if array is not None and array.shape[0] != n:
                raise ValueError(
                    f"{name} has {array.shape[0]} rows but detectors has {n}; "
                    "per-shot correspondence is broken"
                )

    @property
    def n_shots(self) -> int:
        """Number of shots held."""
        return int(self.detectors.shape[0])

    def compute_content_hash(self) -> str:
        """Content hash of this dataset's arrays."""
        return content_hash(self.detectors, self.observables, self.environment_ids, self.mechanisms)

    def unpacked_detectors(self) -> np.ndarray:
        """Detection events as bool ``(shots, n_detectors)``."""
        return np.unpackbits(
            self.detectors, axis=1, count=self.meta.n_detectors, bitorder="little"
        ).astype(bool)

    def unpacked_observables(self) -> np.ndarray:
        """Observable flips as bool ``(shots, n_observables)``."""
        return np.unpackbits(
            self.observables, axis=1, count=self.meta.n_observables, bitorder="little"
        ).astype(bool)


@runtime_checkable
class DatasetReader(Protocol):
    """Lazy, chunk-iterating view over a dataset on disk."""

    @property
    def meta(self) -> DatasetMeta:
        """The manifest, read without loading the shots."""
        ...

    def iter_chunks(self, chunk_size: int | None = None) -> Iterator[InMemoryDataset]:
        """Iterate the file in chunks, never materialising all shots."""
        ...

    def read_all(self) -> InMemoryDataset:
        """Materialise the whole dataset. Only safe when it fits in memory."""
        ...

    def close(self) -> None:
        """Release the underlying file handle."""
        ...


@runtime_checkable
class StreamingDatasetWriter(Protocol):
    """Append-only writer that finalises its manifest on close."""

    def append(
        self,
        detectors: np.ndarray,
        observables: np.ndarray,
        environment_ids: np.ndarray | None = None,
        mechanisms: np.ndarray | None = None,
    ) -> None:
        """Append one chunk. Arrays must correspond row-for-row."""
        ...

    def close(self, meta: DatasetMeta, structure: DemStructure | None = None) -> None:
        """Write the manifest and structure, then close."""
        ...
