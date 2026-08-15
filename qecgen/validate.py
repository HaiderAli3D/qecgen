"""Fast, deterministic structural validation. Runs by default.

Deliberately separate from :mod:`qecgen.qa`. Mixing deterministic structural checks with
statistical ones produces a default test suite that fails for sampling reasons, and a
suite that fails intermittently is a suite that gets ignored.

Nothing here samples, estimates or compares confidence intervals. Every check is a
property of the file as written.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim

from qecgen.dataset import Contract, InMemoryDataset
from qecgen.dem import DemStructure
from qecgen.sampling import packed_width

__all__ = ["CheckResult", "ValidationReport", "validate_dataset", "validate_dem_structure"]


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One named check."""

    name: str
    passed: bool
    detail: str
    """What was actually observed. Written as a neutral observation, not as a
    failure message, because it is displayed for passing checks too."""

    requirement: str = ""
    """What was required. Appended only when the check fails."""

    def __str__(self) -> str:
        """Render as a single status line."""
        status = "PASS" if self.passed else "FAIL"
        suffix = "" if self.passed or not self.requirement else f" -- expected {self.requirement}"
        return f"[{status}] {self.name}: {self.detail}{suffix}"


@dataclass(frozen=True)
class ValidationReport:
    """The full set of check results."""

    results: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        """True when every check passed."""
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        """Only the failing checks."""
        return tuple(r for r in self.results if not r.passed)

    def __str__(self) -> str:
        """Render every check, one per line."""
        return "\n".join(str(r) for r in self.results)


def validate_dataset(dataset: InMemoryDataset, check_hash: bool = True) -> ValidationReport:
    """Run every fast structural check against a dataset.

    Args:
        dataset: The dataset to check.
        check_hash: Recompute and compare the manifest's content hash.

    Returns:
        A :class:`ValidationReport`; inspect ``.ok`` and ``.failures``.
    """
    results: list[CheckResult] = []
    meta = dataset.meta
    n_shots = dataset.n_shots

    # --- shapes and dtypes -------------------------------------------------
    for name in ("detectors", "observables", "mechanisms"):
        array = getattr(dataset, name)
        if array is None:
            continue
        results.append(
            CheckResult(
                f"{name}.dtype",
                array.dtype == np.uint8,
                f"{array.dtype}",
                "uint8 (packed); a float array here means something reconstructed it wrongly",
            )
        )
    expected_det_width = packed_width(meta.n_detectors)
    results.append(
        CheckResult(
            "detectors.packed_width",
            dataset.detectors.shape[1] == expected_det_width,
            f"ceil({meta.n_detectors}/8) = {expected_det_width}, got {dataset.detectors.shape[1]}",
        )
    )
    expected_obs_width = packed_width(meta.n_observables)
    results.append(
        CheckResult(
            "observables.packed_width",
            dataset.observables.shape[1] == expected_obs_width,
            f"ceil({meta.n_observables}/8) = {expected_obs_width}, "
            f"got {dataset.observables.shape[1]}",
        )
    )
    results.append(
        CheckResult(
            "manifest.shots",
            meta.shots == n_shots,
            f"manifest says {meta.shots}, arrays hold {n_shots}",
        )
    )

    # --- bit order ---------------------------------------------------------
    results.append(
        CheckResult(
            "bit_order",
            meta.bit_order == "little",
            f"recorded as {meta.bit_order!r}",
            "'little' -- numpy's packbits default is 'big', the opposite convention",
        )
    )

    # --- per-shot correspondence ------------------------------------------
    for name in ("observables", "environment_ids", "mechanisms"):
        array = getattr(dataset, name)
        if array is None:
            continue
        results.append(
            CheckResult(
                f"{name}.rows",
                array.shape[0] == n_shots,
                f"{array.shape[0]} rows vs {n_shots} shots",
            )
        )

    # --- environments ------------------------------------------------------
    env_ids = dataset.environment_ids
    if env_ids is not None:
        declared = {e.environment_id for e in meta.environments}
        present = {int(v) for v in np.unique(env_ids)}
        results.append(
            CheckResult(
                "environment_ids.subset_of_manifest",
                present <= declared,
                f"present {sorted(present)} vs declared {sorted(declared)}",
            )
        )
        counts = {
            int(k): int(v) for k, v in zip(*np.unique(env_ids, return_counts=True), strict=True)
        }
        expected = {e.environment_id: e.shots for e in meta.environments}
        matches = all(counts.get(k, 0) == v for k, v in expected.items())
        results.append(
            CheckResult(
                "environment_ids.counts",
                matches,
                f"counted {counts}, manifest declares {expected}",
            )
        )
        results.append(
            CheckResult(
                "environment_ids.dtype",
                env_ids.dtype == np.int32,
                f"expected int32, got {env_ids.dtype}",
            )
        )
    else:
        results.append(
            CheckResult(
                "environment_ids.absent_is_single_env",
                len(meta.environments) == 1,
                f"no environment_ids but manifest lists {len(meta.environments)} environments",
            )
        )
        if len(meta.environments) == 1:
            # The multi-env branch checks per-environment counts against the id
            # column; without this twin, a single-environment manifest whose declared
            # shots disagreed with its own arrays validated clean.
            declared_shots = meta.environments[0].shots
            results.append(
                CheckResult(
                    "environment.shots_match_arrays",
                    declared_shots == n_shots,
                    f"environment declares {declared_shots}, arrays hold {n_shots}",
                    "the single environment's declared shot count to match the file",
                )
            )

    # --- contract ----------------------------------------------------------
    if meta.contract is Contract.DEM_MECHANISM:
        results.append(
            CheckResult(
                "contract_b.mechanisms_present",
                dataset.mechanisms is not None,
                f"contract={meta.contract}, mechanisms array "
                f"{'present' if dataset.mechanisms is not None else 'absent'}",
                "a mechanisms array, since the contract is dem_mechanism",
            )
        )
        if dataset.mechanisms is not None:
            # Unconditional. Gating this on a populated n_mechanisms is what let a
            # (shots, 36) label array ship with n_mechanisms=None and validate clean.
            results.append(
                CheckResult(
                    "mechanisms.n_mechanisms_declared",
                    meta.n_mechanisms is not None,
                    f"n_mechanisms={meta.n_mechanisms}",
                    "a mechanism count, without which the label columns cannot be "
                    "interpreted and the width cannot be checked",
                )
            )
            if meta.n_mechanisms is not None:
                expected_mech_width = packed_width(meta.n_mechanisms)
                results.append(
                    CheckResult(
                        "mechanisms.packed_width",
                        dataset.mechanisms.shape[1] == expected_mech_width,
                        f"{dataset.mechanisms.shape[1]} bytes for {meta.n_mechanisms} mechanisms",
                        f"ceil({meta.n_mechanisms}/8) = {expected_mech_width} bytes",
                    )
                )
            results.append(
                CheckResult(
                    "mechanisms.source_environment_declared",
                    meta.mechanism_source_environment_id is not None,
                    f"mechanism_source_environment_id={meta.mechanism_source_environment_id}",
                    "the environment whose DEM defines each mechanism column",
                )
            )
    else:
        results.append(
            CheckResult(
                "contract_a.no_mechanisms",
                dataset.mechanisms is None,
                f"contract={meta.contract}, mechanisms array "
                f"{'absent' if dataset.mechanisms is None else 'present'}",
                "no mechanisms array, since the contract is logical_frame",
            )
        )

    # --- zero noise --------------------------------------------------------
    noiseless = [e for e in meta.environments if e.channels.is_noiseless]
    if noiseless and len(noiseless) == len(meta.environments):
        results.append(
            CheckResult(
                "zero_noise.detectors_all_zero",
                not dataset.detectors.any(),
                f"{int(np.unpackbits(dataset.detectors, bitorder='little').sum())} "
                f"detection events at p=0",
                "zero detection events, since every channel is zero",
            )
        )
        results.append(
            CheckResult(
                "zero_noise.observables_all_zero",
                not dataset.observables.any(),
                f"{int(np.unpackbits(dataset.observables, bitorder='little').sum())} "
                f"observable flips at p=0",
                "zero observable flips, since every channel is zero",
            )
        )

    # --- content hash ------------------------------------------------------
    if check_hash:
        # A missing hash is a failure, not a skip. Every writer in this package sets
        # one, so its absence means the file was produced by something else or was
        # edited; silently passing such a file defeats the point of having a hash.
        if meta.content_hash is None:
            results.append(
                CheckResult(
                    "content_hash",
                    False,
                    "manifest carries no content_hash",
                    "a content hash; pass check_hash=False to accept a file without one",
                )
            )
        else:
            actual = dataset.compute_content_hash()
            results.append(
                CheckResult(
                    "content_hash",
                    actual == meta.content_hash,
                    f"manifest {meta.content_hash[:16]}... vs actual {actual[:16]}...",
                    f"the recomputed {meta.content_hash_algorithm} digest to match",
                )
            )

    # --- structure agrees with the manifest --------------------------------
    if dataset.structure is not None:
        structure = dataset.structure
        agreements = [
            ("detectors", structure.n_detectors, meta.n_detectors),
            ("observables", structure.n_observables, meta.n_observables),
        ]
        if meta.n_mechanisms is not None:
            # Contract B: the labels index the DEM whose count is n_mechanisms. Every
            # writer guarantees the attached structure agrees (frozen_prior refuses a
            # mismatch outright), so a disagreement here means the structure describes
            # a different DEM than the one the label columns index — which the
            # detector/observable comparisons alone cannot see.
            agreements.append(("mechanisms", structure.n_mechanisms, meta.n_mechanisms))
        for label, got, want in agreements:
            results.append(
                CheckResult(
                    f"structure.{label}_match_manifest",
                    got == want,
                    f"structure declares {got}, manifest declares {want}",
                    "the attached DEM to describe this dataset, not a different one",
                )
            )
        results.extend(validate_dem_structure(structure).results)

    return ValidationReport(tuple(results))


def validate_dem_structure(
    structure: DemStructure, dem: stim.DetectorErrorModel | None = None
) -> ValidationReport:
    """Check a parsed DEM for the failure modes that silently corrupt H.

    The column-weight bound is asserted on **components**, not on mechanisms. A
    decomposed mechanism with two graphlike components legitimately touches up to four
    detectors; asserting the bound on mechanisms produces a validator that fails on
    correct data. Measured on d=3 r=3: mechanism weights run 1..4 while component
    weights are strictly 1 or 2.
    """
    results: list[CheckResult] = []

    if not structure.has_matrices:
        # A --structure coords file legitimately carries coordinates and nothing else.
        # Checking it against the full-DEM invariants would fail on correct data.
        results.append(
            CheckResult(
                "dem.coords_only",
                True,
                "coordinates only; matrix and prior checks do not apply",
            )
        )
        results.extend(_coordinate_checks(structure))
        return ValidationReport(tuple(results))

    results.append(
        CheckResult(
            "dem.H_shape",
            structure.h.shape == (structure.n_detectors, structure.n_mechanisms),
            f"expected {(structure.n_detectors, structure.n_mechanisms)}, got {structure.h.shape}",
        )
    )
    results.append(
        CheckResult(
            "dem.L_shape",
            structure.l.shape == (structure.n_observables, structure.n_mechanisms),
            f"expected {(structure.n_observables, structure.n_mechanisms)}, "
            f"got {structure.l.shape}",
        )
    )
    results.append(
        CheckResult(
            "dem.priors_length",
            structure.priors.shape == (structure.n_mechanisms,),
            f"expected ({structure.n_mechanisms},), got {structure.priors.shape}",
        )
    )
    if structure.priors.size:
        in_range = bool(((structure.priors >= 0.0) & (structure.priors <= 1.0)).all())
        results.append(
            CheckResult(
                "dem.priors_in_range",
                in_range,
                f"range [{structure.priors.min():.3g}, {structure.priors.max():.3g}]",
                "all priors within [0, 1]",
            )
        )

    if dem is not None:
        results.append(
            CheckResult(
                "dem.detectors_agree",
                structure.n_detectors == dem.num_detectors,
                f"structure {structure.n_detectors} vs dem {dem.num_detectors}",
            )
        )
        results.append(
            CheckResult(
                "dem.observables_agree",
                structure.n_observables == dem.num_observables,
                f"structure {structure.n_observables} vs dem {dem.num_observables}",
            )
        )
        n_error_instructions = sum(1 for i in dem.flattened() if i.type == "error")
        results.append(
            CheckResult(
                "dem.one_column_per_error_instruction",
                structure.n_mechanisms == n_error_instructions,
                f"n_mechanisms={structure.n_mechanisms}, error(...) instructions="
                f"{n_error_instructions}, components={len(structure.components)}",
                "n_mechanisms to equal the error(...) instruction count, NOT the component count",
            )
        )

    # Gated on mechanisms existing, NOT on components existing: an empty components
    # tuple beside n_mechanisms > 0 is precisely what every_mechanism_has_a_component
    # exists to catch, and gating the block on `if structure.components:` skipped that
    # very check in the one state it was written for.
    if structure.components or structure.n_mechanisms:
        component_hist = structure.component_weight_histogram()
        heavy = {w: c for w, c in component_hist.items() if w > 2}
        results.append(
            CheckResult(
                "dem.component_weight_le_2",
                not heavy,
                f"component weight histogram {component_hist}",
                f"every graphlike component to touch 1 or 2 detectors (found {heavy} above 2)",
            )
        )
        parents = {c.parent_mechanism_id for c in structure.components}
        results.append(
            CheckResult(
                "dem.component_parents_in_range",
                all(0 <= p < structure.n_mechanisms for p in parents),
                f"parent ids span {min(parents)}..{max(parents)}" if parents else "no components",
                f"every parent_mechanism_id within 0..{structure.n_mechanisms - 1}",
            )
        )
        results.append(
            CheckResult(
                "dem.every_mechanism_has_a_component",
                len(parents) == structure.n_mechanisms,
                f"{len(parents)} distinct parents for {structure.n_mechanisms} mechanisms",
                "one component group per mechanism",
            )
        )

    results.extend(_coordinate_checks(structure))

    # Reported, never asserted: mechanism weight legitimately exceeds 2.
    results.append(
        CheckResult(
            "dem.mechanism_weight_distribution",
            True,
            f"reported, not asserted: {structure.mechanism_weight_histogram()}",
        )
    )
    return ValidationReport(tuple(results))


def _coordinate_checks(structure: DemStructure) -> list[CheckResult]:
    """Checks that apply to detector coordinates at any structure level."""
    if not structure.detector_coords.size:
        return []
    return [
        CheckResult(
            "dem.coords_rows",
            structure.detector_coords.shape[0] == structure.n_detectors,
            f"{structure.detector_coords.shape[0]} coord rows for "
            f"{structure.n_detectors} detectors",
        )
    ]
