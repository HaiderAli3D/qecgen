"""Dataset note and root manifest rendering from a machine-readable summary.

The note and the manifest are *views* of ``<name>_summary.json``, never a second source
of numbers. Every count and rate is rendered from the summary dict with ``repr`` so the
text carries the float64 exactly, and the note embeds ``summary_sha256``, the digest of
the summary file it was rendered from. Validation recomputes the numbers from the CSV,
parses them back out of the note through :func:`parse_note_fields`, and compares; the
digest is what ties a note to the specific summary it claims to describe, so a note
regenerated from stale numbers is caught rather than read as documentation.

Field labels live in :data:`NOTE_FIELDS` (the brief's fields) and :data:`_AUX_NOTE_FIELDS`
(context lines), and :data:`NOTE_LABELS` — their union plus the digest label — is the
only set the parser reads, so the parser and the renderer cannot drift apart: a label
renamed in one place is a missing key in the other, and a prose bullet that happens to
contain ``: `` is never mistaken for a field. A summary missing a required field is an
error, not an ``n/a``; the brief lists what a note must state and a silently absent
line would be a note that fails to state it.
"""

from __future__ import annotations

import datetime as dt
import json
import numbers
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "MANIFEST_COLUMNS",
    "NOTE_FIELDS",
    "NOTE_LABELS",
    "SUMMARY_SHA_LABEL",
    "parse_note_fields",
    "render_manifest",
    "render_note",
]

NOTE_FIELDS: dict[str, str] = {
    "dataset_name": "Dataset name",
    "source": "Source",
    "source_paths": "Source paths",
    "source_hashes": "Source hashes",
    "provenance_status": "Provenance status",
    "distance": "Distance",
    "rounds": "Rounds",
    "basis": "Basis",
    "orientation": "Orientation",
    "n_detectors": "Detectors",
    "n_observables": "Observables",
    "noise_model": "Noise model",
    "noise_parameters": "Noise parameters",
    "n_runs": "Unique runs",
    "seed": "Seed",
    "chunk_size": "Chunk size",
    "versions": "Software versions",
    "decoder_method": "Decoder construction method",
    "matching_mode": "Matching mode",
    "decoder_path": "Decoder model path",
    "decoder_sha256": "Decoder model sha256",
    "split_method": "Data partition method",
    "split_counts": "Split sizes",
    "pm_failures": "PyMatching failures",
    "pm_error_rate": "PyMatching logical error rate",
    "pm_ci": "PyMatching 95% confidence interval",
    "always_zero_accuracy": "Always-zero residual accuracy",
}
"""Summary key -> note label, for every field the brief lists under "Dataset note" (the
limitations/deviations lists are rendered as sections, see :func:`render_note`)."""

_AUX_NOTE_FIELDS: dict[str, str] = {
    "config_hash": "Configuration hash",
    "decoder_source": "Decoder model source",
    "split_seed": "Split seed",
    "split_fractions": "Split fractions",
    "sanity": "Sanity model",
    "rendered_at": "Rendered at",
}
"""Summary key -> note label for the labelled lines the renderer emits beyond the brief's
fields. Enumerated, not improvised at the call site, so :data:`NOTE_LABELS` is exactly
the set of labels a note can carry."""

SUMMARY_SHA_LABEL = "summary_sha256"

NOTE_LABELS: frozenset[str] = frozenset(
    (*NOTE_FIELDS.values(), *_AUX_NOTE_FIELDS.values(), SUMMARY_SHA_LABEL)
)
"""Every label :func:`parse_note_fields` reads back; any other ``- text: more`` line is
prose (a limitation such as "si1000 prior: shipped verbatim") and is left alone."""

_FREE_TEXT_SECTIONS: frozenset[str] = frozenset({"Known limitations", "Deviations"})
"""Sections whose bullets are sentences, never fields; the parser skips them entirely so a
sentence that starts with a known label cannot collide with the real field either."""

_UNPUBLISHED_STATUSES: frozenset[str] = frozenset({"blocked", "failed"})
"""What a manifest row may say about a dataset that was attempted and not published."""

_REQUIRED_SUMMARY_KEYS: tuple[str, ...] = (
    "dataset_name",
    "additional",
    "source",
    "source_paths",
    "source_hashes",
    "provenance_status",
    "provenance_limitations",
    "distance",
    "rounds",
    "basis",
    "orientation",
    "n_detectors",
    "n_observables",
    "noise_model",
    "noise_parameters",
    "n_runs",
    "seed",
    "chunk_size",
    "versions",
    "decoder_method",
    "decoder_source",
    "matching_mode",
    "decoder_path",
    "decoder_sha256",
    "config_hash",
    "schema_version",
    "split_method",
    "split_seed",
    "split_fractions",
    "split_counts",
    "pm_failures",
    "pm_error_rate",
    "pm_ci_low",
    "pm_ci_high",
    "always_zero_accuracy",
    "sanity",
    "limitations",
    "deviations",
)

_VERSION_ORDER: tuple[str, ...] = ("stim", "sinter", "pymatching", "numpy", "scipy", "qecgen")

MANIFEST_COLUMNS: tuple[str, ...] = (
    "name",
    "source",
    "distance",
    "rounds",
    "basis",
    "runs",
    "detectors",
    "decoder model source",
    "PyMatching failures",
    "PyMatching error rate",
    "95% CI",
    "always-zero accuracy",
    "sanity-model metrics",
    "status",
)

_SANITY_METRIC_KEYS: tuple[str, ...] = (
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "pm_test_error_rate",
    "corrected_test_error_rate",
    "n_flips",
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _require(summary: Mapping[str, Any], keys: Sequence[str]) -> None:
    missing = [key for key in keys if key not in summary]
    if missing:
        raise ValueError(f"summary is missing required field(s): {missing}")


def _scalar(value: Any) -> str:
    """Render one value: integers as digits, reals by ``repr(float(...))`` (exact),
    ``None`` as an explicit statement, and structured values as canonical JSON so they
    parse back unambiguously.

    Dispatch is on the ``numbers`` ABCs, not on ``int``/``float``, because the summary is
    assembled from NumPy arithmetic: under NumPy 2 ``np.float64`` *is* a ``float`` yet
    ``repr`` of it is ``"np.float64(0.5)"``, and ``np.int64`` is not an ``int`` at all, so
    it fell through to ``json.dumps`` and raised. Booleans are tested first because
    ``bool`` is an ``Integral`` and would otherwise print as ``1``; ``np.bool_`` belongs to
    no ``numbers`` ABC and is named explicitly for the same reason.
    """
    if value is None:
        return "not applicable"
    if isinstance(value, bool | np.bool_):
        return "true" if value else "false"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        return repr(float(value))
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _line(label: str, value: str) -> str:
    return f"- {label}: {value}"


def _bullets(items: Sequence[Any]) -> list[str]:
    """An empty list is stated as such: a heading with nothing under it reads as an
    omission, not as "nothing to report"."""
    if not items:
        return ["- none recorded"]
    return [f"- {item}" for item in items]


def _ci_text(summary: Mapping[str, Any]) -> str:
    low = float(summary["pm_ci_low"])
    high = float(summary["pm_ci_high"])
    return f"[{low!r}, {high!r}]"


def _versions_text(versions: Mapping[str, Any]) -> str:
    missing = [name for name in _VERSION_ORDER if name not in versions]
    if missing:
        raise ValueError(f"summary versions lack {missing}; the note must state each of them")
    ordered = [f"{name}={versions[name]}" for name in _VERSION_ORDER]
    extra = [f"{name}={versions[name]}" for name in sorted(versions) if name not in _VERSION_ORDER]
    return ", ".join(ordered + extra)


def _split_counts_text(counts: Mapping[str, Any]) -> str:
    return ", ".join(f"{name}={int(counts[name])}" for name in counts)


def _sanity_lines(sanity: Any) -> list[str]:
    """Render the sanity block honestly: skipped is stated with its reason, and a model
    that flipped nothing is shown as such rather than omitted."""
    label = _AUX_NOTE_FIELDS["sanity"]
    if sanity is None:
        return [_line(label, "not run")]
    if not isinstance(sanity, Mapping):
        raise ValueError("summary sanity must be a mapping or null")
    if "skipped_reason" in sanity:
        return [_line(label, f"skipped ({sanity['skipped_reason']})")]
    models = sanity.get("models")
    if not isinstance(models, Mapping) or not models:
        return [_line(label, "no model results recorded")]
    lines = [f"- {label}s (fit on train, threshold on validation, evaluated once on test):"]
    for model_name in sorted(models):
        metrics = models[model_name]
        parts = [f"{key}={_scalar(metrics.get(key))}" for key in _SANITY_METRIC_KEYS]
        lines.append(f"  - {model_name}: " + ", ".join(parts))
    return lines


def _sanity_cell(sanity: Any) -> str:
    if sanity is None:
        return "not run"
    if not isinstance(sanity, Mapping):
        return "invalid"
    if "skipped_reason" in sanity:
        return f"skipped ({sanity['skipped_reason']})"
    models = sanity.get("models")
    if not isinstance(models, Mapping) or not models:
        return "no results"
    cells = []
    for model_name in sorted(models):
        metrics = models[model_name]
        cells.append(
            f"{model_name}: balanced_accuracy={_scalar(metrics.get('balanced_accuracy'))}, "
            f"roc_auc={_scalar(metrics.get('roc_auc'))}, "
            f"corrected_test_error_rate={_scalar(metrics.get('corrected_test_error_rate'))}, "
            f"n_flips={_scalar(metrics.get('n_flips'))}"
        )
    return "; ".join(cells)


def render_note(summary: Mapping[str, Any], *, summary_sha256: str) -> str:
    """Render ``<name>_note.md`` from the summary and the digest of its JSON file.

    ``summary_sha256`` is keyword-only and validated as a 64-hex digest (``fullmatch``:
    ``$`` alone lets a trailing newline through): a note carrying a placeholder here
    would defeat the one check that ties it to its numbers.
    """
    _require(summary, _REQUIRED_SUMMARY_KEYS)
    if not isinstance(summary_sha256, str) or not _HEX64.fullmatch(summary_sha256):
        raise ValueError(f"summary_sha256 must be a 64-hex digest, got {summary_sha256!r}")
    name = str(summary["dataset_name"])
    additional = bool(summary["additional"])

    lines = [f"# Residual dataset note: {name}", ""]
    if additional:
        lines.append(
            "**Additional dataset** beyond the required set (see Deviations); not an "
            "independent baseline."
        )
        lines.append("")
    lines.append(
        "PyMatching residual-error dataset (schema version "
        f"{_scalar(summary['schema_version'])}). `pm_weight` is a sum of matching-edge "
        "weights, not a fault count. These artifacts are not qecgen datasets and claim no "
        "Nexus compatibility."
    )
    lines.append("")
    lines.append("## Identity")
    lines.append(_line(NOTE_FIELDS["dataset_name"], name))
    lines.append(_line(NOTE_FIELDS["source"], _scalar(summary["source"])))
    lines.append(_line(NOTE_FIELDS["source_paths"], _scalar(summary["source_paths"])))
    lines.append(_line(NOTE_FIELDS["source_hashes"], _scalar(summary["source_hashes"])))
    lines.append(_line(NOTE_FIELDS["provenance_status"], _scalar(summary["provenance_status"])))
    lines.append(_line(_AUX_NOTE_FIELDS["config_hash"], _scalar(summary["config_hash"])))
    lines.append("")
    lines.append("## Code and noise")
    for key in ("distance", "rounds", "basis", "orientation", "n_detectors", "n_observables"):
        lines.append(_line(NOTE_FIELDS[key], _scalar(summary[key])))
    lines.append(_line(NOTE_FIELDS["noise_model"], _scalar(summary["noise_model"])))
    lines.append(_line(NOTE_FIELDS["noise_parameters"], _scalar(summary["noise_parameters"])))
    lines.append("")
    lines.append("## Rows and generation")
    lines.append(_line(NOTE_FIELDS["n_runs"], str(int(summary["n_runs"]))))
    lines.append(_line(NOTE_FIELDS["seed"], _scalar(summary["seed"])))
    lines.append(_line(NOTE_FIELDS["chunk_size"], _scalar(summary["chunk_size"])))
    lines.append(_line(NOTE_FIELDS["versions"], _versions_text(summary["versions"])))
    lines.append("")
    lines.append("## Decoder")
    lines.append(_line(NOTE_FIELDS["decoder_method"], _scalar(summary["decoder_method"])))
    lines.append(_line(_AUX_NOTE_FIELDS["decoder_source"], _scalar(summary["decoder_source"])))
    lines.append(_line(NOTE_FIELDS["matching_mode"], _scalar(summary["matching_mode"])))
    lines.append(_line(NOTE_FIELDS["decoder_path"], _scalar(summary["decoder_path"])))
    lines.append(_line(NOTE_FIELDS["decoder_sha256"], _scalar(summary["decoder_sha256"])))
    lines.append("")
    lines.append("## Data partition")
    lines.append(_line(NOTE_FIELDS["split_method"], _scalar(summary["split_method"])))
    lines.append(_line(_AUX_NOTE_FIELDS["split_seed"], _scalar(summary["split_seed"])))
    lines.append(_line(_AUX_NOTE_FIELDS["split_fractions"], _scalar(summary["split_fractions"])))
    lines.append(_line(NOTE_FIELDS["split_counts"], _split_counts_text(summary["split_counts"])))
    lines.append("")
    lines.append("## PyMatching baseline")
    lines.append(_line(NOTE_FIELDS["pm_failures"], str(int(summary["pm_failures"]))))
    lines.append(_line(NOTE_FIELDS["pm_error_rate"], repr(float(summary["pm_error_rate"]))))
    lines.append(_line(NOTE_FIELDS["pm_ci"], _ci_text(summary)))
    lines.append(
        _line(NOTE_FIELDS["always_zero_accuracy"], repr(float(summary["always_zero_accuracy"])))
    )
    lines.append("")
    lines.append("## Sanity residual model")
    lines.extend(_sanity_lines(summary["sanity"]))
    lines.append("")
    lines.append("## Known limitations")
    limitations = list(summary["provenance_limitations"]) + list(summary["limitations"])
    lines.extend(_bullets(limitations))
    lines.append("")
    lines.append("## Deviations")
    deviations = list(summary["deviations"])
    if additional:
        deviations.append("additional dataset beyond the four required configurations")
    lines.extend(_bullets(deviations))
    lines.append("")
    lines.append("## Integrity")
    lines.append(_line(SUMMARY_SHA_LABEL, summary_sha256))
    lines.append(
        _line(
            _AUX_NOTE_FIELDS["rendered_at"], dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        )
    )
    lines.append("")
    return "\n".join(lines)


def parse_note_fields(text: str) -> dict[str, str]:
    """Read the labelled ``- label: value`` lines back into a mapping.

    Only a label in :data:`NOTE_LABELS` is a field, and the "Known limitations" and
    "Deviations" sections are skipped outright. Their bullets are sentences that routinely
    contain ``: `` ("si1000 prior: not reproducible from the circuit"), and two of them
    sharing a prefix used to read as a label appearing twice — which made the note
    unparsable, and therefore unvalidatable, for exactly the datasets with the most to
    disclose. Only the first colon splits, so a value that itself contains colons (a
    Windows path, a JSON object) survives; a known label that appears twice is still an
    error because validation would otherwise silently compare against whichever came last.
    """
    fields: dict[str, str] = {}
    skipping = False
    for raw in text.splitlines():
        if raw.startswith("## "):
            skipping = raw[3:].strip() in _FREE_TEXT_SECTIONS
            continue
        if skipping or not raw.startswith("- ") or ": " not in raw:
            continue
        label, value = raw[2:].split(": ", 1)
        if label not in NOTE_LABELS:
            continue
        if label in fields:
            raise ValueError(f"note label {label!r} appears more than once")
        fields[label] = value
    return fields


def _cell(value: str) -> str:
    """Pipes and newlines inside a table cell break the table itself."""
    return value.replace("|", "\\|").replace("\n", " ")


def _summary_row(summary: Mapping[str, Any]) -> list[str]:
    _require(summary, _REQUIRED_SUMMARY_KEYS)
    additional = bool(summary["additional"])
    name = str(summary["dataset_name"])
    status = "completed (additional)" if additional else "completed"
    return [
        _cell(f"{name} (additional)" if additional else name),
        _cell(_scalar(summary["source"])),
        _scalar(summary["distance"]),
        _scalar(summary["rounds"]),
        _scalar(summary["basis"]),
        str(int(summary["n_runs"])),
        str(int(summary["n_detectors"])),
        _cell(_scalar(summary["decoder_source"])),
        str(int(summary["pm_failures"])),
        repr(float(summary["pm_error_rate"])),
        _ci_text(summary),
        repr(float(summary["always_zero_accuracy"])),
        _cell(_sanity_cell(summary["sanity"])),
        status,
    ]


def _entry_status(entry: Mapping[str, Any]) -> str:
    """The status word of an unpublished dataset's record, ``blocked`` when unstated.

    ``blocked`` means the source could not be resolved (fix the input); ``failed`` means
    the pipeline's own refusal fired after resolution (fix the run). ``build_all`` records
    which, and the manifest must say the same word: rendering a failure as "blocked" would
    file a broken build under "missing file" and nobody would look for the bug.
    """
    status = str(entry.get("status", "blocked"))
    if status not in _UNPUBLISHED_STATUSES:
        raise ValueError(
            f"unpublished dataset status must be one of {sorted(_UNPUBLISHED_STATUSES)}, "
            f"got {status!r}"
        )
    return status


def _blocked_row(entry: Mapping[str, Any]) -> list[str]:
    _require(entry, ("dataset_name", "reason"))
    name = str(entry["dataset_name"])
    additional = bool(entry.get("additional", False))
    source = _scalar(entry.get("source", "unavailable"))
    reason = _cell(str(entry["reason"]))
    status = f"{_entry_status(entry)}: {reason}"
    if additional:
        status += " (additional)"
    return [
        _cell(f"{name} (additional)" if additional else name),
        _cell(source),
        *["—"] * 11,
        status,
    ]


def render_manifest(
    summaries: Sequence[Mapping[str, Any]], blocked: Sequence[Mapping[str, Any]]
) -> str:
    """Render ``MANIFEST.md``: one row per dataset, completed first, unpublished after.

    A blocked or failed dataset gets a row with its status word and reason rather than no
    row, so the manifest states what was attempted and why it is absent (see
    :func:`_entry_status` for why the two words are kept apart); an additional dataset is
    flagged in both its name and status so it is never read as one of the required four.
    """
    rows = [_summary_row(s) for s in summaries] + [_blocked_row(b) for b in blocked]
    n_failed = sum(1 for entry in blocked if _entry_status(entry) == "failed")
    lines = [
        "# Residual-error datasets",
        "",
        f"Rendered at {dt.datetime.now(dt.UTC).isoformat(timespec='seconds')}. "
        f"{len(summaries)} completed, {len(blocked) - n_failed} blocked, {n_failed} failed. "
        "Rates are PyMatching's standard-matching logical error rates with exact 95% "
        "Clopper-Pearson intervals; sanity metrics are held-out test results of the "
        "baseline residual classifiers.",
        "",
        "| " + " | ".join(MANIFEST_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in MANIFEST_COLUMNS) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    if not rows:
        lines.append("")
        lines.append("No datasets have been built or attempted.")
    lines.append("")
    return "\n".join(lines)
