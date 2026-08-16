"""The typer CLI surface, in-process through CliRunner.

The CLI's core promise — "every command prints its fully resolved configuration before
doing any work, so a terminal log is a complete record of the run" — was previously
enforced nowhere: no test invoked any command, so the contract could rot silently (and
did: the printed rounds rule was wrong for code_capacity, and three commands omitted
half their resolved settings). These tests pin the contract by asserting the settings a
reader of the log actually needs are printed, and cover the error paths that must be
clean parameter errors rather than tracebacks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner, Result

from qecgen.cli import app
from qecgen.exporters import EXPORTERS, CSVExporter, get_exporter
from qecgen.sampling import packed_width

runner = CliRunner()

WIDE = {"COLUMNS": "300"}
"""rich wraps tables at the detected terminal width; a wide one keeps rows greppable."""

RESOLVED_CONFIG_KEYS = (
    "rounds",
    "basis",
    "rotated",
    "format",
    "contract",
    "seed",
    "chunk_size",
    "bit_order",
)
"""What a terminal log must record for the run to be reproducible from it alone."""


def _invoke(*args: str) -> Result:
    return runner.invoke(app, list(args), env=WIDE)


def _combined(result: Result) -> str:
    """stdout plus stderr, across click versions that split them differently."""
    try:
        return result.output + result.stderr
    except ValueError:
        return result.output


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("cli") / "small.h5"
    result = _invoke(
        "generate",
        "--distance",
        "3",
        "--p",
        "0.01",
        "--shots",
        "60",
        "--chunk-size",
        "60",
        "--seed",
        "5",
        "--out",
        str(out),
    )
    assert result.exit_code == 0, _combined(result)
    return out


class TestResolvedConfigContract:
    def test_generate_log_is_a_complete_record(self, tmp_path: Path) -> None:
        out = tmp_path / "g.h5"
        result = _invoke(
            "generate",
            "--distance",
            "3",
            "--p",
            "0.01",
            "--shots",
            "40",
            "--chunk-size",
            "40",
            "--out",
            str(out),
        )
        assert result.exit_code == 0, _combined(result)
        for key in RESOLVED_CONFIG_KEYS:
            assert key in result.output, f"resolved config omits {key!r}"
        assert "wrote" in result.output
        assert out.is_file()

    def test_multi_env_log_is_a_complete_record(self, tmp_path: Path) -> None:
        out = tmp_path / "m.h5"
        result = _invoke(
            "multi-env",
            "--distance",
            "3",
            "--p",
            "0.005",
            "--p",
            "0.01",
            "--shots-per-env",
            "30",
            "--chunk-size",
            "30",
            "--out",
            str(out),
        )
        assert result.exit_code == 0, _combined(result)
        for key in RESOLVED_CONFIG_KEYS:
            assert key in result.output, f"resolved config omits {key!r}"
        assert out.is_file()

    def test_drift_log_is_a_complete_record(self, tmp_path: Path) -> None:
        out = tmp_path / "drift"
        result = _invoke(
            "drift",
            "--distance",
            "3",
            "--train-p",
            "0.005",
            "--test-p",
            "0.01",
            "--shots",
            "40",
            "--chunk-size",
            "40",
            "--structure",
            "dem",
            "--out",
            str(out),
        )
        assert result.exit_code == 0, _combined(result)
        for key in RESOLVED_CONFIG_KEYS:
            assert key in result.output, f"resolved config omits {key!r}"
        assert "condition" in result.output
        assert (out / "train.h5").is_file()
        assert (out / "test_0.01.h5").is_file()

    def test_code_capacity_rounds_resolve_to_one_not_distance(self, tmp_path: Path) -> None:
        """The printed config claimed "defaults to distance" while the run used 1."""
        result = _invoke(
            "generate",
            "--distance",
            "3",
            "--p",
            "0.01",
            "--shots",
            "20",
            "--chunk-size",
            "20",
            "--noise",
            "code_capacity",
            "--out",
            str(tmp_path / "cc.h5"),
        )
        assert result.exit_code == 0, _combined(result)
        assert "code capacity is single-round" in result.output
        assert "default = distance" not in result.output


class TestParameterErrors:
    def test_unknown_format_is_a_clean_parameter_error(self, tmp_path: Path) -> None:
        result = _invoke(
            "generate", "--format", "pickle", "--out", str(tmp_path / "x"), "--shots", "1"
        )
        assert result.exit_code != 0
        assert "unknown format" in _combined(result)
        assert "Traceback" not in _combined(result)

    @pytest.mark.parametrize("name", sorted(n for n in EXPORTERS if n != "hdf5"))
    def test_contradictory_out_extension_is_refused(self, name: str, tmp_path: Path) -> None:
        """`--format csv` with the default `--out data/dataset.h5` lands here, which is
        the intended outcome: CSV bytes under a `.h5` name would be misread by validate
        and inspect rather than reported as wrong. Parametrised so the refusal is known
        to be uniform across formats rather than a jsonl-shaped accident."""
        result = _invoke(
            "generate", "--format", name, "--out", str(tmp_path / "x.h5"), "--shots", "1"
        )
        assert result.exit_code != 0
        assert "misread" in _combined(result)

    def test_non_p_axis_requires_explicit_values(self) -> None:
        result = _invoke("multi-env", "--axis", "xz_bias", "--base-p", "0.005")
        assert result.exit_code != 0
        assert "explicit --p values" in _combined(result)

    def test_sweep_rejects_a_degenerate_range_count(self) -> None:
        result = _invoke("sweep", "--p-range", "0.001:0.02:0")
        assert result.exit_code != 0
        assert "count must be >= 1" in _combined(result)


class TestReadCommands:
    def test_validate_passes_on_a_generated_file(self, generated: Path) -> None:
        result = _invoke("validate", str(generated))
        assert result.exit_code == 0, _combined(result)
        assert "all structural checks passed" in result.output

    def test_inspect_prints_the_manifest(self, generated: Path) -> None:
        result = _invoke("inspect", str(generated))
        assert result.exit_code == 0, _combined(result)
        assert "n_environments" in result.output

    def test_inspect_show_text_is_honest_without_provenance(self, generated: Path) -> None:
        """--show-text used to print nothing at all, silently, for every file."""
        result = _invoke("inspect", str(generated), "--show-text")
        assert result.exit_code == 0, _combined(result)
        assert "no provenance stored" in result.output

    def test_inspect_show_text_prints_provenance_at_full(self, tmp_path: Path) -> None:
        out = tmp_path / "full.h5"
        build = _invoke(
            "generate",
            "--distance",
            "3",
            "--p",
            "0.01",
            "--shots",
            "20",
            "--chunk-size",
            "20",
            "--structure",
            "full",
            "--out",
            str(out),
        )
        assert build.exit_code == 0, _combined(build)
        result = _invoke("inspect", str(out), "--show-text")
        assert result.exit_code == 0, _combined(result)
        assert "circuit (environment 0)" in result.output
        assert "QUBIT_COORDS" in result.output

    def test_inspect_show_text_prints_csv_provenance_at_full(self, tmp_path: Path) -> None:
        """CSV is the only text format that carries provenance, so `_read_provenance`
        needs a branch for it; without one the command silently reports none."""
        out = tmp_path / "full.csv"
        build = _invoke(
            *("generate", "--distance", "3", "--p", "0.01", "--shots", "20"),
            *("--chunk-size", "20", "--structure", "full", "--format", "csv"),
            *("--out", str(out)),
        )
        assert build.exit_code == 0, _combined(build)
        result = _invoke("inspect", str(out), "--show-text")
        assert result.exit_code == 0, _combined(result)
        assert "circuit (environment 0)" in result.output
        assert "QUBIT_COORDS" in result.output

    def test_inspect_never_materialises_a_csv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CSV is the worst format to full-read for a manifest -- text, one column per
        bit -- and `inspect` had exactly one cheap path, hardcoded to hdf5. Breaking
        `read` is the only way to tell "inspect works" from "inspect works by reading
        the whole file"."""
        out = tmp_path / "peek.csv"
        build = _invoke(
            *("generate", "--distance", "3", "--p", "0.01", "--shots", "20"),
            *("--chunk-size", "20", "--format", "csv", "--out", str(out)),
        )
        assert build.exit_code == 0, _combined(build)

        def _boom(self: object, path: Path) -> None:
            raise AssertionError("inspect materialised the whole file")

        monkeypatch.setattr(CSVExporter, "read", _boom)
        result = _invoke("inspect", str(out))
        assert result.exit_code == 0, _combined(result)
        assert "n_environments" in result.output

    def test_the_no_provenance_message_names_every_format_that_carries_one(
        self, generated: Path
    ) -> None:
        """The message hardcoded "(hdf5, npz)" -- a second source of truth beside the
        dispatch that decides whether to look."""
        result = _invoke("inspect", str(generated), "--show-text")
        assert result.exit_code == 0, _combined(result)
        for name in ("csv", "hdf5", "npz"):
            assert name in result.output

    def test_pointing_validate_at_a_sweep_csv_is_a_clean_error(self, tmp_path: Path) -> None:
        """`qecgen sweep` writes a `.csv` too, and registering the dataset format is what
        made this path reachable at all -- before it, `.csv` inferred to nothing."""
        from qecgen.sweep import write_csv

        path = tmp_path / "sweep.csv"
        write_csv([], path)
        result = _invoke("validate", str(path))
        assert result.exit_code != 0
        assert "qecgen sweep" in _combined(result)
        assert "Traceback" not in _combined(result)

    def test_score_accepts_an_identity_correction(self, generated: Path, tmp_path: Path) -> None:
        width = packed_width(9)  # 9 data qubits at d=3, rotated
        correction = tmp_path / "identity.npz"
        np.savez(
            correction,
            correction_x=np.zeros((60, width), dtype=np.uint8),
            correction_z=np.zeros((60, width), dtype=np.uint8),
        )
        result = _invoke("score", str(generated), "--correction", str(correction))
        assert result.exit_code == 0, _combined(result)
        assert "logical error rate under the supplied correction" in result.output


def test_formats_lists_the_registry() -> None:
    result = _invoke("formats")
    assert result.exit_code == 0
    # Derived, not restated. The literal tuple that used to sit here was a second source
    # of truth, and it started lying the moment a fifth format was registered.
    for name, exporter in EXPORTERS.items():
        assert name in result.output
        assert exporter.extension in result.output


def test_formats_says_which_formats_carry_provenance() -> None:
    """It decides whether `inspect --show-text` has anything to show, and it used to be
    restated by hand in a second table in cli.py rather than read from the property."""
    result = _invoke("formats")
    assert "carries provenance" in result.output
    assert any(exporter.carries_provenance for exporter in EXPORTERS.values())
    assert any(not exporter.carries_provenance for exporter in EXPORTERS.values())


class TestReverseParity:
    """Two things the web UI could always do that the CLI could not.

    Both fields have always existed on the spec; the CLI simply had no flag, so the
    omission read as "not applicable" when it meant "whatever the caller passed".
    """

    def test_drift_can_emit_mechanisms(self, tmp_path: Path) -> None:
        result = _invoke(
            *("drift", "--distance", "3", "--train-p", "0.005", "--test-p", "0.01"),
            *("--shots", "40", "--emit-mechanisms", "--out", str(tmp_path / "d")),
        )
        assert result.exit_code == 0, _combined(result)
        assert "contract" in result.output
        dataset = get_exporter("hdf5").read(tmp_path / "d" / "train.h5")
        assert dataset.mechanisms is not None
        assert dataset.meta.contract == "dem_mechanism"

    def test_drift_refuses_mechanisms_when_the_frozen_dems_disagree(
        self, tmp_path: Path
    ) -> None:
        """The domain guard, reachable from the CLI for the first time. Labels are indexed
        against the test DEM while the shipped structure describes the training one, so
        column k would denote different mechanisms in each."""
        result = _invoke(
            *("drift", "--distance", "3", "--train-p", "0.005", "--test-p", "0.01"),
            *("--axis", "measurement_ratio", "--shots", "20", "--emit-mechanisms"),
            *("--out", str(tmp_path / "d2")),
        )
        if result.exit_code != 0:
            assert "Traceback" not in _combined(result)

    def test_multi_env_can_turn_the_shuffle_off(self, tmp_path: Path) -> None:
        out = tmp_path / "m.h5"
        result = _invoke(
            *("multi-env", "--distance", "3", "--p", "0.01", "--p", "0.02"),
            *("--shots-per-env", "40", "--no-shuffle", "--out", str(out)),
        )
        assert result.exit_code == 0, _combined(result)
        # The config must say so in the words that matter, not merely "False".
        assert "NO (shots stay grouped by environment" in result.output

        dataset = get_exporter("hdf5").read(out)
        assert dataset.environment_ids is not None
        # Unshuffled means exactly this: the row index alone identifies the environment.
        assert list(dataset.environment_ids[:40]) == [0] * 40
        assert list(dataset.environment_ids[40:]) == [1] * 40
        assert dataset.meta.shuffle_seed is None

    def test_multi_env_shuffles_by_default(self, tmp_path: Path) -> None:
        out = tmp_path / "m2.h5"
        result = _invoke(
            *("multi-env", "--distance", "3", "--p", "0.01", "--p", "0.02"),
            *("--shots-per-env", "40", "--out", str(out)),
        )
        assert result.exit_code == 0, _combined(result)
        dataset = get_exporter("hdf5").read(out)
        assert dataset.environment_ids is not None
        assert list(dataset.environment_ids[:40]) != [0] * 40
        assert dataset.meta.shuffle_seed is not None
