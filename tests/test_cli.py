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

    def test_contradictory_out_extension_is_refused(self, tmp_path: Path) -> None:
        result = _invoke(
            "generate", "--format", "jsonl", "--out", str(tmp_path / "x.h5"), "--shots", "1"
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
    for name in ("hdf5", "jsonl", "npz", "parquet"):
        assert name in result.output
