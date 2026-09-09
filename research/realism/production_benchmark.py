"""Fresh-process production-pipeline memory scaling; no hardware-realism claim.

Run from the repository root. Each process imports and executes the actual configured
generation path, including staged publication, hashing and HDF5 compression. This is a
single-machine measurement under possible concurrent CPU/GPU work, not a speed record.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _peak_working_set() -> int:
    """Windows reports the lifetime high-water mark, including import allocations."""
    if os.name != "nt":
        raise RuntimeError("This benchmark records Windows process memory counters")

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Counters),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.PeakWorkingSetSize)


def _sources(snapshot: Path | None = None) -> dict[str, str]:
    result = {}
    files = [*sorted(Path("qecgen").rglob("*.py")), Path(__file__)]
    for path in files:
        relative = path.resolve().relative_to(Path.cwd())
        content = path.read_bytes()
        result[relative.as_posix()] = hashlib.sha256(content).hexdigest()
        if snapshot is not None:
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
    return result


def _worker(config_path: Path) -> None:
    total_start = time.perf_counter()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.parent
    before = _sources(root / "source")
    from qecgen.run import ConfiguredSpec, run

    spec = ConfiguredSpec(raw)
    _write(root / "resolved_config.json", spec.config)
    start = time.perf_counter()
    result = run(spec)
    elapsed = time.perf_counter() - start
    peak = _peak_working_set()
    report = {
        "status": "complete",
        "config": spec.config,
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("stim", "numpy", "h5py", "scipy", "qecgen")
            },
        },
        "generation_seconds": elapsed,
        "total_worker_seconds_including_imports": time.perf_counter() - total_start,
        "shots_per_second": spec.shots / elapsed,
        "peak_working_set_bytes": peak,
        "memory_measurement": "Windows GetProcessMemoryInfo PeakWorkingSetSize; includes imports",
        "source_sha256": before,
        "source_changed_during_run": before != _sources(),
        "content_hash": result[0].content_hash,
        "output_bytes": spec.out.stat().st_size,
        "limitations": [
            "single run per condition, no confidence interval",
            "concurrent project development and GPU evaluation may contend for CPU and memory",
            "scenario parameters are not a calibrated hardware model",
            "process working set is not an allocation tracer or whole-machine memory use",
        ],
    }
    _write(root / "measurement.json", report)


def _config(mode: str, shots: int, path: Path, fmt: str = "hdf5") -> dict[str, Any]:
    config: dict[str, Any] = {
        "version": 1,
        "mode": "legacy" if mode == "legacy" else "device",
        "output": {"path": str(path.resolve()), "format": fmt, "structure": "none"},
        "sampling": {"shots": shots, "seed": 849, "chunk_size": 10_000},
        "circuit": {"distance": 3, "rounds": 10, "basis": "z", "rotated": True},
    }
    if mode == "legacy":
        config["legacy"] = {"noise_model": "stim_uniform_circuit_level", "p": 0.001}
        return config
    config["parameter_provenance"] = {
        "kind": "scenario",
        "description": "Illustrative throughput/memory scenario, not device calibration",
    }
    config["noise"] = {
        "version": 1,
        "probabilities": {
            "one_qubit_gate": 0.001,
            "two_qubit_gate": 0.002,
            "measurement": 0.01,
            "reset": 0.001,
            "idle": 0.001,
        },
        "qubit_overrides": {"1": {"measurement": 0.02}},
    }
    if mode == "dynamic":
        config["noise"].update(
            drift={
                "enabled": True,
                "qubits": [1],
                "pauli": "X",
                "baseline_probability": 0.002,
                "rho": 0.99,
                "sigma_logit": 0.03,
            },
            bursts={
                "enabled": True,
                "qubits": [1, 3],
                "pauli": "X",
                "onset_probability": 0.001,
                "recovery_probability": 0.1,
                "effect_probability": 0.05,
            },
            leakage={
                "enabled": True,
                "qubits": [1],
                "entry_probability": 0.001,
                "recovery_probability": 0.1,
                "reset_removal_probability": 1.0,
                "effect_probability": 0.1,
                "neighbor_edges": [[1, 3]],
                "neighbor_effect_probability": 0.02,
            },
        )
    return config


def _compare(hdf5: Path, npz: Path) -> dict[str, Any]:
    import numpy as np

    from qecgen.dataset import content_hash
    from qecgen.exporters import get_exporter

    left, right = get_exporter("hdf5").read(hdf5), get_exporter("npz").read(npz)
    arrays = ("detectors", "observables", "environment_ids", "mechanisms")
    checks = {}
    for name in arrays:
        a, b = getattr(left, name), getattr(right, name)
        checks[name] = (a is None and b is None) or (
            a is not None and b is not None and np.array_equal(a, b)
        )
    left_hash = content_hash(
        left.detectors, left.observables, left.environment_ids, left.mechanisms
    )
    right_hash = content_hash(
        right.detectors, right.observables, right.environment_ids, right.mechanisms
    )
    return {
        "equal": all(checks.values())
        and left_hash == right_hash == left.meta.content_hash == right.meta.content_hash,
        "arrays_equal": checks,
        "recomputed_content_hashes": [left_hash, right_hash],
        "manifest_content_hashes": [left.meta.content_hash, right.meta.content_hash],
        "shots": left.meta.shots,
    }


def run_benchmark(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    cases = [
        ("static", 100_000),
        ("static", 1_000_000),
        ("static", 10_000_000),
        ("dynamic", 100_000),
        ("dynamic", 1_000_000),
        ("legacy", 1_000_000),
    ]
    report: dict[str, Any] = {"status": "running", "measurements": [], "comparisons": {}}
    _write(output / "report.json", report)
    configs = []
    for mode, shots in cases:
        name = f"{mode}-{shots}-hdf5"
        configs.append((name, _config(mode, shots, output / name / "dataset.h5")))
    for mode in ("static", "dynamic"):
        name = f"{mode}-100000-npz"
        configs.append((name, _config(mode, 100_000, output / name / "dataset.npz", "npz")))
    try:
        for name, config in configs:
            directory = output / name
            directory.mkdir()
            path = directory / "config.json"
            _write(path, config)
            print(f"Starting {name}", flush=True)
            # Files avoid inherited pipe handles keeping Windows subprocess waits alive.
            with (directory / "worker.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "research.realism.production_benchmark",
                        "--worker",
                        str(path),
                    ],
                    stdout=log,
                    stderr=log,
                    check=True,
                    timeout=600,
                )
            measurement = json.loads((directory / "measurement.json").read_text(encoding="utf-8"))
            measurement["name"] = name
            report["measurements"].append(measurement)
            _write(output / "report.json", report)
            if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > 1_000_000_000:
                raise RuntimeError("Benchmark output exceeded its 1 GB disk budget")
        for mode in ("static", "dynamic"):
            report["comparisons"][mode] = _compare(
                output / f"{mode}-100000-hdf5" / "dataset.h5",
                output / f"{mode}-100000-npz" / "dataset.npz",
            )
        if not all(row["equal"] for row in report["comparisons"].values()):
            raise RuntimeError("HDF5 and NPZ content differed")
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        _write(output / "report.json", copy.deepcopy(report))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("data/realism/results/production-performance")
    )
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker)
    else:
        run_benchmark(args.output)


if __name__ == "__main__":
    main()
