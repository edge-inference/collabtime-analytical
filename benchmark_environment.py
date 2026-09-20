"""Capture and optionally constrain the local benchmark environment."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def cpu_model_name() -> str:
    text = _read_text(Path("/proc/cpuinfo")) or ""
    for line in text.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def cpu_governors() -> list[str]:
    governors = {
        value
        for cpu_dir in Path("/sys/devices/system/cpu").glob("cpu[0-9]*")
        if (value := _read_text(cpu_dir / "cpufreq/scaling_governor")) is not None
    }
    return sorted(governors)


def set_cpu_affinity(cpu: int | None) -> None:
    if cpu is None:
        return
    if cpu < 0:
        raise ValueError("cpu affinity index must be nonnegative")
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("CPU affinity is not supported on this platform")
    available = os.sched_getaffinity(0)
    if cpu not in available:
        raise ValueError(f"CPU {cpu} is not in the available affinity set")
    os.sched_setaffinity(0, {cpu})


def capture_environment() -> dict[str, Any]:
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    )
    load_average = os.getloadavg() if hasattr(os, "getloadavg") else ()
    return {
        "platform": platform.platform(),
        "cpu_model": cpu_model_name(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "cpu_governors": cpu_governors(),
        "load_average_1_5_15": list(load_average),
        "python": sys.version,
    }
