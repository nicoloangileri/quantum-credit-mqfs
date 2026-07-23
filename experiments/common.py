"""Shared experiment utilities: timing, JSON I/O, environment capture."""

from __future__ import annotations

import json
import platform
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from quantum_credit import config


class Timer:
    """Wall-clock timer: `with Timer() as t: ...; t.seconds`."""

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self._t0


@contextmanager
def timed(store: dict[str, float], key: str):
    t0 = time.perf_counter()
    yield
    store[key] = time.perf_counter() - t0


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def save_results(name: str, payload: dict[str, Any]) -> Path:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / name
    with open(path, "w") as f:
        json.dump(to_jsonable(payload), f, indent=2)
    print(f"[saved] {path}")
    return path


def load_results(name: str) -> dict[str, Any]:
    with open(config.RESULTS_DIR / name) as f:
        return json.load(f)


def environment_info() -> dict[str, str]:
    import qiskit
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "qiskit": qiskit.__version__,
    }
