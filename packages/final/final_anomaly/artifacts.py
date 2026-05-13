from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Load a joblib model artifact exported by the training script."""
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise TypeError(f"Expected artifact dict, got {type(artifact).__name__}")
    return artifact
