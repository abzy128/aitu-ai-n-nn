from __future__ import annotations

import joblib

from final_anomaly.artifacts import load_artifact


def test_load_artifact_requires_dict(tmp_path) -> None:
    path = tmp_path / "artifact.joblib"
    joblib.dump({"model_name": "demo"}, path)

    assert load_artifact(path)["model_name"] == "demo"
