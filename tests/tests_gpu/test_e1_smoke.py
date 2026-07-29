from __future__ import annotations

import json
from pathlib import Path
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.artifacts.run import load_feature_artifact
from src.experiment.e1 import run_e1
from src.settings import PROJECT_ROOT_PATH


@pytest.mark.gpu
def test_exact_nvidia_e1_smoke(tmp_path):
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("No CUDA device is available.")
    config_dir = str(Path(PROJECT_ROOT_PATH) / "src" / "experiment" / "conf")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="e1",
            overrides=[
                "e1/mode=smoke",
                "stage=all",
                f"artifacts.root={tmp_path.as_posix()}",
            ],
        )
    resolved = OmegaConf.to_container(cfg, resolve=True)
    result = run_e1(resolved)
    assert result.path is not None
    assert (result.path / "manifest.json").exists()
    assert (result.path / "regularization_path.csv").exists()
    assert (result.path / "learning_curve.csv").exists()
    payload = load_feature_artifact(result.path)
    assert payload["manifest"]["artifact_kind"] == "e1-exact-nvidia-run"
    assert payload["status"] == {"status": "complete"}
    assert payload["results"]["values"].shape == (96, 2, 9)
    execution = payload["execution"]
    assert execution["estimator_kind"] == "exact"
    assert execution["execution_metadata"]["backend_kind"] == "nvidia_gpu"
    assert execution["execution_metadata"]["exact"] is True
    assert execution["feature_ordering"] == "reservoir-major-observable-minor/v1"
    identity = json.loads((result.path / "run_identity.json").read_text())
    assert identity["configuration_hash"] == result.configuration_hash
    assert identity["stage"] == "complete"

    resumed = run_e1(resolved)
    assert resumed.path == result.path
    assert load_feature_artifact(resumed.path)["results"]["values"].shape == (96, 2, 9)
