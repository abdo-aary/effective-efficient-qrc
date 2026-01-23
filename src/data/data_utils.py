from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_data_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML dataset config into a plain dict."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML config must be a mapping at the top level.")
    return cfg
