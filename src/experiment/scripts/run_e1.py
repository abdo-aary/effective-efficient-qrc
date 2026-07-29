"""Hydra entrypoint for staged Experiment E1 execution."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from src.experiment.e1 import run_e1

log = logging.getLogger(__name__)


def run(cfg: DictConfig):
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Resolved E1 configuration must be a mapping.")
    return run_e1(resolved)


@hydra.main(version_base=None, config_path="../conf", config_name="e1")
def main(cfg: DictConfig) -> None:
    result = run(cfg)
    log.info("E1 stage=%s path=%s hash=%s", result.stage, result.path, result.configuration_hash)


if __name__ == "__main__":
    main()
