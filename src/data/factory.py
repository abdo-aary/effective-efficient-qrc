"""
Data factory utilities for generating and saving synthetic windowed datasets.

This module provides a thin orchestration layer around:
- Hydra/OmegaConf configuration objects (`omegaconf.DictConfig`)
- Process generators (e.g., VARMA generators) instantiated via Hydra
- Label functionals instantiated via Hydra
- Artifact persistence (e.g., `.npz` or `.npy`) into the project's storage tree

The intended usage is:
1) Compose a Hydra config describing a single dataset parametrization.
2) Call :func:`generate_and_save_dataset` to build an in-memory dataset and persist it.

Notes
-----
- In OmegaConf, some keys may collide with dict-like methods (e.g., `items`, `keys`).
  For safety, this module uses bracket access for such keys, e.g. `cfg.functionals["items"]`.

See Also
--------
src.experiment.scripts.generate_data : Hydra entrypoint to generate datasets from configs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import hashlib
import json

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.data.generate.base import WindowsDataset
from src.settings import PROJECT_ROOT_PATH


@dataclass(frozen=True)
class DatasetArtifact:
    """
    A saved dataset artifact descriptor.

    Attributes
    ----------
    root : pathlib.Path
        Root directory where the artifact is stored.
    data_path : pathlib.Path
        Path to the main saved data file. For `.npz`, this is the `.npz` file.
        For `.npy`, this points to the X array file (`*.X.npy`).
    meta_path : pathlib.Path or None
        Path to a JSON metadata file, if `output.save_meta=true`.
    config_path : pathlib.Path or None
        Path to a resolved YAML config file, if `output.save_config=true`.
    """

    root: Path
    data_path: Path
    meta_path: Path | None
    config_path: Path | None


def _as_project_path(p: str | Path) -> Path:
    """
    Convert a path into an absolute project path when needed.

    Parameters
    ----------
    p : str or pathlib.Path
        User-provided path. If absolute, returned unchanged.
        If relative, it is interpreted relative to `PROJECT_ROOT_PATH`.

    Returns
    -------
    pathlib.Path
        Absolute path suitable for IO operations.
    """
    p = Path(p)
    if p.is_absolute():
        return p
    return Path(PROJECT_ROOT_PATH) / p


def _stable_cfg_hash(cfg: DictConfig) -> str:
    """
    Compute a short, stable hash of a fully-resolved config.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Hydra/OmegaConf config to hash.

    Returns
    -------
    str
        12-character SHA1 prefix of the JSON-serialized resolved config.
    """
    obj = OmegaConf.to_container(cfg, resolve=True)
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def build_generator(cfg: DictConfig) -> Any:
    """
    Instantiate the process generator from the config.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Expected to contain `cfg.process.generator` with a Hydra `_target_`.

    Returns
    -------
    Any
        An instantiated generator object (e.g., VARMAGenerator) implementing
        `make_windows_dataset(...)`.
    """
    return instantiate(cfg.process.generator)


def build_label_functionals(cfg: DictConfig) -> List[Any]:
    """
    Instantiate label functionals from the config.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Expected to contain a list under `cfg.functionals["items"]`, each with
        a Hydra `_target_`.

    Returns
    -------
    list of Any
        A list of instantiated functional objects.

    Raises
    ------
    TypeError
        If `cfg.functionals["items"]` is not a list-like object.

    Notes
    -----
    We use bracket access `cfg.functionals["items"]` (not attribute access)
    because `DictConfig` has a method named `.items()`.
    """
    items = cfg.functionals["items"]
    if not isinstance(items, (list, ListConfig)):
        raise TypeError(f"cfg.functionals['items'] must be a list, got {type(items)}")
    return [instantiate(item) for item in items]


def make_dataset(cfg: DictConfig) -> WindowsDataset:
    """
    Generate a dataset in-memory based on the config.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Must define, at minimum:
        - `cfg.sampling.N`, `cfg.sampling.w`, `cfg.sampling.d`
        - `cfg.process.generator` (Hydra instantiable target)
        - `cfg.functionals["items"]` (list of Hydra instantiable targets)

    Returns
    -------
    src.data.generate.base.WindowsDataset
        Dataset object with attributes:
        - `X`: ndarray of shape (N, w, d)
        - `y`: ndarray of shape (L, N)
        - `meta`: dict containing generation metadata
    """
    N = int(cfg.sampling.N)
    w = int(cfg.sampling.w)
    d = int(cfg.sampling.d)

    gen = build_generator(cfg)
    labs = build_label_functionals(cfg)

    ds = gen.make_windows_dataset(N=N, w=w, d=d, label_functionals=labs)

    # augment meta with some high-level info
    meta = dict(ds.meta)
    meta.update(
        {
            "N": N,
            "w": w,
            "d": d,
            "L": int(ds.y.shape[0]),
            "process_kind": str(cfg.process.kind),
            "functionals_kind": str(cfg.functionals.kind),
            "seed": int(cfg.seed),
            "cfg_hash": _stable_cfg_hash(cfg),
        }
    )
    ds.meta = meta
    return ds


def _auto_name(cfg: DictConfig) -> str:
    """
    Build a deterministic dataset artifact name from key config fields.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Dataset config.

    Returns
    -------
    str
        A filesystem-friendly name encoding process/functional and main hyperparams.
    """
    return "data"


def save_dataset(ds: WindowsDataset, cfg: DictConfig) -> DatasetArtifact:
    """
    Save a dataset to disk according to `cfg.output.*`.

    Parameters
    ----------
    ds : src.data.generate.base.WindowsDataset
        Dataset to persist. Must have `.X`, `.y`, `.meta`.
    cfg : omegaconf.DictConfig
        Must define:
        - `cfg.output.save_dir`
        - `cfg.output.name` ("auto" allowed)
        - `cfg.output.format` in {"npz", "npy"}
        - `cfg.output.overwrite` (bool)

    Returns
    -------
    DatasetArtifact
        Descriptor of saved files (data/meta/config).

    Raises
    ------
    FileExistsError
        If the destination exists and `output.overwrite` is false.
    ValueError
        If `output.format` is unknown.
    """
    out = cfg.output
    save_dir = _as_project_path(out.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    name = str(out.name)
    if name.lower() == "auto":
        name = _auto_name(cfg)

    fmt = str(out.format).lower()
    overwrite = bool(out.get("overwrite", False))

    if fmt == "npz":
        data_path = save_dir / f"{name}.npz"
        if data_path.exists() and not overwrite:
            raise FileExistsError(f"{data_path} exists (set output.overwrite=true to overwrite).")
        np.savez_compressed(data_path, X=ds.X, y=ds.y)

    elif fmt == "npy":
        # store as two separate arrays, but still return a "data_path" pointing to X
        x_path = save_dir / f"{name}.X.npy"
        y_path = save_dir / f"{name}.y.npy"
        if (x_path.exists() or y_path.exists()) and not overwrite:
            raise FileExistsError(f"{name} .npy exists (set output.overwrite=true to overwrite).")
        np.save(x_path, ds.X)
        np.save(y_path, ds.y)
        data_path = x_path

    else:
        raise ValueError(f"Unknown output.format={fmt!r}")

    meta_path = None
    if bool(out.get("save_meta", True)):
        meta_path = save_dir / f"{name}.meta.json"
        meta_path.write_text(json.dumps(ds.meta, indent=2), encoding="utf-8")

    config_path = None
    if bool(out.get("save_config", True)):
        config_path = save_dir / f"{name}.config.yaml"
        OmegaConf.save(cfg, config_path, resolve=True)

    return DatasetArtifact(root=save_dir, data_path=data_path, meta_path=meta_path, config_path=config_path)


def generate_and_save_dataset(cfg: DictConfig) -> Tuple[WindowsDataset, DatasetArtifact]:
    """
    Convenience wrapper: generate a dataset then persist it.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Dataset config.

    Returns
    -------
    (WindowsDataset, DatasetArtifact)
        The in-memory dataset and the corresponding artifact descriptor.
    """
    ds = make_dataset(cfg)
    art = save_dataset(ds, cfg)
    return ds, art
