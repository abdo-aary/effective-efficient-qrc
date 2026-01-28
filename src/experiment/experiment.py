"""High-level experiment wrapper.

The goal of :class:`~src.experiment.experiment.Experiment` is to provide a thin,
test-friendly orchestration layer for the numerical validation section.

Responsibilities (initial version)
---------------------------------
1) Load a :class:`~src.data.generate.base.WindowsDataset` from disk.
2) Instantiate a :class:`~src.models.qrc_matern_krr.QRCMaternKRRRegressor` either
   from an OmegaConf/Hydra :class:`~omegaconf.DictConfig` or from an already
   constructed regressor.
3) Train the regressor via :meth:`~src.models.qrc_matern_krr.QRCMaternKRRRegressor.fit`.
4) (Placeholder) Compute per-functional metrics.
5) Persist the model via the regressor's persistence interface.

This class is intentionally minimalist: it does not aim to replace Hydra scripts;
instead it provides an object-oriented API that can be composed from notebooks,
unit tests, and future CLI entrypoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig

from src.data.factory import DatasetArtifact, load_windows_dataset
from src.data.generate.base import WindowsDataset
from src.models.qrc_matern_krr import QRCMaternKRRRegressor


def _default_functional_names(ds: WindowsDataset) -> list[str]:
    """Best-effort functional names for reporting.

    Priority order:
    1) class names of instantiated label functionals
    2) meta["functionals_kind"] + index
    3) fallback "functional_{i}"
    """
    L = int(ds.y.shape[0]) if getattr(ds.y, "ndim", 0) >= 2 else 1

    if ds.label_functionals and len(ds.label_functionals) == L:
        return [lf.__class__.__name__ for lf in ds.label_functionals]

    kind = None
    if isinstance(ds.meta, dict):
        kind = ds.meta.get("functionals_kind")
    if kind is None:
        return [f"functional_{i}" for i in range(L)]
    return [f"{kind}[{i}]" for i in range(L)]


@dataclass
class Experiment:
    """Orchestrate dataset loading, model training, and persistence."""

    dataset: WindowsDataset
    model: QRCMaternKRRRegressor
    dataset_artifact: Optional[DatasetArtifact] = None

    @classmethod
    def from_paths(
        cls,
        *,
        dataset_path: str | Path,
        model_cfg: DictConfig | None = None,
        model: QRCMaternKRRRegressor | None = None,
        instantiate_functionals: bool = True,
    ) -> "Experiment":
        """Build an experiment from a dataset path plus either a model cfg or model instance."""

        ds, art = load_windows_dataset(dataset_path, instantiate_functionals=instantiate_functionals)

        if model is None:
            if model_cfg is None:
                raise ValueError("You must provide either model_cfg or model")
            model = QRCMaternKRRRegressor.from_config(model_cfg)

        return cls(dataset=ds, model=model, dataset_artifact=art)

    def fit(self, *, num_workers: int = 1, **fit_kwargs: Any) -> "Experiment":
        """Train the wrapped model on the loaded dataset."""
        self.model.fit(self.dataset.X, self.dataset.y, num_workers=int(num_workers), **fit_kwargs)
        return self

    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute metrics per functional.

        Placeholder: returns an empty dict per functional.
        """
        names = _default_functional_names(self.dataset)
        return {name: {} for name in names}

    def save_model(self, path: str | Path) -> None:
        """Persist the trained model.

        This delegates to :meth:`src.models.qrc_matern_krr.QRCMaternKRRRegressor.save`.
        """
        self.model.save(path)

    def load_model(self, path: str | Path, *, featurizer=None, **init_kwargs: Any) -> QRCMaternKRRRegressor:
        """Load a persisted model artifact and attach it to this experiment.

        This delegates to :meth:`src.models.qrc_matern_krr.QRCMaternKRRRegressor.load` and then
        *rehydrates* ``X_train_features_`` from the cached ``Phi_full_`` and stored train indices,
        so that :meth:`~src.models.qrc_matern_krr.QRCMaternKRRRegressor.predict_from_features` works
        immediately without re-running the featurizer.

        Parameters
        ----------
        path:
            Directory containing ``arrays.npz`` and ``meta.json`` created by ``model.save``.
        featurizer:
            Optional featurizer to inject into the loaded model (only needed if you plan to call
            ``model.predict(X)`` on new windows).
        init_kwargs:
            Forwarded to the regressor constructor in ``QRCMaternKRRRegressor.load``.

        Returns
        -------
        QRCMaternKRRRegressor
            The loaded model instance (also stored in ``self.model``).
        """
        loaded = QRCMaternKRRRegressor.load(path, featurizer=featurizer, **init_kwargs)

        # Rehydrate training features needed for kernel prediction.
        Phi_tr = loaded.Phi_full_[loaded.train_idx_]
        Phi_tr = loaded.scaler_.transform(Phi_tr) if loaded.scaler_ is not None else Phi_tr
        loaded.X_train_features_ = Phi_tr

        self.model = loaded
        return loaded
