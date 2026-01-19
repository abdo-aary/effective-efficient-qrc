from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import Matern as SkMatern, ConstantKernel

from src.models.kernel import (
    tune_matern_grid_train_val,
    tune_matern_continuous_train_val,
)
from src.models.qrc_featurizer import QRCFeaturizer

from src.qrc.circuits.configs import RingQRConfig
from src.qrc.circuits.utils import generate_k_local_paulis
from src.qrc.run.circuit_run import ExactAerCircuitsRunner
from src.qrc.run.fmp_retriever import ExactFeatureMapsRetriever
from src.qrc.run.cs_fmp_retriever import CSFeatureMapsRetriever

from qiskit.quantum_info import SparsePauliOp

# -----------------------------
# Small registries (YAML -> impl)
# -----------------------------

_CFG_REGISTRY = {
    "ring": RingQRConfig,
}

_RUNNER_REGISTRY = {
    "exact_aer": ExactAerCircuitsRunner,
}

_FMP_REGISTRY = {
    "exact": ExactFeatureMapsRetriever,
    "cs": CSFeatureMapsRetriever,
}


def _train_test_split_indices(N: int, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_test = max(1, int(test_ratio * N))
    if n_test >= N:
        n_test = N - 1
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx


def _build_kernel(xi: float, nu: float) -> Any:
    # fixed amplitude=1.0; we can expose this if we want
    return ConstantKernel(1.0, constant_value_bounds="fixed") * SkMatern(
        length_scale=float(xi),
        length_scale_bounds="fixed",
        nu=float(nu),
    )


class QRCMaternKRRRegressor(BaseEstimator, RegressorMixin):
    """
    End-to-end: windows -> quantum features -> Matérn KRR.

    - fit(X,y): computes features once, does split + tuning on train-val, fits KRR on train-val,
                stores test split so we can call predict() with X=None.
    - predict(X=None): if X None, predicts on stored test set windows (from fit).
    """

    def __init__(
            self,
            featurizer: QRCFeaturizer,
            *,
            standardize: bool = True,
            test_ratio: float = 0.2,
            split_seed: int = 0,
            tuning: Optional[Dict[str, Any]] = None,
    ):
        self.featurizer = featurizer
        self.standardize = bool(standardize)
        self.test_ratio = float(test_ratio)
        self.split_seed = int(split_seed)
        self.tuning = tuning or {}

        # learned attrs
        self.scaler_: Optional[StandardScaler] = None
        self.kernel_: Any = None
        self.alpha_: Optional[np.ndarray] = None
        self.X_train_features_: Optional[np.ndarray] = None

        self.X_test_: Optional[np.ndarray] = None
        self.y_test_: Optional[np.ndarray] = None
        self.y_pred_test_: Optional[np.ndarray] = None

        self.best_params_: Optional[Dict[str, float]] = None

    @staticmethod
    def from_yaml(path: str) -> "QRCMaternKRRRegressor":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # --- QR config
        qrc_cfg = cfg["qrc"]["cfg"]
        cfg_kind = qrc_cfg["kind"]
        cfg_cls = _CFG_REGISTRY.get(cfg_kind)
        if cfg_cls is None:
            raise ValueError(f"Unknown qrc.cfg.kind={cfg_kind!r}")
        qr_cfg = cfg_cls(
            input_dim=int(qrc_cfg["input_dim"]),
            num_qubits=int(qrc_cfg["num_qubits"]),
            seed=int(qrc_cfg.get("seed", 0)),
        )

        # --- observables
        obs_cfg = cfg["qrc"]["features"]["observables"]
        if obs_cfg["kind"] == "k_local_pauli":
            locality = int(obs_cfg["locality"])
            observables = generate_k_local_paulis(locality=locality, num_qubits=qr_cfg.num_qubits)
        elif obs_cfg["kind"] == "explicit_pauli":
            labels = list(obs_cfg["labels"])
            observables = [SparsePauliOp(lab) for lab in labels]
        else:
            raise ValueError(f"Unknown observables.kind={obs_cfg['kind']!r}")

        # --- runner
        runner_cfg = cfg["qrc"]["runner"]
        runner_cls = _RUNNER_REGISTRY.get(runner_cfg["kind"])
        if runner_cls is None:
            raise ValueError(f"Unknown runner.kind={runner_cfg['kind']!r}")
        runner = runner_cls(qr_cfg)
        runner_kwargs = dict(runner_cfg.get("kwargs", {}))

        # --- feature maps retriever
        fmp_cfg = cfg["qrc"]["features"]
        fmp_kind = fmp_cfg["kind"]
        fmp_cls = _FMP_REGISTRY.get(fmp_kind)
        if fmp_cls is None:
            raise ValueError(f"Unknown features.kind={fmp_kind!r}")

        if fmp_kind == "exact":
            fmp = fmp_cls(qr_cfg, observables)
        elif fmp_kind == "cs":
            # allows defaults like default_shots/default_n_groups in constructor if we want
            fmp = fmp_cls(qr_cfg, observables)
        else:
            raise ValueError(f"Unhandled features.kind={fmp_kind!r}")

        fmp_kwargs = dict(fmp_cfg.get("kwargs", {}))

        # --- pubs config
        pubs_cfg = cfg["qrc"]["pubs"]
        pubs_family = pubs_cfg["family"]
        angle_positioning = pubs_cfg["angle_positioning"]
        pubs_kwargs = dict(pubs_cfg)
        pubs_kwargs.pop("family")
        pubs_kwargs.pop("angle_positioning")

        # --- model config
        model_cfg = cfg["model"]
        standardize = bool(model_cfg.get("preprocess", {}).get("standardize", True))
        split = model_cfg.get("split", {})
        test_ratio = float(split.get("test_ratio", 0.2))
        split_seed = int(split.get("seed", 0))
        tuning = dict(model_cfg.get("tuning", {}))

        featurizer = QRCFeaturizer(
            cfg=qr_cfg,
            runner=runner,
            fmp_retriever=fmp,
            pubs_family=pubs_family,
            angle_positioning_name=angle_positioning,
            pubs_kwargs=pubs_kwargs,
            runner_kwargs=runner_kwargs,
            fmp_kwargs=fmp_kwargs,
        )
        return QRCMaternKRRRegressor(
            featurizer,
            standardize=standardize,
            test_ratio=test_ratio,
            split_seed=split_seed,
            tuning=tuning,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)

        if X.ndim != 3:
            raise ValueError(f"X must be (N,w,d). Got {X.shape}.")
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"y must have shape (N,), got {y.shape} for X {X.shape}.")

        # quantum featurization (single expensive call)
        Phi = self.featurizer.transform(X)  # (N,D)

        # split train_val vs test
        tr_idx, te_idx = _train_test_split_indices(X.shape[0], self.test_ratio, self.split_seed)
        Phi_tr, y_tr = Phi[tr_idx], y[tr_idx]
        Phi_te, y_te = Phi[te_idx], y[te_idx]

        # optional standardization (recommended for kernel lengthscale stability)
        if self.standardize:
            self.scaler_ = StandardScaler()
            Phi_tr = self.scaler_.fit_transform(Phi_tr)
            Phi_te = self.scaler_.transform(Phi_te)
        else:
            self.scaler_ = None

        # tune Matérn hyperparameters on train_val
        strategy = self.tuning.get("strategy", "grid")
        val_ratio = float(self.tuning.get("val_ratio", 0.2))
        tune_seed = int(self.tuning.get("seed", 0))
        reg = float(self.tuning.get("reg", 1e-6))
        xi_bounds = tuple(self.tuning.get("xi_bounds", (1e-3, 1e3)))

        if strategy == "grid":
            nu_grid = self.tuning.get("nu_grid", (0.5, 1.5, 2.5, 5.0))
            xi_maxiter = int(self.tuning.get("xi_maxiter", 80))
            best_params, _ = tune_matern_grid_train_val(
                Phi_tr, y_tr,
                val_ratio=val_ratio,
                seed=tune_seed,
                reg=reg,
                xi_bounds=xi_bounds,
                nu_grid=nu_grid,
                xi_maxiter=xi_maxiter,
            )
        elif strategy == "powell":
            nu_bounds = tuple(self.tuning.get("nu_bounds", (0.2, 5.0)))
            n_restarts = int(self.tuning.get("n_restarts", 8))
            best_params, _ = tune_matern_continuous_train_val(
                Phi_tr, y_tr,
                val_ratio=val_ratio,
                seed=tune_seed,
                reg=reg,
                xi_bounds=xi_bounds,
                nu_bounds=nu_bounds,
                n_restarts=n_restarts,
            )
        else:
            raise ValueError(f"Unknown tuning.strategy={strategy!r}")

        self.best_params_ = dict(best_params)
        xi = float(best_params["xi"])
        nu = float(best_params["nu"])
        reg = float(best_params["reg"])

        # fit final KRR on all train_val
        self.kernel_ = _build_kernel(xi=xi, nu=nu)
        Ktt = self.kernel_(Phi_tr, Phi_tr)
        A = Ktt + reg * np.eye(Ktt.shape[0])
        self.alpha_ = np.linalg.solve(A, y_tr)
        self.X_train_features_ = Phi_tr

        # store test set to allow predict() with X=None
        self.X_test_ = X[te_idx]
        self.y_test_ = y_te

        # optional: compute & store test predictions immediately
        self.y_pred_test_ = self._predict_from_features(Phi_te)
        return self

    def _predict_from_features(self, Phi: np.ndarray) -> np.ndarray:
        if self.alpha_ is None or self.kernel_ is None or self.X_train_features_ is None:
            raise RuntimeError("Model is not fitted yet.")
        Kxt = self.kernel_(Phi, self.X_train_features_)
        return Kxt @ self.alpha_

    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        # sklearn-like: predict(X). Convenience: predict() predicts on stored test set.
        if X is None:
            if self.X_test_ is None:
                raise ValueError("No X provided and no stored test set. Call fit(...) first or pass X.")

            # Return the test predictions previously computed in .fit().
            return self.y_pred_test_

        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"X must be (N,w,d). Got {X.shape}.")

        Phi = self.featurizer.transform(X)
        if self.scaler_ is not None:
            Phi = self.scaler_.transform(Phi)
        return self._predict_from_features(Phi)
