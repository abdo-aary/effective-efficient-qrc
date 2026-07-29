"""Quantum reservoir featurization.

This module defines :class:`src.models.qrc_featurizer.QRCFeaturizer`, a lightweight
transformer that turns a window dataset ``X`` of shape ``(N, w, d)`` into a
structured feature batch. New code delegates to a backend-neutral program,
feature estimator, and execution backend. A legacy constructor remains for old
experiment manifests and uses the relocated Aer circuit/runner components.

The featurizer is intentionally *stateless*, so the same ``Phi`` can be reused for many downstream supervised
tasks (including multi-output labels) without rerunning circuits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from src.core.legacy_config import BaseQRConfig
from src.backends.aer.circuits import CircuitFactory
from src.backends.qiskit_utils import angle_positioning_linear, angle_positioning_tanh
from src.backends.aer.legacy_runner import BaseCircuitsRunner
from src.features.legacy_retrievers import BaseFeatureMapsRetriever
from src.core.program import QuaRKProgram
from src.core.requests import ExecutionSpec
from src.core.results import FeatureBatch
from src.estimators.base import FeatureEstimator

_ANGLE_POS_REGISTRY = {
    "linear": angle_positioning_linear,
    "tanh": angle_positioning_tanh,
}


@dataclass
class QRCFeaturizer:
    """
    Stateless quantum featurizer for window datasets.

    Parameters
    ----------
    qrc_cfg : src.core.legacy_config.BaseQRConfig
        Quantum reservoir configuration (number of qubits, input dimension, seed, ...).
    runner : src.backends.aer.legacy_runner.BaseCircuitsRunner
        Runner used by the compatibility constructor to execute PUBs.
    fmp_retriever : src.features.legacy_retrievers.BaseFeatureMapsRetriever
        Retriever that converts runner results into a feature matrix ``Phi``.
    pubs_family : str
        Name of the pubs family used to build circuits (e.g. ``"ising_ring_swap"``).
    angle_positioning_name : str
        Name of the angle positioning map to be applied to window values before they are used as circuit angles.
        Small numerical constant forwarded to the pubs builder (used for stability in some encodings).
    runner_kwargs : dict
        Keyword arguments forwarded to ``runner.run_pubs(...)`` (e.g. ``device="GPU"``).
    fmp_kwargs : dict
        Keyword arguments forwarded to ``fmp_retriever.get_feature_maps(...)`` (e.g. ``shots`` for CS retriever).

    Notes
    -----
    This class is conceptually similar to an ``sklearn`` transformer, but kept minimal on purpose.
    """
    qrc_cfg: BaseQRConfig | None = None
    runner: BaseCircuitsRunner | None = None
    fmp_retriever: BaseFeatureMapsRetriever | None = None
    pubs_family: str | None = None
    angle_positioning_name: str | None = None
    pubs_kwargs: Dict[str, Any] = field(default_factory=dict)
    runner_kwargs: Dict[str, Any] = field(default_factory=dict)
    fmp_kwargs: Dict[str, Any] = field(default_factory=dict)
    program: QuaRKProgram | None = None
    backend: Any | None = None
    estimator: FeatureEstimator | None = None
    execution: ExecutionSpec = field(default_factory=ExecutionSpec)
    last_feature_batch_: FeatureBatch | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_backend_api(
        cls,
        *,
        program: QuaRKProgram,
        backend: Any,
        estimator: FeatureEstimator,
        execution: ExecutionSpec | None = None,
    ) -> "QRCFeaturizer":
        """Construct a featurizer with no runner, retriever, or PUB objects."""

        return cls(
            program=program,
            backend=backend,
            estimator=estimator,
            execution=execution or ExecutionSpec(),
        )

    def transform_batch(self, X: np.ndarray) -> FeatureBatch:
        """Return structured features from the backend-neutral API."""

        if self.program is None or self.backend is None or self.estimator is None:
            raise RuntimeError(
                "transform_batch requires program, backend, and estimator; "
                "this featurizer was constructed through the legacy runner API."
            )
        batch = self.estimator.estimate(
            self.program,
            X,
            self.backend,
            self.execution,
        )
        self.last_feature_batch_ = batch
        return batch

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Compute a feature matrix for a batch of windows.

        Parameters
        ----------
        X : numpy.ndarray
            Window dataset with shape ``(N, w, d)``.

        Returns
        -------
        numpy.ndarray
            Feature matrix ``Phi`` with shape ``(N, D)``.

        Raises
        ------
        ValueError
            If the input shape is not ``(N, w, d)``.
        """
        if X.ndim != 3:
            raise ValueError(f"X must be (N,w,d). Got {X.shape}.")

        if self.program is not None or self.backend is not None or self.estimator is not None:
            if self.program is None or self.backend is None or self.estimator is None:
                raise ValueError(
                    "Backend-neutral featurization requires program, backend, and estimator together."
                )
            return self.transform_batch(X).flatten_for_readout()

        if self.qrc_cfg is None or self.runner is None or self.fmp_retriever is None:
            raise ValueError(
                "Legacy featurization requires qrc_cfg, runner, and fmp_retriever."
            )
        if self.pubs_family is None or self.angle_positioning_name is None:
            raise ValueError(
                "Legacy featurization requires pubs_family and angle_positioning_name."
            )

        angle_positioning = _ANGLE_POS_REGISTRY.get(self.angle_positioning_name)
        if angle_positioning is None:
            raise ValueError(f"Unknown angle_positioning={self.angle_positioning_name!r}")

        if self.pubs_family == "ising_ring_swap":
            pubs = CircuitFactory.create_pubs_dataset_reservoirs_IsingRingSWAP(
                qrc_cfg=self.qrc_cfg,
                angle_positioning=angle_positioning,
                X=X,
                **self.pubs_kwargs,
            )
        else:
            raise ValueError(f"Unknown pubs_family={self.pubs_family!r}. Add it to the featurizer.")

        runner_kwargs = dict(self.runner_kwargs)
        if getattr(self.runner, "requires_observables", False):
            runner_kwargs.setdefault("observables", getattr(self.fmp_retriever, "observables", None))
        if getattr(self.runner, "requires_angle_positioning_name", False):
            runner_kwargs.setdefault("angle_positioning_name", self.angle_positioning_name)
        results = self.runner.run_pubs(pubs=pubs, **runner_kwargs)

        # CSFeatureMapsRetriever expects shots/seed/n_groups in get_feature_maps
        Phi = self.fmp_retriever.get_feature_maps(results, **self.fmp_kwargs)
        if Phi.ndim != 2 or Phi.shape[0] != X.shape[0]:
            raise ValueError(f"Feature maps must be (N,D). Got {Phi.shape}.")
        return Phi
