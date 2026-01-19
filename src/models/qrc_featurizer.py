from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from src.qrc.circuits.configs import BaseQRConfig
from src.qrc.circuits.circuit_factory import CircuitFactory
from src.qrc.circuits.utils import angle_positioning_linear
from src.qrc.run.circuit_run import BaseCircuitsRunner
from src.qrc.run.fmp_retriever import BaseFeatureMapsRetriever

_ANGLE_POS_REGISTRY = {
    "linear": angle_positioning_linear,
}


@dataclass
class QRCFeaturizer:
    """
    Stateless featurizer: X(N,w,d) -> Phi(N,D)

    Can later be made sklearn-compatible by inheriting TransformerMixin,
    but keeping it simple is fine.
    """
    cfg: BaseQRConfig
    runner: BaseCircuitsRunner
    fmp_retriever: BaseFeatureMapsRetriever
    pubs_family: str
    angle_positioning_name: str
    pubs_kwargs: Dict[str, Any]
    runner_kwargs: Dict[str, Any]
    fmp_kwargs: Dict[str, Any]

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(f"X must be (N,w,d). Got {X.shape}.")

        # Right now we have one concrete family in CircuitFactory
        if self.pubs_family != "ising_ring_swap":
            raise ValueError(f"Unknown pubs_family={self.pubs_family!r}. Add it to the featurizer.")

        angle_positioning = _ANGLE_POS_REGISTRY.get(self.angle_positioning_name)
        if angle_positioning is None:
            raise ValueError(f"Unknown angle_positioning={self.angle_positioning_name!r}")

        pubs = CircuitFactory.create_pubs_dataset_reservoirs_IsingRingSWAP(
            cfg=self.cfg,
            angle_positioning=angle_positioning,
            X=X,
            **self.pubs_kwargs,
        )

        results = self.runner.run_pubs(pubs=pubs, **self.runner_kwargs)

        # CSFeatureMapsRetriever expects shots/seed/n_groups in get_feature_maps
        Phi = self.fmp_retriever.get_feature_maps(results, **self.fmp_kwargs)
        if Phi.ndim != 2 or Phi.shape[0] != X.shape[0]:
            raise ValueError(f"Feature maps must be (N,D). Got {Phi.shape}.")
        return Phi
