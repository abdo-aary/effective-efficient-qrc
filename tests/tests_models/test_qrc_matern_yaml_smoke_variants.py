import numpy as np
import pytest
import yaml
import src.models.qrc_matern_krr as mk
from src.qrc.run.circuit_run import ExactResults


def _cfg_dict(
        *,
        num_qubits: int = 2,
        input_dim: int = 4,
        runner_kwargs: dict,
        fmp_kind: str = "exact",
        fmp_kwargs: dict | None = None,
        angle_positioning: str = "tanh",
        tuning_strategy: str = "grid",
        standardize: bool = True,
):
    if fmp_kwargs is None:
        fmp_kwargs = {}

    # labels must match num_qubits length for explicit_pauli
    n = int(num_qubits)
    if n == 1:
        labels = ["Z", "X"]
    else:
        # Use simple 1-local Z observables on different qubits: "ZII...", "IZI...", ...
        labels = []
        for i in range(min(n, 3)):  # keep it small for smoke tests
            lab = ["I"] * n
            lab[i] = "Z"
            labels.append("".join(lab))

    cfg = {
        "qrc": {
            "cfg": {
                "kind": "ring",
                "input_dim": int(input_dim),
                "num_qubits": int(num_qubits),
                "seed": 0,
            },
            "pubs": {
                "family": "ising_ring_swap",
                "angle_positioning": str(angle_positioning),
                "num_reservoirs": 2,
                "lam_0": 0.1,
                "seed": 0,
                "eps": 1e-8,
            },
            "runner": {
                "kind": "exact_aer",
                "kwargs": dict(runner_kwargs),
            },
            "features": {
                "kind": str(fmp_kind),
                "observables": {
                    "kind": "explicit_pauli",
                    "labels": labels,
                },
                "kwargs": dict(fmp_kwargs),
            },
        },
        "model": {
            "preprocess": {"standardize": bool(standardize)},
            "split": {"test_ratio": 0.25, "seed": 0},
            "tuning": {},  # filled below
        },
    }

    # IMPORTANT: your from_yaml expects model.tuning (not root.tuning) :contentReference[oaicite:2]{index=2}
    if tuning_strategy == "grid":
        cfg["model"]["tuning"] = {
            "strategy": "grid",
            "val_ratio": 0.25,
            "seed": 0,
            "reg": 1e-6,
            "xi_bounds": [1e-3, 1e3],
            "nu_grid": [1.5],
            "xi_maxiter": 8,
        }
    elif tuning_strategy == "powell":
        cfg["model"]["tuning"] = {
            "strategy": "powell",
            "val_ratio": 0.25,
            "seed": 0,
            "reg": 1e-6,
            "xi_bounds": [1e-3, 1e3],
            "nu_bounds": [0.2, 5.0],
            "n_restarts": 1,
        }
    else:
        raise ValueError(f"Unknown tuning_strategy={tuning_strategy!r}")

    return cfg


def _write_cfg_yaml(path, cfg: dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _patch_runner_to_fake_states(monkeypatch):
    """
    Patch mk.ExactAerCircuitsRunner.run_pubs so:
      - it records kwargs
      - it returns a valid ExactResults(states=..., cfg=...)
      - it never calls Aer simulation (fast + GPU-agnostic)
    """

    def fake_run_pubs(self, pubs, **kwargs):
        self._last_run_kwargs = dict(kwargs)

        # infer N, R, dim
        N = len(pubs)
        if N == 0:
            raise ValueError("pubs must be non-empty")
        vals0 = pubs[0][1]
        R = int(vals0.shape[0])

        n = int(self.cfg.num_qubits)
        dim = 1 << n

        def sigmoid(x):
            x = float(x)
            if x >= 0:
                z = np.exp(-x)
                return 1.0 / (1.0 + z)
            else:
                z = np.exp(x)
                return z / (1.0 + z)

        states = np.zeros((N, R, dim, dim), dtype=complex)

        # make states depend (slightly) on params so features vary
        for i, (_qc, vals) in enumerate(pubs):
            vals = np.asarray(vals, dtype=float)
            for r in range(R):
                s = float(np.mean(vals[r]))
                p = sigmoid(s)
                p = float(np.clip(p, 1e-6, 1 - 1e-6))
                diag = np.full(dim, (1.0 - p) / (dim - 1), dtype=float)
                diag[0] = p
                rho = np.diag(diag).astype(complex)
                states[i, r] = rho

        return ExactResults(states=states, cfg=self.cfg)

    monkeypatch.setattr(mk.ExactAerCircuitsRunner, "run_pubs", fake_run_pubs, raising=True)


@pytest.mark.parametrize(
    "runner_kwargs",
    [
        {"device": "CPU"},
        {"device": "GPU"},  # we only check forwarding, not actual GPU availability
        {"device": "GPU", "max_parallel_threads": 2, "max_parallel_experiments": 2, "max_parallel_shots": 2,
         "seed_simulator": 123, "optimization_level": 0},
    ],
)
@pytest.mark.parametrize("tuning_strategy", ["grid", "powell"])
def test_from_yaml_end_to_end_smoke_runner_kwargs_forwarded(tmp_path, monkeypatch, runner_kwargs, tuning_strategy):
    _patch_runner_to_fake_states(monkeypatch)

    cfg = _cfg_dict(
        num_qubits=2,
        input_dim=4,
        runner_kwargs=runner_kwargs,
        fmp_kind="exact",
        fmp_kwargs={},
        tuning_strategy=tuning_strategy,
        standardize=True,
    )

    p = tmp_path / "cfg.yaml"
    _write_cfg_yaml(p, cfg)

    mdl = mk.QRCMaternKRRRegressor.from_yaml(str(p))

    rng = np.random.default_rng(0)
    N, w, d = 24, 5, 4
    X = rng.normal(size=(N, w, d))
    y = np.sin(2.0 * X.mean(axis=(1, 2))) + 0.01 * rng.standard_normal(N)

    mdl.fit(X, y)
    yhat = mdl.predict()

    assert yhat.shape == mdl.y_test_.shape
    assert np.all(np.isfinite(yhat))

    # verify that kwargs reached runner.run_pubs
    last = getattr(mdl.featurizer.runner, "_last_run_kwargs", None)
    assert isinstance(last, dict)
    for k, v in runner_kwargs.items():
        assert last.get(k) == v


def test_from_yaml_end_to_end_smoke_cs_feature_maps_kwargs_forwarded(tmp_path, monkeypatch):
    _patch_runner_to_fake_states(monkeypatch)

    # Patch CSFeatureMapsRetriever.get_feature_maps to record kwargs (forwarding check)
    orig_get = mk.CSFeatureMapsRetriever.get_feature_maps

    def wrapped_get(self, results, **kwargs):
        self._last_fmp_kwargs = dict(kwargs)
        return orig_get(self, results, **kwargs)

    monkeypatch.setattr(mk.CSFeatureMapsRetriever, "get_feature_maps", wrapped_get, raising=True)

    fmp_kwargs = {"shots": 32, "seed": 0, "n_groups": 4}

    cfg = _cfg_dict(
        num_qubits=2,
        input_dim=4,
        runner_kwargs={"device": "GPU"},  # just checking forwarding
        fmp_kind="cs",
        fmp_kwargs=fmp_kwargs,
        tuning_strategy="grid",
        standardize=False,
    )

    p = tmp_path / "cfg.yaml"
    _write_cfg_yaml(p, cfg)

    mdl = mk.QRCMaternKRRRegressor.from_yaml(str(p))

    rng = np.random.default_rng(1)
    N, w, d = 20, 4, 4
    X = rng.normal(size=(N, w, d))
    y = (X[:, -1, 0] - 0.5 * X[:, -1, 1]) + 0.01 * rng.standard_normal(N)

    mdl.fit(X, y)
    yhat = mdl.predict()
    assert np.all(np.isfinite(yhat))

    # ensure CS kwargs forwarded
    last_fmp = getattr(mdl.featurizer.fmp_retriever, "_last_fmp_kwargs", None)
    assert isinstance(last_fmp, dict)
    for k, v in fmp_kwargs.items():
        assert last_fmp.get(k) == v


def _aer_available_devices(backend) -> list[str]:
    if hasattr(backend, "available_devices"):
        try:
            return list(backend.available_devices())
        except Exception:
            return []
    return []


def _backend_device_option(backend):
    opts = getattr(backend, "options", None)
    if opts is None:
        return None
    return getattr(opts, "device", None)


def test_from_yaml_real_aer_gpu_device_sets_backend_option_or_skips(tmp_path):
    """
    End-to-end (YAML -> model -> featurizer.transform -> ExactAerCircuitsRunner.run_pubs):
    If Aer GPU is available, ensure the backend is actually configured with device='GPU'.
    Skips cleanly otherwise.
    """
    # IMPORTANT: do NOT call _patch_runner_to_fake_states(monkeypatch) here.

    rng = np.random.default_rng(0)
    N, w, d = 2, 2, 3
    X = rng.normal(size=(N, w, d)).astype(float)  # (N=1, w=2, d=1)

    cfg = _cfg_dict(
        num_qubits=3,
        input_dim=d,
        runner_kwargs={"device": "GPU", "optimization_level": 0, "seed_simulator": 0},
        fmp_kind="exact",
        fmp_kwargs={},
        tuning_strategy="grid",  # irrelevant: we won't call fit()
        standardize=False,
    )

    p = tmp_path / "cfg.yaml"
    _write_cfg_yaml(p, cfg)

    mdl = mk.QRCMaternKRRRegressor.from_yaml(str(p))

    backend = mdl.featurizer.runner.backend
    available = _aer_available_devices(backend)
    if "GPU" not in available:
        pytest.skip(f"Aer GPU not available (available_devices={available})")

    Phi = mdl.featurizer.transform(X)
    assert Phi.shape[0] == N
    assert np.all(np.isfinite(Phi))

    # This is the “for sure for sure” check:
    assert _backend_device_option(backend) == "GPU"
