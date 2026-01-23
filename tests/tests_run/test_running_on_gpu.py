import numpy as np
import pytest


def _aer_available_devices(backend) -> list[str]:
    if hasattr(backend, "available_devices"):
        try:
            return list(backend.available_devices())
        except Exception:
            return []
    return []


def _backend_device_option(backend):
    # AerSimulator exposes options as an object; device is usually an attribute.
    opts = getattr(backend, "options", None)
    if opts is None:
        return None
    return getattr(opts, "device", None)


def test_real_gpu_runner_executes_factory_pubs_or_skips():
    """
    Optional integration test:
    - Runs a real pub produced by CircuitFactory (so metadata keys exist).
    - Requests Aer GPU device.
    - Skips if GPU isn't available for this Aer install.
    """
    from src.qrc.circuits.configs import RingQRConfig
    from src.qrc.circuits.circuit_factory import CircuitFactory
    from src.qrc.circuits.utils import angle_positioning_tanh
    from src.qrc.run.circuit_run import ExactAerCircuitsRunner

    rng = np.random.default_rng(0)
    # tiny dataset: N=1 window, w=2 timesteps, d=input_dim=2
    N, w, d = 1, 2, 1
    X = rng.normal(size=(N, w, d)).astype(float)

    num_qubits = 2
    cfg = RingQRConfig(input_dim=d, num_qubits=num_qubits, seed=0)
    runner = ExactAerCircuitsRunner(cfg)
    backend = runner.backend

    available = _aer_available_devices(backend)
    if "GPU" not in available:
        pytest.skip(f"Aer GPU not available (available_devices={available})")

    num_reservoirs = 3

    pubs = CircuitFactory.create_pubs_dataset_reservoirs_IsingRingSWAP(
        cfg=cfg,
        angle_positioning=angle_positioning_tanh,
        X=X,
        lam_0=0.2,
        num_reservoirs=num_reservoirs,  # keep it tiny
        seed=0,
        eps=1e-8,
    )

    res = runner.run_pubs(
        pubs=pubs,
        device="GPU",
        optimization_level=0,
        seed_simulator=0,
        max_threads=1,
        max_parallel_threads=1,
        max_parallel_experiments=1,
        max_parallel_shots=1,
    )

    assert res.states.shape == (N, num_reservoirs, 2 ** num_qubits, 2 ** num_qubits)
    assert _backend_device_option(backend) == "GPU"
