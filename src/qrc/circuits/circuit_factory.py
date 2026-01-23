"""src.qrc.circuits.circuit_factory
================================

Factory utilities to build the *quantum reservoir circuits* used throughout the
project.

The main entry points are:

- :meth:`CircuitFactory.createIsingRingCircuitSWAP` and
  :meth:`CircuitFactory.createIsingRingCircuitDynamic`, which build a single
  *per-time-step* reservoir circuit implementing

  ``data injection -> Ising unitary W -> contraction E_λ``

- :meth:`CircuitFactory.instantiateFullIsingRingEvolution`, which composes the
  per-time-step circuit over a full input window to obtain a circuit whose
  *input angles are bound*, while *reservoir parameters* (Ising couplings/fields
  and the contraction strength) remain free.

- :meth:`CircuitFactory.create_pubs_dataset_reservoirs_IsingRingSWAP`, which
  prepares a PUBS dataset compatible with :class:`src.qrc.run.circuit_run.ExactAerCircuitsRunner`.

Notes
-----
Metadata contract
~~~~~~~~~~~~~~~~~
Most downstream code (notably the Aer runner) expects the circuits returned by
this module to carry the following metadata keys:

- ``"z"``: :class:`qiskit.circuit.ParameterVector` for the injected coordinates
- ``"J"``: :class:`qiskit.circuit.ParameterVector` for ZZ couplings (one per edge)
- ``"h_x"``: :class:`qiskit.circuit.ParameterVector` for local X fields (one per qubit)
- ``"h_z"``: :class:`qiskit.circuit.ParameterVector` for local Z fields (one per qubit)
- ``"lam"``: :class:`qiskit.circuit.Parameter` for the contraction strength λ

The runner uses this metadata to reconstruct the expected parameter column order
for the PUBS arrays, which is *not* necessarily the same as ``qc.parameters``.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter

from src.qrc.circuits.configs import RingQRConfig


class CircuitFactory:
    """Factory for building QRC circuits and PUBS datasets.

    This class centralizes the construction of the reservoir circuit family used
    in the experiments/pipeline. All methods are `@staticmethod` to keep usage
    lightweight and stateless.

    See Also
    --------
    src.qrc.run.circuit_run.ExactAerCircuitsRunner
        Runner expecting the metadata contract described in this module docstring.
    """

    @staticmethod
    def createIsingRingCircuitDynamic(
        cfg: RingQRConfig,
        angle_positioning: callable,
        method: str = "density_matrix",
    ) -> QuantumCircuit:
        """Create the *stochastic* (dynamic) per-step Ising ring reservoir circuit.

        The circuit implements, on the reservoir register:

        1) **Data injection**: apply ``Ry(theta(z_j))`` on each reservoir qubit.
        2) **Ising unitary** ``W``: a layer of ``RZZ`` gates on ring edges followed
           by local ``Rz`` and ``Rx`` fields.
        3) **Contraction** ``E_λ`` via a *coin qubit* and **mid-circuit measurement**
           with conditional reset:

           - With probability ``λ``: do nothing (keep the post-``W`` state).
           - With probability ``1-λ``: reset the reservoir to ``|+>^{⊗n}``.

        Parameters
        ----------
        cfg : RingQRConfig
            Reservoir configuration, including the number of qubits and topology.
        angle_positioning : callable
            Function mapping a :class:`qiskit.circuit.ParameterVector` ``z`` to a
            list of angle expressions of length ``cfg.num_qubits``.
            Typical choices are :func:`src.qrc.circuits.utils.angle_positioning_linear`
            or :func:`src.qrc.circuits.utils.angle_positioning_tanh`.
        method : str, default="density_matrix"
            Implementation backend assumption. Only ``"density_matrix"`` is supported
            for this dynamic/conditional circuit variant.

        Returns
        -------
        QuantumCircuit
            A parameterized circuit with ``cfg.num_qubits + 1`` qubits (reservoir
            + coin) and 1 classical bit. The circuit carries metadata keys
            ``"z"``, ``"J"``, ``"h_x"``, ``"h_z"``, ``"lam"``.

        Raises
        ------
        NotImplementedError
            If `method` is not ``"density_matrix"``.

        Notes
        -----
        This construction uses classical control (``if_test``) and mid-circuit
        measurement, which may not be supported on all simulators/hardware.
        For a fully unitary and deterministic contraction, use
        :meth:`createIsingRingCircuitSWAP`.
        """
        if method != "density_matrix":
            raise NotImplementedError(
                f"Only method='density_matrix' is supported here; got {method}."
            )

        n = cfg.num_qubits
        topology = cfg.topology

        # Input (projected) parameters z in R^n (n = cfg.num_qubits)
        z = ParameterVector("z", cfg.num_qubits)

        # Shared injection angles (length should be n)
        angles = angle_positioning(z)

        zz_params = ParameterVector("J", len(topology.edges))
        rx_params = ParameterVector("h_x", n)
        rz_params = ParameterVector("h_z", n)
        lam_param = Parameter("lam")

        # +1 coin qubit (constant overhead), +1 classical bit for conditional reset
        qc = QuantumCircuit(n + 1, 1, name="SMC")
        qubits = list(range(n))
        coin = n  # last qubit
        cbit = qc.clbits[0]

        # 1) Data injection
        for i in range(n):
            qc.ry(angles[i], qubits[i])

        # 2) Ising unitary W
        for e, (q1, q2) in enumerate(topology.edges):
            qc.rzz(zz_params[e], q1, q2)

        for i in range(n):
            qc.rz(rz_params[i], qubits[i])
        for i in range(n):
            qc.rx(rx_params[i], qubits[i])

        # 3) Stochastic contraction via coin + conditional reset-to-|+>^{⊗n}
        qc.reset(coin)
        theta = 2 * (lam_param ** 0.5).arcsin()  # Ry(theta)|0> => P(|1>) = lam
        qc.ry(theta, coin)
        qc.measure(coin, cbit)

        # If coin == 0: reset each reservoir qubit to |0>, then H -> |+>
        with qc.if_test((cbit, 0)):
            qc.reset(qubits)
            qc.h(qubits)

        # Store parameter handles as metadata (used by runners)
        qc.metadata["z"] = z
        qc.metadata["J"] = zz_params
        qc.metadata["h_x"] = rx_params
        qc.metadata["h_z"] = rz_params
        qc.metadata["lam"] = lam_param

        return qc

    @staticmethod
    def createIsingRingCircuitSWAP(
        cfg: RingQRConfig,
        angle_positioning: callable,
    ) -> QuantumCircuit:
        r"""Create the *deterministic* per-step Ising ring reservoir circuit (SWAP dilation).

        This circuit implements the same high-level transformation as
        :meth:`createIsingRingCircuitDynamic`, but without mid-circuit measurement.
        The contraction channel is realized via a Stinespring dilation:

        .. math::

            E_\lambda(\rho) = \lambda\,\rho + (1-\lambda)\,|+\rangle\langle+|^{\otimes n}.

        Construction:

        - Reservoir register: qubits ``0..n-1``
        - Environment register: auxiliary qubits ``n..2n-1`` prepared in ``|+>^{⊗n}``
        - Coin qubit: ``2n`` prepared so that ``P(coin=1)=1-λ``
        - Controlled-SWAP between reservoir and environment conditioned on coin
        - Reset the environment to trace it out deterministically

        Parameters
        ----------
        cfg : RingQRConfig
            Reservoir configuration, including topology.
        angle_positioning : callable
            Map from injected coordinate vector ``z`` to per-qubit ``Ry`` angles.

        Returns
        -------
        QuantumCircuit
            Parameterized circuit with ``2*cfg.num_qubits + 1`` qubits. Metadata keys
            ``"z"``, ``"J"``, ``"h_x"``, ``"h_z"``, ``"lam"`` are set.

        Notes
        -----
        This version is simulator-friendly and backend-agnostic because it avoids
        mid-circuit measurement/conditional control.
        """

        n = cfg.num_qubits
        topology = cfg.topology

        z = ParameterVector("z", n)
        angles = angle_positioning(z)

        zz_params = ParameterVector("J", len(topology.edges))
        rx_params = ParameterVector("h_x", n)
        rz_params = ParameterVector("h_z", n)
        lam_param = Parameter("lam")

        # Qubit layout:
        #   reservoir: 0..n-1
        #   aux (+ state): n..2n-1
        #   coin: 2n
        qc = QuantumCircuit(2 * n + 1, name="SMC_det")
        res = list(range(n))
        aux = list(range(n, 2 * n))
        coin = 2 * n

        # 1) Data injection on reservoir
        for i in range(n):
            qc.ry(angles[i], res[i])

        # 2) Ising unitary W on reservoir
        for e, (q1, q2) in enumerate(topology.edges):
            qc.rzz(zz_params[e], q1, q2)

        for i in range(n):
            qc.rz(rz_params[i], res[i])
        for i in range(n):
            qc.rx(rx_params[i], res[i])

        # 3) Deterministic contraction via unitary dilation + env reset
        qc.reset(aux)
        qc.reset(coin)
        qc.h(aux)

        theta = 2 * ((1 - lam_param) ** 0.5).arcsin()
        qc.ry(theta, coin)

        for i in range(n):
            qc.cswap(coin, res[i], aux[i])

        qc.reset(coin)
        qc.reset(aux)

        qc.metadata["z"] = z
        qc.metadata["J"] = zz_params
        qc.metadata["h_x"] = rx_params
        qc.metadata["h_z"] = rz_params
        qc.metadata["lam"] = lam_param

        return qc

    @staticmethod
    def instantiateFullIsingRingEvolution(
        cfg: RingQRConfig,
        angle_positioning: callable,
        x_window: np.ndarray,
    ) -> QuantumCircuit:
        """Instantiate (bind) the input window and return a circuit free in reservoir parameters.

        Given a time window ``x_window`` of shape ``(w, d)`` (with ``d=cfg.input_dim``),
        this routine:

        1) Builds the per-step circuit (SWAP contraction version).
        2) Initializes the reservoir to ``|+>^{⊗n}``.
        3) For each time step ``t``, projects ``x_t`` to ``z_t = x_t @ cfg.projection`` and
           binds the circuit's input parameters ``z`` to ``z_t``.
        4) Composes the bound step circuit into a single window circuit.

        The returned circuit has *no remaining input parameters* (``z`` is bound),
        but keeps reservoir parameters ``J, h_x, h_z, lam`` free.

        Parameters
        ----------
        cfg : RingQRConfig
            Reservoir configuration. Must satisfy ``cfg.input_dim == x_window.shape[-1]``.
        angle_positioning : callable
            Injection angle map used by the per-step circuit.
        x_window : numpy.ndarray
            Input window of shape ``(w, d)``. By convention, ``x_window[0]`` corresponds
            to the oldest sample in the window.

        Returns
        -------
        QuantumCircuit
            A composed circuit representing the full evolution over the window.
            Metadata keys ``"J"``, ``"h_x"``, ``"h_z"``, ``"lam"`` are copied from the
            per-step circuit for downstream runners.

        Raises
        ------
        AssertionError
            If `x_window` does not have shape ``(w, cfg.input_dim)``.
        """
        assert x_window.shape[-1] == cfg.input_dim, (
            "Mismatch between the window dimension and input_dim in the provided config. "
            f"Got x_window.shape[-1]={x_window.shape[-1]} and cfg.input_dim={cfg.input_dim}."
        )
        assert x_window.ndim == 2, (
            f"x_window should be a 2D array of shape (w, d), got x_window.ndim={x_window.ndim}."
        )

        qc_reservoir = CircuitFactory.createIsingRingCircuitSWAP(
            cfg=cfg, angle_positioning=angle_positioning
        )

        n = cfg.num_qubits
        qubits = list(range(n))
        z_params = qc_reservoir.metadata["z"]

        qc = QuantumCircuit(qc_reservoir.num_qubits, name="QRC")
        qc.h(qubits[:n])

        for x_t in x_window:
            z_t = x_t @ cfg.projection
            bind_map_input = dict(zip(z_params, z_t))
            qc_step_bound = qc_reservoir.assign_parameters(bind_map_input)
            qc.compose(qc_step_bound, inplace=True)

        qc.metadata["J"] = qc_reservoir.metadata["J"]
        qc.metadata["h_x"] = qc_reservoir.metadata["h_x"]
        qc.metadata["h_z"] = qc_reservoir.metadata["h_z"]
        qc.metadata["lam"] = qc_reservoir.metadata["lam"]
        return qc

    @staticmethod
    def set_reservoirs_parameterizationSWAP(
        cfg: RingQRConfig,
        angle_positioning: callable,
        num_reservoirs: int,
        lam_0: float,
        seed: float = 12345,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """Sample reservoir parameter sets for ``R`` reservoirs (SWAP circuit variant).

        Reservoir parameters are sampled as follows:

        - ZZ couplings ``J``: iid Uniform(-π, π) for each edge.
        - Local fields ``h_x`` and ``h_z``: iid Uniform(-π, π) per qubit.
        - Contraction strengths ``λ``:
            - reservoir 0 uses ``lam_0`` (user-chosen)
            - reservoirs 1..R-1 use iid Uniform(eps, 1-eps)

        Parameters
        ----------
        cfg : RingQRConfig
            Reservoir configuration.
        angle_positioning : callable
            Injection angle map (only used to build the parameter vectors; it does
            not affect the sampled values).
        num_reservoirs : int
            Number of reservoirs ``R`` to sample.
        lam_0 : float
            Fixed contraction strength for the first reservoir. Must satisfy
            ``eps < lam_0 < 1-eps``.
        seed : float, default=12345
            RNG seed for deterministic sampling.
        eps : float, default=1e-8
            Margin to avoid sampling exactly 0 or 1 for λ.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(R, P)`` where ``P = |J| + |h_x| + |h_z| + 1`` is the
            number of free reservoir parameters. Columns follow the *metadata order*
            expected by downstream runners:

            ``param_order = list(J) + list(h_x) + list(h_z) + [lam]``.

        Raises
        ------
        AssertionError
            If inputs violate basic constraints (e.g. `num_reservoirs` < 1).
        """
        R = num_reservoirs
        assert R >= 1, f"num_reservoirs must be >= 1, got {num_reservoirs}"
        assert eps < lam_0 < 1 - eps, f"lam_0 must be in (eps, 1-eps); got lam_0={lam_0}, eps={eps}"

        qc = CircuitFactory.createIsingRingCircuitSWAP(cfg=cfg, angle_positioning=angle_positioning)
        rng = np.random.default_rng(seed)

        rzz_params = qc.metadata["J"]
        rx_params = qc.metadata["h_x"]
        rz_params = qc.metadata["h_z"]
        lam_param = qc.metadata["lam"]

        rzz_values = rng.uniform(-np.pi, np.pi, size=(R, len(rzz_params)))
        rx_values = rng.uniform(-np.pi, np.pi, size=(R, len(rx_params)))
        rz_values = rng.uniform(-np.pi, np.pi, size=(R, len(rz_params)))

        lam_values = np.empty(R, dtype=float)
        lam_values[0] = float(lam_0)
        if R > 1:
            lam_values[1:] = rng.uniform(eps, 1 - eps, size=R - 1)

        param_order = list(rzz_params) + list(rx_params) + list(rz_params) + [lam_param]
        P = len(param_order)
        param_values = np.empty((R, P), dtype=float)

        for r in range(R):
            bind = {}
            bind.update(dict(zip(rzz_params, rzz_values[r])))
            bind.update(dict(zip(rx_params, rx_values[r])))
            bind.update(dict(zip(rz_params, rz_values[r])))
            bind[lam_param] = float(lam_values[r])
            param_values[r, :] = [bind[p] for p in param_order]

        return param_values

    @staticmethod
    def create_pub_reservoirs_IsingRingSWAP(
        cfg: RingQRConfig,
        angle_positioning: callable,
        x_window: np.ndarray,
        num_reservoirs: int,
        param_values: np.ndarray,
    ):
        """Create a single PUB for a given window using shared reservoir parameters.

        A "PUB" is a tuple ``(qc, param_values)`` where:

        - ``qc`` is the *window circuit* with input parameters already bound.
        - ``param_values`` is a 2D array of shape ``(R, P)`` with ``R=num_reservoirs``
          different reservoir parameterizations to be evaluated on the **same** circuit.

        Parameters
        ----------
        cfg : RingQRConfig
            Reservoir configuration.
        angle_positioning : callable
            Injection angle map.
        x_window : numpy.ndarray
            Window of shape ``(w, d)``.
        num_reservoirs : int
            Number of reservoir parameterizations ``R``. Used for basic sanity checks.
        param_values : numpy.ndarray
            Array of shape ``(R, P)`` produced by :meth:`set_reservoirs_parameterizationSWAP`.

        Returns
        -------
        tuple
            ``(qc, param_values)`` ready for runners such as
            :class:`src.qrc.run.circuit_run.ExactAerCircuitsRunner`.
        """
        R = num_reservoirs
        assert R >= 1, f"num_reservoirs must be >= 1, got {num_reservoirs}"

        qc = CircuitFactory.instantiateFullIsingRingEvolution(
            cfg=cfg,
            angle_positioning=angle_positioning,
            x_window=x_window,
        )

        return (qc, param_values)

    @staticmethod
    def create_pubs_dataset_reservoirs_IsingRingSWAP(
        cfg: RingQRConfig,
        angle_positioning: callable,
        X: np.ndarray,
        lam_0: float,
        num_reservoirs: int,
        seed: int = 12354,
        eps: float = 1e-8,
    ):
        """Build a PUBS dataset for a window dataset ``X``.

        This function samples a *single shared* set of reservoir parameters
        (``param_values``) and reuses it for each input window in ``X``. This is
        often desirable when benchmarking different readouts on the same reservoir
        ensemble.

        Parameters
        ----------
        cfg : RingQRConfig
            Reservoir configuration.
        angle_positioning : callable
            Injection angle map.
        X : numpy.ndarray
            Input windows of shape ``(N, w, d)``.
        lam_0 : float
            Fixed contraction strength for the first reservoir.
        num_reservoirs : int
            Number of reservoir parameterizations ``R``.
        seed : int, default=12354
            Seed used to sample reservoir parameters (shared across all windows).
        eps : float, default=1e-8
            Margin to avoid sampling λ extremely close to 0 or 1.

        Returns
        -------
        list
            List of length ``N`` where each element is a PUB ``(qc, param_values)``.

        Raises
        ------
        AssertionError
            If `X` does not have shape ``(N, w, d)``.
        """
        assert X.ndim == 3, f"X should be an array of shape (N, w, d), got X.shape={X.shape}"

        param_values = CircuitFactory.set_reservoirs_parameterizationSWAP(
            cfg=cfg,
            angle_positioning=angle_positioning,
            num_reservoirs=num_reservoirs,
            lam_0=lam_0,
            seed=seed,
            eps=eps,
        )

        pubs = []
        for x_window in X:
            pub = CircuitFactory.create_pub_reservoirs_IsingRingSWAP(
                cfg=cfg,
                angle_positioning=angle_positioning,
                x_window=x_window,
                param_values=param_values,
                num_reservoirs=num_reservoirs,
            )
            pubs.append(pub)

        return pubs
