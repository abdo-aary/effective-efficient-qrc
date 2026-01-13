"""
# Implement a factory that returns a single parametrized big quantum circuit that contains R blocks of n qubits
# each. Each block must implement in a parametrized way the data injection followed by Ising unitary W_r, then
# contraction.
"""
from __future__ import annotations

from typing import List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter

from src.configs import RingQRConfig


class CircuitFactory:
    """Factory for building SMC tests_circuits as in the paper."""

    @staticmethod
    def createIsingRingCircuitDynamic(
            cfg: RingQRConfig,
            angle_positioning: callable,
            method: str = "density_matrix",
    ) -> QuantumCircuit:
        """
        Build the per-reservoir circuit (one circuit per reservoir), each implementing:
          data injection -> Ising unitary W_r -> contraction E_{lambda_r}.
        """
        if method != "density_matrix":
            raise NotImplementedError(
                f"Only method='density_matrix' is supported here; got {method}."
            )

        n = cfg.num_qubits
        topology = cfg.topology

        # Input (projected) parameters z in R^d (d = cfg.input_dim)
        z = ParameterVector("z", cfg.num_qubits)

        # Shared injection angles (length should be n)
        angles = angle_positioning(z)

        zz_params = ParameterVector(f"J", len(topology.edges))
        rx_params = ParameterVector(f"h_x", n)
        rz_params = ParameterVector(f"h_z", n)
        lam_param = Parameter("lam")

        # +1 coin qubit (constant overhead), +1 classical bit for conditional reset
        qc = QuantumCircuit(n + 1, 1, name=f"SMC")
        qubits = list(range(n))
        coin = n  # last qubit
        cbit = qc.clbits[0]

        # -------------------------
        # 1) Data injection: Πx -> z then Ry(theta(z_j)) on each qubit
        # -------------------------
        for i in range(n):
            qc.ry(angles[i], qubits[i])

        # -------------------------
        # 2) Ising unitary W_r
        #    Product over edges: Rzz(J_{ab})
        #    Then local fields: Rz(h_z), Rx(h_x)
        # -------------------------
        for e, (q1, q2) in enumerate(topology.edges):
            qc.rzz(zz_params[e], q1, q2)

        for i in range(n):
            qc.rz(rz_params[i], qubits[i])
        for i in range(n):
            qc.rx(rx_params[i], qubits[i])

        # -------------------------
        # 3) Contraction E_{lambda_r} via coin + conditional reset-to-|+>^{⊗n}
        #
        # Coin: P(coin=1)=lambda -> do nothing
        #       P(coin=0)=1-lambda -> reset qubits to |+>^{⊗n}
        # -------------------------
        # lam = float(reservoirs[0].dep_strength)  # lambda_r from config

        qc.reset(coin)
        theta = 2 * (lam_param ** 0.5).arcsin()  # Ry(theta)|0> => P(|1>) = lam
        qc.ry(theta, coin)

        # measure coin into the (single) classical bit
        qc.measure(coin, cbit)

        # If coin == 0: reset each qubits qubit to |0>, then H -> |+>
        with qc.if_test((cbit, 0)):
            qc.reset(qubits)
            qc.h(qubits)

        # Store the parameter vectors as a meta data
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
        """"
        Deterministic implementation of data injection -> Ising unitary W -> contraction E_lambda
        where
            E_lambda(rho) = lambda * rho + (1-lambda) * |+><+|^{⊗n}.
        Uses a Stinespring dilation with n ancillas + 1 coin qubit, then resets env.
        """

        n = cfg.num_qubits
        topology = cfg.topology

        # Projected input z lives in R^n (since cfg.projection maps R^d -> R^n)
        z = ParameterVector("z", n)
        angles = angle_positioning(z)  # length n with angle_positioning_linear(z)

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

        # -------------------------
        # 1) Data injection on reservoir
        # -------------------------
        for i in range(n):
            qc.ry(angles[i], res[i])

        # -------------------------
        # 2) Ising unitary W on reservoir
        # -------------------------
        for e, (q1, q2) in enumerate(topology.edges):
            qc.rzz(zz_params[e], q1, q2)

        for i in range(n):
            qc.rz(rz_params[i], res[i])
        for i in range(n):
            qc.rx(rx_params[i], res[i])

        # -------------------------
        # 3) Deterministic contraction E_lambda via unitary dilation + env reset
        #
        # Prepare aux in |+>^{⊗n}.
        # Prepare coin so that P(coin=1) = 1 - lam.
        # If coin=1: swap(res, aux) -> reservoir becomes |+>^{⊗n}.
        # Then reset coin+aux to trace out env => exact mixture on reservoir.
        # -------------------------

        # ensure fresh environment each step (important when composing steps)
        qc.reset(aux)
        qc.reset(coin)

        # aux := |+>^{⊗n}
        qc.h(aux)

        # coin: Ry(theta)|0> with sin^2(theta/2) = 1 - lam  => theta = 2 * asin(sqrt(1-lam))
        theta = 2 * ((1 - lam_param) ** 0.5).arcsin()
        qc.ry(theta, coin)

        # controlled swap all qubits
        for i in range(n):
            qc.cswap(coin, res[i], aux[i])

        # trace out environment deterministically
        qc.reset(coin)
        qc.reset(aux)

        # metadata (same keys as your original method)
        qc.metadata["z"] = z
        qc.metadata["J"] = zz_params
        qc.metadata["h_x"] = rx_params
        qc.metadata["h_z"] = rz_params
        qc.metadata["lam"] = lam_param

        return qc

    @staticmethod
    def instantiateFullIsingRingEvolution(cfg: RingQRConfig,
                                          angle_positioning: callable,
                                          x_window: np.ndarray, ) -> QuantumCircuit:
        """
        :param cfg:
        :param angle_positioning:
        :param x_window: this is a window of shape (w, d), where by convention x_window[0] <=> x_{-w + 1}.
        :return:
        """
        assert x_window.shape[-1] == cfg.input_dim, (f"Mismatch between the window dimension and input_dim in the "
                                                     f"provided config. Got x_window.shape[-1] = {x_window.shape[-1]}, "
                                                     f"and cfg.input_dim = {cfg.input_dim}!")

        assert x_window.ndim == 2, f"x_window should be an array of two dimensions, got x_window.ndim = {x_window.ndim}!"

        qc_reservoir = CircuitFactory.createIsingRingCircuitSWAP(cfg=cfg,
                                                                 angle_positioning=angle_positioning, )

        n = cfg.num_qubits
        qubits = list(range(n))
        z_params = qc_reservoir.metadata["z"]

        # Initialize to the state \psi = |+>^n
        qc = QuantumCircuit(qc_reservoir.num_qubits, name=f"QRC")
        qc.h(qubits[:n])

        # # Append at the beginning of the circuit such an initialization
        for t, x_t in enumerate(x_window):
            # Project into z using the projection
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
    def create_pub_reservoir_IsingRingSWAP(
            cfg: RingQRConfig,
            angle_positioning: callable,
            x_window: np.ndarray,
            lam_0: float,
            num_reservoirs: int,
            seed: int = 12354,
            eps: float = 1e-8,
    ):
        """
        Returns a single "pub" (qc, param_values) where param_values contains R different
        parameterizations (J, h_x, h_z, lam) for the SAME circuit qc corresponding to the
        provided input window x_window.

        - First reservoir uses lam_0
        - Remaining R-1 reservoirs use iid Uniform(eps, 1-eps)
        - All random draws are deterministic via `seed`.
        """
        # --- sanitize / determinism
        R = num_reservoirs
        assert R >= 1, f"num_reservoirs must be >= 1, got {num_reservoirs}"
        assert eps < lam_0 < 1 - eps, f"lam_0 must be in (eps, 1-eps); got lam_0={lam_0}, eps={eps}"

        rng = np.random.default_rng(seed)

        # Build the FULL window circuit (z already bound inside instantiateFullIsingRingEvolution)
        qc = CircuitFactory.instantiateFullIsingRingEvolution(
            cfg=cfg,
            angle_positioning=angle_positioning,
            x_window=x_window,
        )

        # Parameters that remain free in qc
        rzz_params = qc.metadata["J"]
        rx_params = qc.metadata["h_x"]
        rz_params = qc.metadata["h_z"]
        lam_param = qc.metadata["lam"]

        # Sample R different reservoir parameterizations
        rzz_values = rng.uniform(-np.pi, np.pi, size=(R, len(rzz_params)))
        rx_values = rng.uniform(-np.pi, np.pi, size=(R, len(rx_params)))
        rz_values = rng.uniform(-np.pi, np.pi, size=(R, len(rz_params)))

        lam_values = np.empty(R, dtype=float)
        lam_values[0] = float(lam_0)
        if R > 1:
            lam_values[1:] = rng.uniform(eps, 1 - eps, size=R - 1)

        # IMPORTANT: parameter values must follow the circuit's parameter order
        param_order = list(qc.parameters)  # Qiskit’s canonical order used by primitives
        P = len(param_order)

        param_values = np.empty((R, P), dtype=float)

        for r in range(R):
            bind = {}
            bind.update(dict(zip(rzz_params, rzz_values[r])))
            bind.update(dict(zip(rx_params, rx_values[r])))
            bind.update(dict(zip(rz_params, rz_values[r])))
            bind[lam_param] = float(lam_values[r])

            # row in qc.parameters order
            param_values[r, :] = [bind[p] for p in param_order]

        # Optional: store order + sampled lambdas for debugging/repro
        qc.metadata["param_order"] = param_order
        qc.metadata["lam_values"] = lam_values.tolist()

        pub = (qc, param_values)
        return pub
