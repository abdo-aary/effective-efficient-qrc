# `src.qrc.circuits` package overview

This package contains the **circuit-building utilities** used to turn a dataset of input windows
`X ∈ R^{N×w×d}` into a list of **PUBs** (parameterized quantum circuit + parameter values) that can be
executed by a runner (e.g., an Aer runner) and then featurized by downstream components.

It is intentionally lightweight and focused on:

- **Config objects** describing qubit topology and Johnson–Lindenstrauss (JL) projections.
- **Angle positioning** utilities (data-to-rotation maps).
- A **CircuitFactory** that constructs the paper’s *SMC / Ising ring* circuits and packages them into PUBs.
- Utility helpers for **observable generation** and **shot-count heuristics**.

---

## Big picture

### Data flow (high level)

1. **Configure** a reservoir topology + JL projection: `RingQRConfig(input_dim=d, num_qubits=n)`.
2. Choose a **data injection map**: `angle_positioning`.
3. Use **CircuitFactory** to:
   - build a *single-step* SMC circuit (data injection → Ising unitary → contraction),
   - “unroll” it over a window `x_window ∈ R^{w×d}` by repeatedly binding `z_t = x_t @ Π`,
   - sample **R** reservoir parameterizations `(J, h_x, h_z, λ)` shared across all windows,
   - return a single **template PUB**: ``[(qc_template, vals)]`` where ``vals`` has shape ``(N, R, P_total)``.
4. Downstream (outside this package):
   - a **runner** executes PUBs and returns `Results`,
   - a **feature map retriever** produces features `Φ ∈ R^{N×D}` using chosen observables.

### High-level class diagram

```mermaid
classDiagram
direction LR


class BaseQRConfig {
  <<abstract>>
  +QRTopologyConfig topology
  +int input_dim
  +int num_qubits
  +ndarray projection
  +int seed
}

class RingQRConfig {
  +RingQRConfig(input_dim, num_qubits, seed)
}

class QRTopologyConfig {
  +int num_qubits
  +Sequence~(int,int)~ edges
}

class CircuitFactory {
  <<static>>
  +createIsingRingCircuitSWAP(cfg, angle_positioning) QuantumCircuit
  +instantiateFullIsingRingEvolutionTemplate(cfg, angle_positioning, w) (QuantumCircuit, List~ParameterVector~)
  +set_reservoirs_parameterizationSWAP(cfg, angle_positioning, num_reservoirs, lam_0, seed, eps) ndarray
  +create_pubs_dataset_reservoirs_IsingRingSWAP(cfg, angle_positioning, X, num_reservoirs, lam_0, seed, eps) List~Pub~
}

class Pub {
  <<tuple>>
  +QuantumCircuit qc
  +ndarray param_values  # shape (N,R,P_total)
}

class QuantumCircuit {
  +metadata : dict
}

class QcMetadataContract {
  +J : ParameterVector
  +h_x : ParameterVector
  +h_z : ParameterVector
  +lam : Parameter
  +z : ParameterVector
  +z_steps : List~ParameterVector~
  +param_order : List~Parameter~
}


class angle_positioning {
  <<function>>
}

class generate_k_local_paulis {
  <<function>>
}

class get_theoretical_shots {
  <<function>>
}

class projectionPi {
  <<data>>
  +ndarray Π
}
RingQRConfig --> projectionPi : samples JL matrix

BaseQRConfig <|-- RingQRConfig
RingQRConfig --> QRTopologyConfig : builds
RingQRConfig --> projectionPi : samples JL matrix
CircuitFactory --> RingQRConfig : consumes
CircuitFactory --> angle_positioning : uses
CircuitFactory --> QuantumCircuit : produces
CircuitFactory --> Pub : packages
QuantumCircuit --> QcMetadataContract : metadata keys
```

---

## Modules and main objects

## `configs.py`

### `sample_jl_projections_gaussian(input_dim, zeta, seed) -> np.ndarray`
Samples a JL projection matrix `Π ∈ R^{d×ζ}` with entries `~ N(0, 1/ζ)`.
Used as `z = x @ Π` to map a `d`-dimensional input into an injection space of size `ζ`
(typically `ζ = num_qubits`).

### `QRTopologyConfig`
Dataclass holding:
- `num_qubits`: number of reservoir qubits `n`,
- `edges`: list/tuple of two-qubit interaction edges `(q1, q2)`.

### `BaseQRConfig`
Abstract base class describing what a circuit config must expose:
- topology, input dimension, qubit count, JL projection, seed.

### `RingQRConfig(BaseQRConfig)`
Concrete config for a **ring topology** reservoir:
- Builds a ring interaction graph via `ring_topology(num_qubits)`,
- Samples the JL projection `Π` with `zeta=num_qubits`.

### `ring_topology(num_qubits) -> QRTopologyConfig`
Creates edges:
- `(0,1), (1,2), …, (n-2,n-1)` and (if `n>2`) closes the ring with `(n-1,0)`.

---

## `utils.py`

### Angle positioning (data → rotation angles)

Both functions accept `z` as a Qiskit `ParameterVector` and return a list of angle expressions.
We use by default the following angle positionning utiliy.
- `angle_positioning_tanh(z, scale=π(1-1e-6))`  
  Elementwise: `θ_j(z) = scale · tanh(z_j)` implemented using `exp()` only
  (compatible with Qiskit’s `ParameterExpression`).

### Observables generation

- `generate_k_local_paulis(locality, num_qubits) -> List[SparsePauliOp]`  
  Generates all Pauli strings on `num_qubits` with locality ≤ `k`
  (i.e., up to `k` non-identity letters).  
  **Convention**: qubit 0 corresponds to the **rightmost** character in the Pauli string.

### Shot count heuristic for classical shadows

- `get_theoretical_shots(eps, delta, locality, num_data_pts, num_obs, num_draws) -> int`  
  Returns a recommended number of snapshots ensuring (heuristically) a uniform error bound
  across all draws/data points/observables.

---

## `circuit_factory.py`

### Core idea: PUBs

A **PUB** is a tuple ``(qc, param_values)`` where:

- ``qc`` is a `QuantumCircuit` produced by the factory,
- ``param_values`` is an array of numeric parameter values aligned with a deterministic
  parameter order stored in ``qc.metadata["param_order"]``.

In the **dataset** entry point (`create_pubs_dataset_reservoirs_IsingRingSWAP`), we return a
*single* **template PUB**:

- ``qc`` is a **window template circuit** (w steps, unbound inputs),
- ``param_values`` has shape ``(N, R, P_total)``:
  - ``N``: number of windows,
  - ``R``: number of reservoirs,
  - ``P_total = (w·n) + (|J| + |h_x| + |h_z| + 1)``.

The runner should bind parameters for a given (window i, reservoir r) using:

``dict(zip(qc.metadata["param_order"], param_values[i, r]))``.

### `CircuitFactory.createIsingRingCircuitSWAP(cfg, angle_positioning)`
Builds a **deterministic** single-step contraction implementing

`E_λ(ρ) = λ ρ + (1-λ) |+⟩⟨+|^{⊗n}`

via a Stinespring-like dilation:
- reservoir qubits + `n` aux qubits initialized to `|+⟩^{⊗n}` + a coin qubit controlling swaps,
- reset of environment qubits to trace them out.

### `CircuitFactory.instantiateFullIsingRingEvolutionTemplate(cfg, angle_positioning, w)`
Builds a *single* **parameterized window circuit** of length `w` where:
- each step `t` uses its own injected input vector `z_t` (unbound parameters),
- reservoir parameters `(J, h_x, h_z, λ)` are shared across steps.

Returns `(qc_template, z_steps)` and stores shared parameters in `qc_template.metadata`
as well as `qc_template.metadata["z_steps"]`.


### `CircuitFactory.set_reservoirs_parameterizationSWAP(cfg, angle_positioning, num_reservoirs, lam_0, seed, eps)`
Samples **R** reservoir parameterizations (shared across all windows):
- `J`, `h_x`, `h_z` drawn i.i.d. Uniform(`[-π, π]`),
- `λ_0 = lam_0` for reservoir 0,
- remaining `λ_r ~ Uniform(eps, 1-eps)`.

Returns `param_values ∈ R^{R×P}` in the expected metadata-driven order:
`[J..., h_x..., h_z..., lam]`.

### `CircuitFactory.create_pubs_dataset_reservoirs_IsingRingSWAP(...)`
Dataset entry point.

- Input: `X ∈ R^{N×w×d}`
- Output: `pubs = [(qc_template, vals)]` (length **1** list)

where:
- `qc_template` is a window template circuit (w steps, unbound injected inputs),
- `vals` has shape `(N, R, P_total)` and follows the column order in
  `qc_template.metadata["param_order"]`.

Internally, this function samples the reservoir parameter table by calling
`set_reservoirs_parameterizationSWAP(...)` (so you no longer pass a
`parameters_reservoirs` argument).

---

## Minimal usage snippet

```python
import numpy as np
from src.qrc.circuits.qrc_configs import RingQRConfig
from src.qrc.circuits.utils import angle_positioning_tanh
from src.qrc.circuits.circuit_factory import CircuitFactory

cfg = RingQRConfig(input_dim=3, num_qubits=3, seed=0)

# X has shape (N, w, d)
X = np.random.default_rng(0).normal(size=(10, 20, 3))

pubs = CircuitFactory.create_pubs_dataset_reservoirs_IsingRingSWAP(
    qrc_cfg=cfg,
    angle_positioning=angle_positioning_tanh,
    X=X,
    num_reservoirs=8,
    lam_0=0.2,
    seed=0,
)

qc_template, vals = pubs[0]  # vals.shape == (N, R, P_total)

# Example: bind parameters for window i and reservoir r
i, r = 0, 0
bind = dict(zip(qc_template.metadata["param_order"], vals[i, r]))
qc_bound = qc_template.assign_parameters(bind, inplace=False)
```

---

## Conventions & shapes

- `X`: input windows, shape `(N, w, d)`.
- `x_window`: a single window, shape `(w, d)`.
- `Π = cfg.projection`: JL projection, shape `(d, n)` where `n = cfg.num_qubits`.
- `z_t = x_t @ Π`: injected vector, shape `(n,)`.
- `vals`: PUB parameter tensor, shape `(N, R, P_total)` with:
  - `P_total = (w·n) + (|J| + |h_x| + |h_z| + 1)`.

---

## Common pitfalls

- **Observable label length must match `num_qubits`** downstream (e.g., `"ZII"` for `n=3`).
- If you build circuits outside `CircuitFactory`, they **won’t** have the metadata keys
  required by runners that rely on `qc.metadata["J","h_x","h_z","lam"]`.
