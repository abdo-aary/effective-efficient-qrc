import numpy as np
from qiskit.circuit import ParameterVector


def angle_positioning_linear(
        z: ParameterVector,
        *,
        scale: float = np.pi * (1.0 - 1e-6),  # keep strict (-pi, pi) if |z_j|<1
):
    """
    Elementwise θ(z_j) = scale * z_j.

    Properties (for each coordinate):
      - continuous
      - strictly monotone increasing if scale>0
      - injective
      - maps (-1,1) -> (-scale, scale) ⊂ (-pi, pi) if scale < pi
    """
    return [scale * z[j] for j in range(len(z))]
