from __future__ import annotations

import numpy as np

from src.data.label.context import TeacherContext
from src.data.label.functionals import (
    DelayedRecallFunctional,
    NormalizedExpMemoryFunctional,
    OneStepFutureFunctional,
    SparseCrossLagFunctional,
    U3,
    V3,
    VolterraFunctional,
)


def _context(future):
    return TeacherContext(
        prediction_origins=np.asarray([24]),
        future_indices=np.asarray([25]),
        future_observations=np.asarray([future], dtype=float),
    )


def test_future_is_strictly_outside_input_window():
    window = np.zeros((25, 3))
    first = OneStepFutureFunctional().evaluate(
        window, index=0, context=_context([1.0, 0.0, 0.0])
    )
    second = OneStepFutureFunctional().evaluate(
        window, index=0, context=_context([-1.0, 0.0, 0.0])
    )
    assert first == 1.0 / 3.0
    assert second == -1.0 / 3.0
    np.testing.assert_array_equal(window, np.zeros_like(window))


def test_memory_delay_cross_and_volterra_match_direct_formulas():
    window = np.linspace(-0.8, 0.8, 75).reshape(25, 3)
    context = _context([0.0, 0.0, 0.0])
    weights = 0.8 ** np.arange(25)
    reverse = window[::-1]
    A = np.sum(weights)
    Lu = (reverse @ U3) @ weights
    Lv = (reverse @ V3) @ weights
    assert np.isclose(
        NormalizedExpMemoryFunctional().evaluate(window, index=0, context=context),
        Lu / A,
    )
    assert np.isclose(
        VolterraFunctional().evaluate(window, index=0, context=context),
        (Lu + 0.5 * Lv**2) / (A + 0.5 * A**2),
    )
    assert np.isclose(
        DelayedRecallFunctional(5).evaluate(window, index=0, context=context),
        window[-6] @ U3,
    )
    assert np.isclose(
        SparseCrossLagFunctional().evaluate(window, index=0, context=context),
        (window[-1] @ U3) * (window[-16] @ V3),
    )
