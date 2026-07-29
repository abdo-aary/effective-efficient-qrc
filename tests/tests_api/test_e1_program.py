from __future__ import annotations

import numpy as np

from src.core.factories import make_protocol_program
from src.core.observables import ObservableSet
from src.core.seeds import SeedBundle


def test_reference_program_shape_order_and_seed_isolation():
    seeds = SeedBundle.from_root(1101)
    program = make_protocol_program(
        input_dim=3,
        num_qubits=5,
        num_reservoirs=3,
        locality=2,
        reset_rate=0.5,
        window_length=25,
        seeds=seeds,
    )
    assert program.observables.size == 105
    assert program.observables.labels[:6] == ("IIIIX", "IIIIY", "IIIIZ", "IIIXI", "IIIYI", "IIIZI")
    np.testing.assert_array_equal(program.reservoirs.reset_rates, [0.5, 0.5, 0.5])
    assert program.projection.matrix.shape == (3, 5)
    assert program.fingerprint() == make_protocol_program(
        input_dim=3,
        num_qubits=5,
        num_reservoirs=3,
        locality=2,
        reset_rate=0.5,
        window_length=25,
        seeds=SeedBundle.from_root(1101),
    ).fingerprint()
    assert (
        ObservableSet.local_paulis(num_qubits=5, locality=2).labels
        == program.observables.labels
    )


def test_seed_bundle_loads_v1_without_shifting_old_streams():
    new = SeedBundle.from_root(42)
    payload = new.to_dict()
    payload["spawn_keys"].pop("task_functionals")
    payload.pop("schema_version")
    restored = SeedBundle.from_dict(payload)
    for name in new.names[:-1]:
        assert restored.sequence(name).spawn_key == new.sequence(name).spawn_key
    assert (
        restored.sequence("task_functionals").spawn_key
        == new.sequence("task_functionals").spawn_key
    )
