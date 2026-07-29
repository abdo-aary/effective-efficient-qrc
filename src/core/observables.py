"""Backend-neutral observable descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

_PAULI_CHARS = frozenset("IXYZ")


@dataclass(frozen=True)
class ObservableSet:
    """Ordered single-term Pauli observables.

    Labels use Qiskit's conventional textual order: the rightmost character
    acts on qubit zero.  The labels themselves are framework independent.
    """

    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        labels = tuple(str(label).upper() for label in self.labels)
        if not labels:
            raise ValueError("ObservableSet must contain at least one observable.")
        width = len(labels[0])
        if width < 1:
            raise ValueError("Observable labels must be non-empty.")
        for label in labels:
            if len(label) != width:
                raise ValueError("All observable labels must have the same width.")
            invalid = set(label) - _PAULI_CHARS
            if invalid:
                raise ValueError(f"Invalid Pauli characters {sorted(invalid)} in {label!r}.")
        if len(set(labels)) != len(labels):
            raise ValueError("Observable labels must be unique to preserve feature identity.")
        object.__setattr__(self, "labels", labels)

    @property
    def num_qubits(self) -> int:
        return len(self.labels[0])

    @property
    def size(self) -> int:
        return len(self.labels)

    @property
    def localities(self) -> tuple[int, ...]:
        return tuple(sum(char != "I" for char in label) for label in self.labels)

    @classmethod
    def local_paulis(cls, *, num_qubits: int, locality: int) -> "ObservableSet":
        """Return all nonidentity Pauli words of weight at most ``locality``."""

        n = int(num_qubits)
        k = min(int(locality), n)
        if n < 1:
            raise ValueError("num_qubits must be positive.")
        if k < 1:
            raise ValueError("locality must be positive.")
        labels: list[str] = []
        for weight in range(1, k + 1):
            for support in combinations(range(n), weight):
                for letters in product(("X", "Y", "Z"), repeat=weight):
                    chars = ["I"] * n
                    for qubit, letter in zip(support, letters):
                        chars[n - 1 - qubit] = letter
                    labels.append("".join(chars))
        return cls(tuple(labels))

    @classmethod
    def from_qiskit(cls, observables: object) -> "ObservableSet":
        """Boundary helper accepting single-term Qiskit Pauli-like objects.

        Qiskit is intentionally not imported here.  Objects are inspected by
        their public ``paulis``/``to_label`` interfaces.
        """

        labels: list[str] = []
        for observable in observables:  # type: ignore[union-attr]
            paulis = getattr(observable, "paulis", None)
            if paulis is None or len(paulis) != 1:
                raise ValueError("Only single-term Pauli observables are supported.")
            labels.append(str(paulis[0].to_label()))
        return cls(tuple(labels))
