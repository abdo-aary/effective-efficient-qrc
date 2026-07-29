"""Canonical QuaRK feature ordering."""

ORDERING_VERSION = "reservoir-major-observable-minor/v1"


def flat_feature_index(reservoir: int, observable: int, num_observables: int) -> int:
    return int(reservoir) * int(num_observables) + int(observable)
