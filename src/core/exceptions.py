"""Typed errors raised by the backend-neutral QuaRK API."""


class QuaRKError(Exception):
    """Base class for public QuaRK errors."""


class CapabilityError(QuaRKError):
    """A backend cannot satisfy a requested execution capability."""


class UnsupportedEstimatorError(CapabilityError):
    """The selected backend does not implement the requested estimator."""


class BackendDependencyError(QuaRKError, ImportError):
    """An optional backend dependency is unavailable."""


class CompilationError(QuaRKError):
    """A backend could not compile a valid feature-extraction plan."""


class ExecutionError(QuaRKError):
    """A compiled plan failed during execution or result decoding."""
