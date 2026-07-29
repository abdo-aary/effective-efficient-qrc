"""Serializable IBM Runtime job handles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IBMJobHandle:
    job_id: str
    backend_name: str
    program_fingerprint: str
    estimator_kind: str
    submitted_at: str
    group_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "backend_name": self.backend_name,
            "program_fingerprint": self.program_fingerprint,
            "estimator_kind": self.estimator_kind,
            "submitted_at": self.submitted_at,
            "group_count": self.group_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "IBMJobHandle":
        return cls(
            job_id=str(payload["job_id"]),
            backend_name=str(payload["backend_name"]),
            program_fingerprint=str(payload["program_fingerprint"]),
            estimator_kind=str(payload["estimator_kind"]),
            submitted_at=str(payload["submitted_at"]),
            group_count=int(payload["group_count"]),
        )
