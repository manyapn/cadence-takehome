"""The replay path: verify, resolve, rules, respond. No model calls."""

from __future__ import annotations

from core import PatientSubmission
from observations import Observations
from resolve import resolve_vitals
from rules import TriageResult, respond, run_checks
from verify import verify


def in_fixed_order(items: list) -> list:
    return sorted(items, key=lambda item: item.model_dump_json())


def evaluate(submission: dict[str, object] | PatientSubmission, observations: Observations) -> TriageResult:
    submission = PatientSubmission.model_validate(submission)
    # Sorting the inputs keeps the output independent of the order they arrive in.
    submission = submission.model_copy(
        update={
            name: in_fixed_order(getattr(submission, name))
            for name in ("vitals", "labs", "medications", "documents")
        }
    )
    observations = Observations(
        **{name: in_fixed_order(getattr(observations, name)) for name in Observations.model_fields}
    )

    verified = verify(submission, observations)
    vitals = resolve_vitals(submission, verified.readings)
    return respond(run_checks(submission, verified, vitals))
