"""The replay path: verify, resolve, rules, respond. No model calls."""

from __future__ import annotations

from core import PatientSubmission
from observations import Observations
from resolve import resolve_anticoagulants, resolve_vitals
from rules import TriageResult, notes_unread, respond, run_checks
from verify import verify


def in_fixed_order(items: list) -> list:
    return sorted(items, key=lambda item: item.model_dump_json())


def evaluate(
    submission: dict[str, object] | PatientSubmission,
    observations: Observations | None,
    extraction_error: str | None = None,
) -> TriageResult:
    submission = PatientSubmission.model_validate(submission)
    observations = observations or Observations(**{name: [] for name in Observations.model_fields})
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
    drugs = resolve_anticoagulants(submission, verified)
    checks = run_checks(submission, verified, vitals, drugs)
    if extraction_error:
        checks = [notes_unread(check, extraction_error) for check in checks]
    return respond(checks, verified.dropped)
