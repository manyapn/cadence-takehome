from __future__ import annotations

import pytest

from cases import load_case
from core import PatientSubmission
from observations import AnticoagulantMention, AnticoagulationPlan, PlanSide, VerifiedObservations
from pipeline import evaluate
from resolve import resolve_anticoagulants
from rules import check_anticoagulation
from verify import verify


@pytest.mark.parametrize("number", [18, 20, 33, 19, 25, 27, 3])
def test_case_matches_its_label(number: int) -> None:
    submission, label, observations = load_case(number)

    assert verify(PatientSubmission.model_validate(submission), observations).dropped == []
    result = evaluate(submission, observations)
    assert result.decision == label["decision"]
    assert sorted({issue.category for issue in result.issues}) == sorted(label["categories"])


def test_case_18_fails_because_the_plan_is_for_a_drug_no_longer_taken() -> None:
    submission, _, observations = load_case(18)

    (issue,) = evaluate(submission, observations).issues

    assert issue.category == "ANTICOAGULATION_MANAGEMENT"
    assert "Patient takes apixaban and there is no documented perioperative plan for it." in issue.finding
    assert "The plan dated 2026-02-06 is for warfarin, which the patient no longer takes." in issue.finding
    assert any(item.value == "warfarin after the procedure: instruction" for item in issue.evidence)


def test_case_18_is_ready_if_the_patient_were_still_on_warfarin() -> None:
    submission, _, observations = load_case(18)
    submission["medications"] = [{"name": "warfarin 5 mg PO daily", "active": True}]
    observations.anticoagulant_mentions = [
        mention for mention in observations.anticoagulant_mentions if mention.status == "taking" and mention.generic_name == "warfarin"
    ]

    assert evaluate(submission, observations).decision == "READY"


def test_case_20_names_the_deferred_side() -> None:
    submission, _, observations = load_case(20)

    (issue,) = evaluate(submission, observations).issues

    assert "before the procedure is instruction, after the procedure is deferred" in issue.finding
    assert any(item.quote == "Post-operative anticoagulation will be addressed after surgery." for item in issue.evidence)


def test_case_33_is_unknown_not_a_failure() -> None:
    submission, _, observations = load_case(33)

    (issue,) = evaluate(submission, observations).issues

    assert issue.category == "MISSING_REQUIRED_DATA"
    assert "not confirmed whether the patient currently takes warfarin" in issue.finding


SUBMISSION = PatientSubmission.model_validate(
    {
        "medications": [{"name": "Xarelto 20 mg daily", "active": True}],
        "documents": [
            {"doc_id": "older", "date": "2026-03-01", "text": ""},
            {"doc_id": "newer", "date": "2026-03-20", "text": ""},
        ],
    }
)
BOTH_SIDES = {"preop": PlanSide(label="instruction", quote="stop"), "postop": PlanSide(label="instruction", quote="restart")}


def mention(drug: str, generic: str, status: str, doc_id: str = "newer") -> AnticoagulantMention:
    return AnticoagulantMention(doc_id=doc_id, drug_as_written=drug, generic_name=generic, status=status, quote=drug)


def rule_3(submission: PatientSubmission, **verified: list) -> str:
    drugs = resolve_anticoagulants(submission, VerifiedObservations(**verified))
    return check_anticoagulation(submission, drugs).status


def test_structured_anticoagulant_without_a_plan_fails() -> None:
    assert rule_3(SUBMISSION) == "fail"


def test_newest_note_decides_between_notes() -> None:
    no_list = SUBMISSION.model_copy(update={"medications": []})
    stopped_later = [mention("warfarin", "warfarin", "taking", "older"), mention("Coumadin", "warfarin", "discontinued")]
    started_later = [mention("warfarin", "warfarin", "discontinued", "older"), mention("Coumadin", "warfarin", "taking")]

    assert rule_3(no_list, anticoagulant_mentions=stopped_later) == "not_applicable"
    assert rule_3(no_list, anticoagulant_mentions=started_later) == "fail"


def test_medication_list_and_newest_note_disagreeing_is_unknown() -> None:
    assert rule_3(SUBMISSION, anticoagulant_mentions=[mention("Xarelto", "rivaroxaban", "discontinued")]) == "unknown"


def test_antiplatelet_reported_as_an_anticoagulant_is_ignored() -> None:
    no_list = SUBMISSION.model_copy(update={"medications": []})
    mentions = [mention("ASA 81 mg", "aspirin", "taking"), mention("Plavix", "clopidogrel", "taking")]

    assert rule_3(no_list, anticoagulant_mentions=mentions) == "not_applicable"


def test_drug_outside_the_table_is_unknown_unless_a_plan_covers_it() -> None:
    no_list = SUBMISSION.model_copy(update={"medications": []})
    mentions = [mention("Innohep", "tinzaparin", "taking")]
    plan = AnticoagulationPlan(doc_id="newer", drug_as_written="tinzaparin", **BOTH_SIDES)

    assert rule_3(no_list, anticoagulant_mentions=mentions) == "unknown"
    assert rule_3(no_list, anticoagulant_mentions=mentions, anticoagulation_plans=[plan]) == "pass"
    stopped = [mention("Innohep", "tinzaparin", "discontinued")]
    assert rule_3(no_list, anticoagulant_mentions=stopped) == "not_applicable"


def test_plan_must_cover_both_sides_in_one_document() -> None:
    before_only = AnticoagulationPlan(
        doc_id="older", drug_as_written="Xarelto", preop=BOTH_SIDES["preop"], postop=PlanSide(label="absent", quote=None)
    )
    complete = AnticoagulationPlan(doc_id="newer", drug_as_written="rivaroxaban", **BOTH_SIDES)

    assert rule_3(SUBMISSION, anticoagulation_plans=[before_only]) == "fail"
    assert rule_3(SUBMISSION, anticoagulation_plans=[before_only, complete]) == "pass"


def test_dropped_repeat_of_a_drug_already_known_to_be_taken_changes_nothing() -> None:
    submission, _, observations = load_case(3)
    repeat = observations.anticoagulant_mentions[0].model_copy(update={"quote": "Eliqius 5 mg PO BID"})
    observations.anticoagulant_mentions.append(repeat)

    result = evaluate(submission, observations)

    assert len(result.unverified) == 1
    assert result.decision == "READY"


def test_dropped_mention_of_a_drug_not_otherwise_known_leaves_the_check_open() -> None:
    submission, _, observations = load_case(3)
    other = observations.anticoagulant_mentions[0].model_copy(
        update={"drug_as_written": "warfarin", "generic_name": "warfarin", "quote": "also on warfarin"}
    )
    observations.anticoagulant_mentions.append(other)

    result = evaluate(submission, observations)

    assert next(check for check in result.checks if check.name == "anticoagulation_plan").status == "unknown"
    assert result.decision == "NEEDS_FOLLOW_UP"
