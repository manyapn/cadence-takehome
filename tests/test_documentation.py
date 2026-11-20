from __future__ import annotations

from datetime import date, timedelta

import pytest

from cases import load_case
from core import PatientSubmission
from observations import HpDocument
from pipeline import evaluate
from rules import Check, TriageResult
from verify import verify


def check_named(result: TriageResult, name: str) -> Check:
    return next(check for check in result.checks if check.name == name)


def document(submission: dict[str, object], doc_type: str) -> dict[str, object]:
    return next(document for document in submission["documents"] if document["type"] == doc_type)


def categories(result: TriageResult) -> list[str]:
    return sorted({issue.category for issue in result.issues})


@pytest.mark.parametrize("number", [1, 13, 14, 7, 8, 24])
def test_case_matches_its_label(number: int) -> None:
    submission, label, observations = load_case(number)

    assert verify(PatientSubmission.model_validate(submission), observations).dropped == []
    result = evaluate(submission, observations)
    assert result.decision == label["decision"]
    assert categories(result) == sorted(label["categories"])


@pytest.mark.parametrize(
    ("days_before", "decision", "expected_categories"),
    [(0, "READY", []), (30, "READY", []), (31, "NEEDS_FOLLOW_UP", ["REQUIRED_DOCUMENTATION"])],
)
def test_hp_window_is_0_to_30_calendar_days(days_before: int, decision: str, expected_categories: list[str]) -> None:
    submission, _, observations = load_case(1)
    procedure_date = date.fromisoformat(submission["procedure"]["procedure_date"])
    document(submission, "History and Pyhsical")["date"] = (procedure_date - timedelta(days=days_before)).isoformat()

    result = evaluate(submission, observations)

    assert f"is {days_before} days before the procedure" in check_named(result, "history_and_physical").summary
    assert result.decision == decision
    assert categories(result) == expected_categories


def test_case_1_counts_the_preop_hp_filed_under_a_misspelled_type() -> None:
    submission, _, observations = load_case(1)

    hp = check_named(evaluate(submission, observations), "history_and_physical")

    assert hp.status == "pass"
    assert [item.reference for item in hp.evidence] == [document(submission, "History and Pyhsical")["doc_id"]]
    assert hp.summary == "H&P dated 2026-02-10 is 20 days before the procedure (limit 30)."


def test_case_1_wellness_note_alone_does_not_satisfy_the_hp_rule() -> None:
    submission, _, observations = load_case(1)
    observations.hp_documents = [hp for hp in observations.hp_documents if hp.kind == "wellness_or_other"]
    assert observations.hp_documents

    result = evaluate(submission, observations)

    assert check_named(result, "history_and_physical").status == "fail"
    assert result.decision == "NEEDS_FOLLOW_UP"
    assert categories(result) == ["REQUIRED_DOCUMENTATION"]


def test_wellness_note_inside_the_window_is_named_but_does_not_count() -> None:
    submission, _, observations = load_case(1)
    observations.hp_documents = [hp for hp in observations.hp_documents if hp.kind == "wellness_or_other"]
    document(submission, "History and Physical")["date"] = "2026-02-20"

    hp = check_named(evaluate(submission, observations), "history_and_physical")

    assert hp.status == "fail"
    assert "reads as a wellness or other visit, not a pre-operative evaluation" in hp.summary


def test_case_13_has_no_hp_and_lists_the_documents_checked() -> None:
    submission, _, observations = load_case(13)

    hp = check_named(evaluate(submission, observations), "history_and_physical")

    assert hp.status == "fail"
    assert hp.summary == "No pre-operative H&P found in 5 documents."
    assert sorted(hp.sources_checked) == sorted(f"{d['type']} ({d['date']})" for d in submission["documents"])


def test_case_14_hp_is_one_day_past_the_window() -> None:
    submission, _, observations = load_case(14)

    result = evaluate(submission, observations)

    hp = check_named(result, "history_and_physical")
    assert hp.status == "fail"
    assert hp.summary == "H&P dated 2026-02-12 is 31 days before the procedure (limit 30)."
    assert check_named(result, "surgical_consent").status == "pass"


def test_case_7_later_signed_consent_supersedes_the_pending_one() -> None:
    submission, _, observations = load_case(7)
    assert sorted(consent.patient_signed for consent in observations.consents) == ["no", "yes"]

    result = evaluate(submission, observations)

    consent = check_named(result, "surgical_consent")
    assert consent.status == "pass"
    assert {item.reference for item in consent.evidence} == {document(submission, "Consent for Surgery")["doc_id"]}
    assert result.decision == "READY"


def test_case_7_needs_follow_up_without_the_signed_consent() -> None:
    submission, _, observations = load_case(7)
    observations.consents = [consent for consent in observations.consents if consent.patient_signed != "yes"]

    result = evaluate(submission, observations)

    assert check_named(result, "surgical_consent").summary == "Consent on file is not signed by the patient."
    assert result.decision == "NEEDS_FOLLOW_UP"
    assert categories(result) == ["REQUIRED_DOCUMENTATION"]


def test_case_8_has_no_consent_and_lists_the_documents_checked() -> None:
    submission, _, observations = load_case(8)

    consent = check_named(evaluate(submission, observations), "surgical_consent")

    assert consent.status == "fail"
    assert consent.summary == "No surgical consent found in 3 documents."
    assert sorted(consent.sources_checked) == sorted(f"{d['type']} ({d['date']})" for d in submission["documents"])


def test_case_24_signed_consent_names_a_different_procedure() -> None:
    submission, _, observations = load_case(24)

    consent = check_named(evaluate(submission, observations), "surgical_consent")

    assert consent.status == "fail"
    assert consent.summary == "Signed consent names coronary artery bypass grafting, not the scheduled Elective thyroidectomy."


def test_case_24_consent_passes_if_the_scheduled_procedure_matched_it() -> None:
    submission, _, observations = load_case(24)
    submission["procedure"]["procedure_type"] = "Elective coronary artery bypass grafting"

    assert check_named(evaluate(submission, observations), "surgical_consent").status == "pass"


def test_physician_attestation_is_not_a_patient_signature() -> None:
    submission, _, observations = load_case(1)
    consent_form = document(submission, "Consent - Elective Procedure")
    consent_form["text"] = consent_form["text"].replace(
        "Patient signature on file - signed in clinic 02/25/2026 13:37 (scanned copy attached).",
        "Patient signature: ______________________   Date/Time: __________\n[Form printed for patient review; not yet signed]",
    )
    assert "Electronically signed by Kaitlyn Desai, DO on 02/25/2026." in consent_form["text"]
    (consent,) = observations.consents
    consent.patient_signed, consent.signature_quote = "no", "[Form printed for patient review; not yet signed]"

    result = evaluate(submission, observations)

    assert result.unverified == []
    assert check_named(result, "surgical_consent").status == "fail"
    assert check_named(result, "surgical_consent").summary == "Consent on file is not signed by the patient."
    assert categories(result) == ["REQUIRED_DOCUMENTATION"]


def test_consent_with_a_fabricated_signature_quote_is_unknown_not_pass_or_fail() -> None:
    submission, _, observations = load_case(1)
    (consent,) = observations.consents
    consent.signature_quote = "Signed by patient: Danielle Simmons, 02/25/2026 13:37. Witness: Andrew Wong."

    result = evaluate(submission, observations)

    check = check_named(result, "surgical_consent")
    assert (check.status, check.unknown_cause) == ("unknown", "processing_failure")
    assert [fact.kind for fact in result.unverified] == ["consent"]
    assert result.decision == "NEEDS_FOLLOW_UP"
    assert categories(result) == ["MISSING_REQUIRED_DATA"]


def test_anesthesia_note_called_an_h_and_p_by_the_extractor_does_not_count() -> None:
    submission, label, observations = load_case(13)
    note = next(document for document in submission["documents"] if "anesthesia" in document["type"].lower())
    observations.hp_documents.append(
        HpDocument(doc_id=note["doc_id"], kind="preop_evaluation", quote=note["text"].splitlines()[0])
    )

    result = evaluate(submission, observations)

    check = next(check for check in result.checks if check.name == "history_and_physical")
    assert check.status == "fail"
    assert "not counted, because neither its type nor its heading calls it an H&P" in check.summary
    assert sorted({issue.category for issue in result.issues}) == sorted(label["categories"])
