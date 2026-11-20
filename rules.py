"""One check per policy requirement, and the response derived from the checks."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from core import Decision, IssueCategory, PatientSubmission
from observations import DroppedFact, Reading, VerifiedObservations
from resolve import VitalResolution, anticoagulant
from verify import names

Status = Literal["pass", "fail", "unknown", "not_applicable"]

LAB_CODES = {"CBC": {"CBC", "LAB-CBC"}, "CMP": {"CMP", "LAB-CMP"}}
VITAL_LIMITS = {
    "blood_pressure": "systolic >= 180 or diastolic >= 110 mmHg",
    "temperature": "> 100.4 F",
}


class Evidence(BaseModel):

    source: Literal["structured", "note"]
    reference: str
    date: str | None = None
    value: str | None = None
    quote: str | None = None


class Check(BaseModel):

    name: str
    rule: str
    category: IssueCategory
    status: Status
    summary: str
    evidence: list[Evidence] = []


class Issue(BaseModel):

    category: IssueCategory
    rule: str
    finding: str
    evidence: list[Evidence]


class TriageResult(BaseModel):

    decision: Decision
    issues: list[Issue]
    checks: list[Check]
    explanation: str


def days_before_procedure(submission: PatientSubmission, when: str) -> int | None:
    procedure_date = submission.procedure.procedure_date if submission.procedure else None
    if procedure_date is None:
        return None
    return (date.fromisoformat(procedure_date[:10]) - date.fromisoformat(when[:10])).days


# -------------------------
# Rule 1: documentation
# -------------------------


def check_hp(submission: PatientSubmission, verified: VerifiedObservations) -> Check:
    check = {"name": "history_and_physical", "rule": "Rule 1", "category": "REQUIRED_DOCUMENTATION"}
    documents = {document.doc_id: document for document in submission.documents}
    preop = [hp for hp in verified.hp_documents if hp.kind == "preop_evaluation" and documents[hp.doc_id].date]
    if not preop:
        status, evidence = "fail", []
        summary = f"No pre-operative H&P found in {len(submission.documents)} documents."
    else:
        newest = max(preop, key=lambda hp: documents[hp.doc_id].date)
        completed = documents[newest.doc_id].date
        evidence = [Evidence(source="note", reference=newest.doc_id, date=completed, quote=newest.quote)]
        days = days_before_procedure(submission, completed)
        if days is None:
            status = "unknown"
            summary = f"H&P dated {completed}, but the procedure date is missing, so the 30-day window cannot be checked."
        else:
            status = "pass" if 0 <= days <= 30 else "fail"
            summary = f"H&P dated {completed} is {days} days before the procedure (limit 30)."

    if status == "fail":
        for other in verified.hp_documents:
            document = documents[other.doc_id]
            days = days_before_procedure(submission, document.date) if document.date else None
            if other.kind == "wellness_or_other" and days is not None and 0 <= days <= 30:
                summary += (
                    f" A document filed as '{document.type}' and dated {document.date} is inside the window,"
                    " but it reads as a wellness or other visit, not a pre-operative evaluation."
                )
                evidence.append(Evidence(source="note", reference=other.doc_id, date=document.date, quote=other.quote))
    return Check(**check, status=status, summary=summary, evidence=evidence)


def check_consent(submission: PatientSubmission, verified: VerifiedObservations) -> Check:
    check = {"name": "surgical_consent", "rule": "Rule 1", "category": "REQUIRED_DOCUMENTATION"}
    documents = {document.doc_id: document for document in submission.documents}
    scheduled = (submission.procedure.procedure_type if submission.procedure else None) or ""

    def evidence(consents) -> list[Evidence]:
        return [
            Evidence(
                source="note",
                reference=consent.doc_id,
                date=documents[consent.doc_id].date,
                value=f"patient signed: {consent.patient_signed}; procedure: {consent.procedure_named}",
                quote=quote,
            )
            for consent in consents
            for quote in (consent.signature_quote, consent.procedure_quote)
            if quote
        ]

    signed = [consent for consent in verified.consents if consent.patient_signed == "yes"]
    valid = [consent for consent in signed if consent.procedure_named and names(scheduled, consent.procedure_named)]
    if valid:
        return Check(
            **check,
            status="pass",
            summary=f"Consent for {valid[0].procedure_named} is signed by the patient.",
            evidence=evidence(valid),
        )
    if signed:
        summary = f"Signed consent names {signed[0].procedure_named}, not the scheduled {scheduled}."
    elif verified.consents:
        summary = "Consent on file is not signed by the patient."
    else:
        summary = f"No surgical consent found in {len(submission.documents)} documents."
    return Check(**check, status="fail", summary=summary, evidence=evidence(signed or verified.consents))


# -------------------------
# Rule 2: testing
# -------------------------


def check_lab(test: str, submission: PatientSubmission) -> Check:
    check = {"name": test.lower(), "rule": "Rule 2", "category": "REQUIRED_TESTING"}
    risk = submission.procedure.procedure_risk if submission.procedure else None
    if test == "CMP" and risk in ("LOW", "MODERATE"):
        return Check(**check, status="not_applicable", summary=f"CMP is not required for a {risk} risk procedure.")
    if test == "CMP" and risk is None:
        return Check(
            **check,
            status="unknown",
            summary="Procedure risk is missing, so it is unknown whether a CMP is required.",
        )
    results = [
        lab for lab in submission.labs if lab.code in LAB_CODES[test] and lab.status == "final" and lab.effective_at
    ]
    if not results:
        return Check(
            **check,
            status="fail",
            summary=f"No final {test} result among {len(submission.labs)} lab results.",
        )
    latest = max(results, key=lambda lab: lab.effective_at)
    evidence = [
        Evidence(
            source="structured",
            reference=latest.id or "labs",
            date=latest.effective_at,
            value=f"{latest.code} {latest.display}",
        )
    ]
    days = days_before_procedure(submission, latest.effective_at)
    if risk is None or days is None:
        missing = "procedure risk" if risk is None else "procedure date"
        return Check(
            **check,
            status="unknown",
            summary=f"Latest {test} is dated {latest.effective_at[:10]}, but the {missing} is missing, so its window cannot be checked.",
            evidence=evidence,
        )
    window = 14 if risk == "HIGH" else 30
    return Check(
        **check,
        status="pass" if 0 <= days <= window else "fail",
        summary=f"Latest {test} ({latest.effective_at[:10]}) is {days} days before the procedure (limit {window} for {risk} risk).",
        evidence=evidence,
    )


# -------------------------
# Rule 3: anticoagulation
# -------------------------


def check_anticoagulation(submission: PatientSubmission, verified: VerifiedObservations) -> Check:
    check = {"name": "anticoagulation_plan", "rule": "Rule 3", "category": "ANTICOAGULATION_MANAGEMENT"}
    documents = {document.doc_id: document for document in submission.documents}
    reported: dict[str, set[str]] = {}
    evidence: dict[str, list[Evidence]] = {}

    for medication in submission.medications:
        if drug := anticoagulant(medication.name or ""):
            status = {True: "taking", False: "discontinued", None: "uncertain"}[medication.active]
            reported.setdefault(drug, set()).add(status)
            evidence.setdefault(drug, []).append(
                Evidence(source="structured", reference="medications", value=f"{medication.name} (active: {medication.active})")
            )
    for mention in verified.anticoagulant_mentions:
        if drug := anticoagulant(mention.drug_as_written):
            reported.setdefault(drug, set()).add(mention.status)
            evidence.setdefault(drug, []).append(
                Evidence(
                    source="note",
                    reference=mention.doc_id,
                    date=documents[mention.doc_id].date,
                    value=f"{mention.drug_as_written}: {mention.status}",
                    quote=mention.quote,
                )
            )

    taking = sorted(drug for drug, statuses in reported.items() if "taking" in statuses)
    uncertain = sorted(drug for drug, statuses in reported.items() if "uncertain" in statuses and drug not in taking)
    if not taking and not uncertain:
        return Check(**check, status="not_applicable", summary="No current anticoagulant in the medication list or notes.")

    gaps = []
    for drug in taking:
        plans = [plan for plan in verified.anticoagulation_plans if anticoagulant(plan.drug_as_written) == drug]
        for plan in plans:
            for side, label in ((plan.preop, "before"), (plan.postop, "after")):
                if side.quote:
                    evidence[drug].append(
                        Evidence(
                            source="note",
                            reference=plan.doc_id,
                            date=documents[plan.doc_id].date,
                            value=f"{label} the procedure: {side.label}",
                            quote=side.quote,
                        )
                    )
        if not plans:
            gaps.append(f"no documented plan for {drug}")
        elif not any(plan.preop.label == plan.postop.label == "instruction" for plan in plans):
            gaps.append(f"the plan for {drug} does not give instructions for both before and after the procedure")

    listed = [item for drug in taking + uncertain for item in evidence[drug]]
    if gaps:
        return Check(**check, status="fail", summary=f"Patient takes {', '.join(taking)}: {'; '.join(gaps)}.", evidence=listed)
    if uncertain:
        return Check(
            **check,
            status="unknown",
            summary=f"It is unclear whether the patient currently takes {', '.join(uncertain)}.",
            evidence=listed,
        )
    return Check(
        **check,
        status="pass",
        summary=f"Patient takes {', '.join(taking)}, with a plan covering before and after the procedure.",
        evidence=listed,
    )


# -------------------------
# Rule 4: acute safety exclusions
# -------------------------


def exceeds_limit(reading: Reading) -> bool:
    if reading.kind == "blood_pressure":
        return reading.systolic >= 180 or reading.diastolic >= 110
    return reading.temperature_f > Decimal("100.4")


def reading_value(reading: Reading) -> str:
    if reading.kind == "blood_pressure":
        return f"{reading.systolic}/{reading.diastolic} mmHg"
    return f"{reading.temperature_f} F"


def reading_evidence(reading: Reading) -> Evidence:
    return Evidence(
        source=reading.source,
        reference=reading.reference,
        date=" ".join(filter(None, [reading.date, reading.time])) or None,
        value=reading_value(reading),
        quote=reading.quote,
    )


def dropped_evidence(dropped: list[DroppedFact]) -> list[Evidence]:
    return [Evidence(source="note", reference=fact.doc_id, value=f"unverified {fact.kind}: {fact.reason}") for fact in dropped]


def dropped_readings_that_matter(
    kind: str,
    resolution: VitalResolution,
    submission: PatientSubmission,
    verified: VerifiedObservations,
) -> list[DroppedFact]:
    documents = {document.doc_id: document for document in submission.documents}
    dates = [reading.date for reading in resolution.governing]
    governing_date = min(dates) if dates and all(dates) else None

    def could_be_latest(fact: DroppedFact) -> bool:
        document = documents.get(fact.doc_id)
        return governing_date is None or document is None or not document.date or document.date >= governing_date

    return [fact for fact in verified.dropped if fact.kind == "vital" and fact.claim["kind"] == kind and could_be_latest(fact)]


def check_vital(kind: str, resolution: VitalResolution, dropped: list[DroppedFact]) -> Check:
    check = {"name": kind, "rule": "Rule 4", "category": "ACUTE_SAFETY_EXCLUSION"}
    label = kind.replace("_", " ")
    over = [exceeds_limit(reading) for reading in resolution.governing]
    latest = " and ".join(
        f"{reading_value(reading)} ({' '.join(filter(None, [reading.date, reading.time]))})"
        for reading in resolution.governing
    )
    overridden = []

    if over and all(over):
        status, summary = "fail", f"Most recent {label} {latest} meets the exclusion limit ({VITAL_LIMITS[kind]})."
    elif dropped:
        status = "unknown"
        summary = (
            f"A {label} reading extracted from a note could not be verified against its document,"
            f" so the most recent {label} is not established."
        )
        if latest:
            summary += f" The latest verified reading is {latest}."
    elif not over:
        status, summary = "unknown", f"No {label} reading in the structured vitals or the notes."
    elif any(over):
        status = "unknown"
        summary = (
            f"{label.capitalize()} readings {latest} cannot be put in order, and only one side of them"
            f" meets the exclusion limit ({VITAL_LIMITS[kind]})."
        )
    else:
        status, summary = "pass", f"Most recent {label} {latest} is below the exclusion limit ({VITAL_LIMITS[kind]})."
        overridden = [entry for entry in resolution.superseded if exceeds_limit(entry.reading)]
        for entry in overridden:
            summary += f" It supersedes {reading_value(entry.reading)} ({entry.reason})."

    evidence = [reading_evidence(reading) for reading in resolution.governing + [entry.reading for entry in overridden]]
    if status == "unknown":
        evidence += dropped_evidence(dropped)
    return Check(**check, status=status, summary=summary, evidence=evidence)


# -------------------------
# Response
# -------------------------


def run_checks(
    submission: PatientSubmission,
    verified: VerifiedObservations,
    vitals: dict[str, VitalResolution],
) -> list[Check]:
    def unsettled_by_drops(check: Check, when: tuple[str, ...], *kinds: str) -> Check:
        dropped = [fact for fact in verified.dropped if fact.kind in kinds]
        if check.status not in when or not dropped:
            return check
        return check.model_copy(
            update={
                "status": "unknown",
                "summary": f"{check.summary} This is not settled: an extracted {dropped[0].kind.replace('_', ' ')} could not be verified against its document.",
                "evidence": check.evidence + dropped_evidence(dropped),
            }
        )

    anticoagulation = check_anticoagulation(submission, verified)
    anticoagulation = unsettled_by_drops(anticoagulation, ("not_applicable", "pass"), "anticoagulant_mention")
    anticoagulation = unsettled_by_drops(anticoagulation, ("fail",), "anticoagulation_plan")
    return [
        unsettled_by_drops(check_hp(submission, verified), ("fail",), "hp_document"),
        unsettled_by_drops(check_consent(submission, verified), ("fail",), "consent"),
        check_lab("CBC", submission),
        check_lab("CMP", submission),
        anticoagulation,
        *(
            check_vital(kind, vitals[kind], dropped_readings_that_matter(kind, vitals[kind], submission, verified))
            for kind in ("blood_pressure", "temperature")
        ),
    ]


def respond(checks: list[Check]) -> TriageResult:
    issues = [
        Issue(
            category=check.category if check.status == "fail" else "MISSING_REQUIRED_DATA",
            rule=check.rule,
            finding=check.summary,
            evidence=check.evidence,
        )
        for check in checks
        if check.status in ("fail", "unknown")
    ]
    if any(issue.category == "ACUTE_SAFETY_EXCLUSION" for issue in issues):
        decision = "NOT_CLEARED"
    elif issues:
        decision = "NEEDS_FOLLOW_UP"
    else:
        decision = "READY"
    lines = [f"{check.rule} {check.name}: {check.status}. {check.summary}" for check in checks]
    return TriageResult(decision=decision, issues=issues, checks=checks, explanation="\n".join([decision] + lines))
