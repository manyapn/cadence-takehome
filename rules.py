"""One check per policy requirement, and the response derived from the checks."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from core import Decision, IssueCategory, PatientSubmission
from observations import DroppedFact, Reading, VerifiedObservations
from resolve import DrugStatus, VitalResolution, resolve_lab
from verify import names

Status = Literal["pass", "fail", "unknown", "not_applicable"]
UnknownCause = Literal["chart_gap", "processing_failure"]

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
    unknown_cause: UnknownCause | None = None
    summary: str
    evidence: list[Evidence] = []
    sources_checked: list[str] = []


class Issue(BaseModel):

    category: IssueCategory
    rule: str
    unknown_cause: UnknownCause | None = None
    finding: str
    evidence: list[Evidence]
    sources_checked: list[str] = []


class TriageResult(BaseModel):

    decision: Decision
    issues: list[Issue]
    checks: list[Check]
    explanation: str
    unverified: list[DroppedFact] = []


def documents_checked(submission: PatientSubmission) -> list[str]:
    return [f"{document.type} ({document.date})" for document in submission.documents]


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
        check["sources_checked"] = documents_checked(submission)
    else:
        newest = max(preop, key=lambda hp: documents[hp.doc_id].date)
        completed = documents[newest.doc_id].date
        evidence = [Evidence(source="note", reference=newest.doc_id, date=completed, quote=newest.quote)]
        days = days_before_procedure(submission, completed)
        if days is None:
            status, check["unknown_cause"] = "unknown", "chart_gap"
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
        check["sources_checked"] = documents_checked(submission)
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
            unknown_cause="chart_gap",
            summary="Procedure risk is missing, so it is unknown whether a CMP is required.",
            sources_checked=["procedure record: procedure_risk is empty"],
        )

    def lab_evidence(lab) -> Evidence:
        return Evidence(
            source="structured",
            reference=lab.id or "labs",
            date=lab.effective_at,
            value=f"{lab.code} {lab.display} ({lab.status})",
        )

    resolution = resolve_lab(submission, test)
    latest = resolution.latest
    not_counted = "".join(
        f" A newer {test} dated {lab.effective_at[:10]} is marked {lab.status} and was not counted."
        for lab in resolution.newer_not_final
    )
    evidence = [lab_evidence(lab) for lab in ([latest] if latest else []) + resolution.newer_not_final]
    if latest is None:
        return Check(
            **check,
            status="fail",
            summary=f"No final {test} result among {len(submission.labs)} lab results.{not_counted}",
            evidence=evidence,
            sources_checked=[f"{lab.code} {lab.display} ({(lab.effective_at or '')[:10]}, {lab.status})" for lab in submission.labs],
        )
    days = days_before_procedure(submission, latest.effective_at)
    if risk is None or days is None:
        missing = "procedure risk" if risk is None else "procedure date"
        return Check(
            **check,
            status="unknown",
            unknown_cause="chart_gap",
            summary=f"Latest {test} is dated {latest.effective_at[:10]}, but the {missing} is missing, so its window cannot be checked.{not_counted}",
            evidence=evidence,
        )
    window = 14 if risk == "HIGH" else 30
    return Check(
        **check,
        status="pass" if 0 <= days <= window else "fail",
        summary=f"Latest {test} ({latest.effective_at[:10]}) is {days} days before the procedure (limit {window} for {risk} risk).{not_counted}",
        evidence=evidence,
    )


# -------------------------
# Rule 3: anticoagulation
# -------------------------


def check_anticoagulation(submission: PatientSubmission, drugs: list[DrugStatus]) -> Check:
    check = {"name": "anticoagulation_plan", "rule": "Rule 3", "category": "ANTICOAGULATION_MANAGEMENT"}
    dates = {document.doc_id: document.date for document in submission.documents}
    sides = (("preop", "before the procedure"), ("postop", "after the procedure"))

    def covered(drug: DrugStatus) -> bool:
        return any(plan.preop.label == plan.postop.label == "instruction" for plan in drug.plans)

    def evidence(drug: DrugStatus) -> list[Evidence]:
        return [
            *(
                Evidence(source="structured", reference="medications", value=f"{medication.name} (active: {medication.active})")
                for medication in drug.medications
            ),
            *(
                Evidence(
                    source="note",
                    reference=mention.doc_id,
                    date=dates[mention.doc_id],
                    value=f"{mention.drug_as_written}: {mention.status}",
                    quote=mention.quote,
                )
                for mention in drug.mentions
            ),
            *(
                Evidence(
                    source="note",
                    reference=plan.doc_id,
                    date=dates[plan.doc_id],
                    value=f"{drug.drug} {wording}: {getattr(plan, side).label}",
                    quote=getattr(plan, side).quote,
                )
                for plan in drug.plans
                for side, wording in sides
            ),
        ]

    current = [drug for drug in drugs if drug.status != "discontinued"]
    if not current:
        return Check(
            **check,
            status="not_applicable",
            summary="No current anticoagulant in the medication list or notes.",
            evidence=[item for drug in drugs for item in evidence(drug)],
            sources_checked=[f"medication list: {medication.name} (active: {medication.active})" for medication in submission.medications]
            + documents_checked(submission),
        )
    open_drugs = [drug for drug in current if not covered(drug)]
    if not open_drugs:
        return Check(
            **check,
            status="pass",
            summary=f"A plan covers {', '.join(drug.drug for drug in current)} before and after the procedure.",
            evidence=[item for drug in current for item in evidence(drug)],
        )

    findings = []
    for drug in open_drugs:
        if drug.status == "uncertain":
            findings.append(f"It is not confirmed whether the patient currently takes {drug.drug}, and no complete plan covers it.")
        elif not drug.recognised:
            findings.append(
                f"The notes report {drug.drug} as an anticoagulant the patient takes. It is not in the drug table,"
                " so it could not be confirmed as one, and no complete plan covers it."
            )
        elif not drug.plans:
            findings.append(f"Patient takes {drug.drug} and there is no documented perioperative plan for it.")
        else:
            for plan in drug.plans:
                labels = ", ".join(f"{wording} is {getattr(plan, side).label}" for side, wording in sides)
                findings.append(f"Patient takes {drug.drug}, and the plan dated {dates[plan.doc_id]} is incomplete: {labels}.")
    stale = [drug for drug in drugs if drug.status == "discontinued" and drug.plans]
    for drug in stale:
        findings.append(
            f"The plan dated {', '.join(dates[plan.doc_id] for plan in drug.plans)} is for {drug.drug}, which the patient no longer takes."
        )
    definite = any(drug.status == "taking" and drug.recognised for drug in open_drugs)
    uncertain = any(drug.status == "uncertain" for drug in open_drugs)
    return Check(
        **check,
        status="fail" if definite else "unknown",
        unknown_cause=None if definite else "chart_gap" if uncertain else "processing_failure",
        summary=" ".join(findings),
        evidence=[item for drug in current + stale for item in evidence(drug)],
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


def check_vital(kind: str, resolution: VitalResolution, dropped: list[DroppedFact], sources_checked: list[str] = []) -> Check:
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
        status, check["unknown_cause"] = "unknown", "processing_failure"
        summary = (
            f"A {label} reading extracted from a note could not be verified against its document,"
            f" so the most recent {label} is not established."
        )
        if latest:
            summary += f" The latest verified reading is {latest}."
    elif not over:
        status, check["unknown_cause"] = "unknown", "chart_gap"
        summary = f"No {label} reading in the structured vitals or the notes."
    elif any(over):
        status, check["unknown_cause"] = "unknown", "chart_gap"
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
    return Check(
        **check,
        status=status,
        summary=summary,
        evidence=evidence,
        sources_checked=sources_checked if not resolution.governing else [],
    )


# -------------------------
# Response
# -------------------------


def run_checks(
    submission: PatientSubmission,
    verified: VerifiedObservations,
    vitals: dict[str, VitalResolution],
    drugs: list[DrugStatus],
) -> list[Check]:
    def unsettled_by_drops(check: Check, when: tuple[str, ...], *kinds: str) -> Check:
        dropped = [fact for fact in verified.dropped if fact.kind in kinds]
        if check.status not in when or not dropped:
            return check
        return check.model_copy(
            update={
                "status": "unknown",
                "unknown_cause": "processing_failure",
                "summary": f"{check.summary} This is not settled: an extracted {dropped[0].kind.replace('_', ' ')} could not be verified against its document.",
                "evidence": check.evidence + dropped_evidence(dropped),
            }
        )

    anticoagulation = check_anticoagulation(submission, drugs)
    anticoagulation = unsettled_by_drops(anticoagulation, ("not_applicable", "pass"), "anticoagulant_mention")
    anticoagulation = unsettled_by_drops(anticoagulation, ("fail",), "anticoagulation_plan")
    return [
        unsettled_by_drops(check_hp(submission, verified), ("fail",), "hp_document"),
        unsettled_by_drops(check_consent(submission, verified), ("fail",), "consent"),
        check_lab("CBC", submission),
        check_lab("CMP", submission),
        anticoagulation,
        *(
            check_vital(
                kind,
                vitals[kind],
                dropped_readings_that_matter(kind, vitals[kind], submission, verified),
                [f"structured vitals ({len(submission.vitals)} entries)"] + documents_checked(submission),
            )
            for kind in ("blood_pressure", "temperature")
        ),
    ]


def respond(checks: list[Check], unverified: list[DroppedFact] = []) -> TriageResult:
    issues = [
        Issue(
            category=check.category if check.status == "fail" else "MISSING_REQUIRED_DATA",
            rule=check.rule,
            unknown_cause=check.unknown_cause,
            finding=check.summary,
            evidence=check.evidence,
            sources_checked=check.sources_checked,
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
    return TriageResult(
        decision=decision,
        issues=issues,
        checks=checks,
        explanation="\n".join([decision] + lines),
        unverified=unverified,
    )
