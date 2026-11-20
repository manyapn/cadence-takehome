"""The one model call: read the documents and report what they say."""

from __future__ import annotations

from functools import cache

from pydantic import BaseModel

from core import PatientSubmission
from observations import Observations

EXTRACTION_PROMPT = """
You read the documents in a surgical patient's chart and report what they say. You do not judge, choose, or apply any rule. Another program does that with your report.

Every item you report names the document it came from and carries a quote. A quote is text copied from that document character for character, with nothing added, removed, corrected, or joined from separate places. Copy symbols such as the degree sign as they are. Keep quotes short. If you cannot quote it, do not report it.

Report all of the following. Return an empty list for a kind that does not occur.

vitals
- One entry for every blood pressure and every temperature written in any document. Include first readings, repeated readings, readings added later in the same document, and readings inside an exam section. Do not skip a reading because a newer one exists.
- Blood pressure: fill systolic and diastolic, leave temperature and unit null. The quote must contain this one blood pressure and no other.
- Temperature: fill temperature and unit exactly as written, leave systolic and diastolic null. Do not convert between units. The quote must contain the number followed by its unit.
- time: the clock time the document gives for that reading, copied as written, or null if it gives none. When there is a time, the quote must start at that time and run through the reading, even when other values sit between them.
- Do not report pulse, oxygen level, pain scores, or a statement that readings are recorded somewhere else.

hp_documents
- One entry for every document that is a history and physical, or is filed or titled as one.
- kind is preop_evaluation when the content is an assessment of this patient ahead of the planned operation. kind is wellness_or_other for a routine, annual, or new-patient visit, or anything else. Decide from the content, not from the document's type label.
- quote: the words that show which kind it is.

consents
- One entry for every consent form or consent discussion document for the operation. Do not report passing remarks about consent in other documents.
- patient_signed is yes only when the document shows the patient's own signature was obtained, in any form. A clinician's signature or attestation is not the patient's. It is no when the document shows the patient has not signed. Otherwise unclear.
- signature_quote: the words that show the patient's signature status.
- procedure_named: the operation the consent is for, as written. procedure_quote: the words naming it.

anticoagulant_mentions
- Statements that the patient takes, took, or may take an anticoagulant, under a brand or generic name. Anticoagulants act on clotting factors: vitamin K antagonists, direct oral anticoagulants, and heparins. Antiplatelet drugs are not anticoagulants and are not reported.
- One entry per drug per document, using the most direct statement.
- status is taking when the document says the patient currently takes it, discontinued when it says the patient stopped or no longer takes it, uncertain when the document itself is unsure.
- A drug that is only proposed as a temporary cover around the operation is not a current medication. Do not report it.
- drug_as_written: the name as it appears in the quote. generic_name: its generic name in lower case.

anticoagulation_plans
- One entry per drug per document that says how an anticoagulant is to be handled around the operation.
- preop covers the time before the operation and postop the time after. For each side: label is instruction when the document states what to do and when, deferred when it leaves the decision to someone else or to a later time, absent when it says nothing. quote is the words for that side, or null when absent.
- drug_as_written: the name as it appears in the document.

scheduling_statements
- Statements that give the date of the operation, or say the date is not set yet.
- procedure_date: the date in YYYY-MM-DD form, or null. pending: true when the document says the date is not set.

Use the document id shown in each document's header as doc_id.
""".strip()


class ExtractionError(RuntimeError):
    pass


@cache
def shared_client():
    from openai import OpenAI

    return OpenAI(timeout=60.0, max_retries=2)


def render_documents(submission: PatientSubmission) -> tuple[str, dict[str, str]]:
    # Short ids are easier for the model to copy exactly than UUIDs.
    doc_ids = {f"D{number}": document.doc_id for number, document in enumerate(submission.documents, start=1)}
    blocks = [
        f"=== document {short_id} | type: {document.type} | date: {document.date} ===\n{document.text or ''}"
        for short_id, document in zip(doc_ids, submission.documents)
    ]
    return "\n\n".join(blocks), doc_ids


def extract(
    submission: PatientSubmission,
    *,
    model: str,
    instructions: str = EXTRACTION_PROMPT,
    schema: type[BaseModel] = Observations,
):
    chart, doc_ids = render_documents(submission)
    response = shared_client().responses.parse(
        model=model,
        instructions=instructions,
        input=chart,
        text_format=schema,
        temperature=0,
        max_output_tokens=8000,
        store=False,
    )

    if response.status != "completed":
        raise ExtractionError(f"response {response.status}: {response.incomplete_details}")
    for item in response.output:
        for part in getattr(item, "content", None) or []:
            if part.type == "refusal":
                raise ExtractionError(f"model refused: {part.refusal}")
    observations = response.output_parsed
    if observations is None:
        raise ExtractionError("response had no parsed output")

    for items in (getattr(observations, name) for name in schema.model_fields):
        for item in items:
            item.doc_id = doc_ids.get(item.doc_id) or item.doc_id
    return observations
