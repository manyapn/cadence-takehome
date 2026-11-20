"""What the extractor reports from the notes, and what survives verification."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel

# -------------------------
# Extractor output
# -------------------------


class VitalObservation(BaseModel):

    doc_id: str
    kind: Literal["blood_pressure", "temperature"]
    systolic: int | None
    diastolic: int | None
    temperature: float | None
    unit: Literal["F", "C"] | None
    time: str | None
    quote: str


class HpDocument(BaseModel):

    doc_id: str
    kind: Literal["preop_evaluation", "wellness_or_other"]
    quote: str


class ConsentObservation(BaseModel):

    doc_id: str
    patient_signed: Literal["yes", "no", "unclear"]
    signature_quote: str | None
    procedure_named: str | None
    procedure_quote: str | None


class AnticoagulantMention(BaseModel):

    doc_id: str
    drug_as_written: str
    generic_name: str
    status: Literal["taking", "discontinued", "uncertain"]
    quote: str


class PlanSide(BaseModel):

    label: Literal["instruction", "deferred", "absent"]
    quote: str | None


class AnticoagulationPlan(BaseModel):

    doc_id: str
    drug_as_written: str
    preop: PlanSide
    postop: PlanSide


class SchedulingStatement(BaseModel):

    doc_id: str
    procedure_date: str | None
    pending: bool
    quote: str


class Observations(BaseModel):

    vitals: list[VitalObservation]
    hp_documents: list[HpDocument]
    consents: list[ConsentObservation]
    anticoagulant_mentions: list[AnticoagulantMention]
    anticoagulation_plans: list[AnticoagulationPlan]
    scheduling_statements: list[SchedulingStatement]


# -------------------------
# After verification
# -------------------------


class Reading(BaseModel):

    kind: Literal["blood_pressure", "temperature"]
    systolic: float | int | None = None
    diastolic: float | int | None = None
    temperature_f: Decimal | None = None
    source: Literal["structured", "note"]
    reference: str
    date: str | None
    time: str | None = None
    quote: str | None = None
    offset: int | None = None


class DroppedFact(BaseModel):

    kind: str
    doc_id: str
    reason: str
    claim: dict[str, Any]


class VerifiedObservations(BaseModel):

    readings: list[Reading] = []
    hp_documents: list[HpDocument] = []
    consents: list[ConsentObservation] = []
    anticoagulant_mentions: list[AnticoagulantMention] = []
    anticoagulation_plans: list[AnticoagulationPlan] = []
    scheduling_statements: list[SchedulingStatement] = []
    dropped: list[DroppedFact] = []
