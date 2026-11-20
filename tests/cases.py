from __future__ import annotations

import json
from pathlib import Path

from observations import Observations

ROOT = Path(__file__).resolve().parent.parent


def load_case(number: int) -> tuple[dict[str, object], dict[str, object], Observations]:
    """The sample submission, its label, and the hand-checked observations fixture."""

    case_id = f"case_{number:05d}"
    with (ROOT / "data" / "patients_sample_50.jsonl").open() as handle:
        case = next(case for case in map(json.loads, handle) if case["case_id"] == case_id)
    fixture = ROOT / "tests" / "fixtures" / f"{case_id}_observations.json"
    return case["submission"], case["label"], Observations.model_validate_json(fixture.read_text())
