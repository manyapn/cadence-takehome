# pre-op triage: implementation plan

This plan is a handoff. It was written in a separate planning session after reading the repo, the brief, and all 50 sample cases. You (Claude) are implementing it with me one step at a time.

## how we work

- One step per turn. Stop when the step's "done when" is met, show me the result, and wait.
- Do not start a later step early, and do not change files outside the step's scope.
- Tests come with the code in the same step. Run them and show the output.
- Ask before any model call, and say roughly how many calls it will make. The key is in `OPENAI_API_KEY`. Never print it.
- Items marked **decision** are mine. Ask me, lay out the tradeoff, and do not pick for me.
- If a label looks wrong or a rule can't reproduce it, say so. Do not bend a rule to fit a label.
- Treat every dataset fact in this plan as a claim. Step 0 re-derives them. If one is wrong, tell me before relying on it.
- Match the repo: flat modules next to `core.py`, double quotes, type hints, `uv` scripts. No frameworks, no abstraction without two callers.
- Prompts must not contain phrases copied from the sample notes.
- Commit once per step with a plain imperative message.

## the task

- The brief is `docs/brief.pdf` (I will add it). Its appendix policy is the only source of clearance criteria. `core.py` holds a condensed copy in `BASELINE_SYSTEM_PROMPT`.
- Input: one patient submission (demographics, procedure, vitals, labs, medications, conditions, free-text documents).
- Output: `decision` (`READY`, `NEEDS_FOLLOW_UP`, `NOT_CLEARED`), `issues[]` each with a `category`, and `explanation`. The evidence design is ours.
- A pre-op nurse reviews every non-READY case and a sample of READY ones, so every finding has to be checkable from the output alone.
- Scoring (`run_evals.py`): schema valid, decision equals label, and the set of issue categories equals the label's set exactly.

## policy summary

- Rule 1: H&P completed within 30 days of the procedure date, and a signed surgical consent. Missing or outdated is `NEEDS_FOLLOW_UP`.
- Rule 2: LOW or MODERATE risk needs a CBC within 30 days. HIGH needs a CBC and a CMP within 14 days. Only the most recent result of each test counts.
- Rule 3: a patient currently on an anticoagulant needs a documented plan that says how the drug is managed before and after the procedure. None, incomplete, or ambiguous is `NEEDS_FOLLOW_UP`.
- Rule 4: systolic >= 180, diastolic >= 110, or temperature > 100.4 F at review is `NOT_CLEARED`. The most recent reading counts; a recheck within range is sufficient.
- A required field that is missing or unknown is `NEEDS_FOLLOW_UP`.

## design

The model reads. Code decides.

| stage | file | job |
|---|---|---|
| read | `extract.py` | One model call. Reports every relevant observation in the notes, each with a `doc_id` and a verbatim quote. It never picks a "latest" value and never applies policy. |
| verify | `verify.py` | Each quote must be in its cited document (matching on letters, digits, and `. / :` only; the output carries the document's own wording) and contain the numbers it claims. A BP quote holds exactly one `systolic/diastolic` pair, and it is the claimed pair. Within one note, time order and text order must agree. Failed facts are dropped and recorded. |
| resolve | `resolve.py` | Decides which observation governs and records why one superseded another. No thresholds here. |
| rules | `rules.py` | One check per policy requirement, returning `pass`, `fail`, `unknown`, or `not_applicable` with evidence. Every check runs, even after an exclusion. |
| respond | `rules.py` | `decision`, `issues`, and `explanation` are all derived from the same check results. No second model call. |

`pipeline.evaluate(submission, observations)` runs verify, resolve, rules, and respond with zero model calls. That is the replay path every test except the live ones uses.

### why this split

- The thresholds are exact (day 30 passes, day 31 fails; diastolic 110 fails), so they belong in code.
- The deciding facts are in free text in many formats, so reading needs a model.
- The harness compares full outputs across 10 runs, so nothing in the output can be free-written by the model.
- Reading, selecting, and judging fail in different ways. Separate stages mean a wrong answer can be traced to one of them.

### extractor contract

Observations it returns (strict schema, every field required, nullable where it can be absent):

- `vitals[]`: `doc_id`, kind (`blood_pressure` or `temperature`), systolic, diastolic, temperature, unit (`F` or `C`), time as written, quote. Every documented reading, including arrival readings and rechecks.
- `hp_documents[]`: `doc_id`, kind (`preop_evaluation` or `wellness_or_other`), quote.
- `consents[]`: `doc_id`, patient signed (`yes`, `no`, `unclear`), signature quote, procedure named, procedure quote.
- `anticoagulant_mentions[]`: `doc_id`, drug as written, generic name, status (`taking`, `discontinued`, `uncertain`), quote.
- `anticoagulation_plans[]`: `doc_id`, drug as written, pre-op side and post-op side each labeled `instruction`, `deferred`, or `absent`, with a quote for each.
- `scheduling_statements[]`: `doc_id`, an explicit procedure date or a statement that it is pending, quote.

The date of a note reading comes from the document's `date` field in code, not from the model.

### resolver contract

- Vitals: structured vitals and verified note readings go into one list. The latest governs. Order is by date first.
- Same date, same note: text order (the quote's character offset). The verifier has already checked that it agrees with the times.
- Same date, different sources, both with a time: the later clock time governs. Times are zero-padded before comparing.
- Same date, different sources, one without a time: the order is arbitrary, so neither supersedes the other.
- "Recheck" is wording for the explanation. Selection is by time. A recheck that is still high governs just the same.
- Readings that can't be ordered are both returned as governing. Both over a threshold fails, both under passes, and one over with one under is `unknown` (`MISSING_REQUIRED_DATA`), because the chart does not say which is the latest.
- The result must not depend on the order observations arrive in.
- Labs: map code and display to CBC, CMP, or neither. `final` results only. Newest first.
- H&P: the newest document classified as a pre-op evaluation.
- Consent: a consent counts when the patient signed it and it names the scheduled procedure. A later signed consent supersedes an earlier pending one.
- Anticoagulants: per drug, combine the structured `active` flag with note mentions into taking, discontinued, or uncertain. Among notes the newest document decides. If the medication list and the newest note disagree, the status is uncertain.
- Brand and generic names map to one generic through a fixed table. Known antiplatelets are ignored even if the extractor reports them.
- A drug the extractor calls an anticoagulant that is in neither table is kept and marked unrecognised. If a complete plan covers it the rule passes either way; if not, the check is `unknown`, because code cannot confirm it is an anticoagulant.
- Plans: match each plan to the drug the patient is currently taking and carry its pre-op and post-op labels and quotes through unchanged. A drug is covered only when one document gives an instruction for both sides.

### rules contract

- Rule 3 alone decides plan completeness: a plan for the active drug with both sides labeled `instruction`.
- Celsius converts to Fahrenheit with `Decimal`, no rounding, so 38.0 C is exactly 100.4 F and passes.
- Day counts are calendar days: procedure date minus document or lab date. 0 to 30 (or 14) passes.
- A `fail` raises the rule's own category. An `unknown` raises `MISSING_REQUIRED_DATA`.
- Decision: any `ACUTE_SAFETY_EXCLUSION` gives `NOT_CLEARED`. Any other open check gives `NEEDS_FOLLOW_UP`. Otherwise `READY`. All issues are listed regardless.

### unknown has two causes

- Chart gap: the chart was read and the item is not there.
- Processing failure: a fact was dropped by verification, or a call failed.

Rules:

- A dropped fact never becomes a claim about the chart. If it could change a check, that check is `unknown` with the reason stated.
- A dropped vital only matters if its document is dated on or after the governing reading.
- A verified exclusion stands no matter what else was dropped.

### output shape

- `decision`
- `issues[]`: `category`, `rule`, `finding`, `evidence[]`
- `checks[]`: every rule with status, summary, and evidence, so a READY case can be verified too
- `explanation`: templated from the checks
- Evidence item: source (`structured` or `note`), reference (doc id, lab id, or vitals index), date, value, and quote for note-derived facts
- Evidence for a missing item lists which sources were checked

## what the sample shows

Claims from the planning session. Step 0 re-derives them.

**Shape**

- 50 cases: 17 `READY`, 25 `NEEDS_FOLLOW_UP`, 8 `NOT_CLEARED`.
- The notes are template-generated: 1,181 distinct lines, 856 after replacing digit runs. A regex solution would overfit; the brief says real documents are longer.
- `submission_received_at` falls on the procedure date in all 48 dated cases.

**Vitals**

- In 21 cases the latest BP is only in note text.
- Formats include a bare `119/73 mmHg`, `NIBP`, a temperature right after a heart rate, Celsius, indented lines, and a BP in an addendum with the temperature in the body.
- Structured vitals are unsorted in most cases.
- Many notes say "see flowsheet" in place of vitals. H&Ps carry their own older vitals line.
- 5 cases have no temperature anywhere (9, 11, 13, 21, 22). Case 22 has no vitals at all.

**Labs**

- 13 distinct `(code, display)` pairs. `LAB-CBC` and `LAB-CMP` ("w/ Magnesium") count. A BMP does not.
- 3 `entered-in-error` CBCs (cases 4, 6, 41).

**H&P**

- Document `type` and the note's own header are decoupled. Types include a typo ("History and Pyhsical").
- 29 cases have an annual wellness or establish-care note filed under an H&P-like type. None is within 30 days of the procedure (closest is 34).

**Consent**

- Signed variants: scanned signature on file, `/s/` e-signature, "Signed by patient ... Witness".
- Unsigned variants: signature pending, blank signature line, e-signature request not completed, or no consent document at all.
- Every consent has a physician attestation signature, including the unsigned ones.
- Some consents refuse blood products. That is not a policy criterion.
- Most H&Ps say formal consent is "per surgical team." That is benign.

**Anticoagulants**

- Brand names only, drugs reported in a nursing note but absent from the medication list, inactive historical warfarin, and a medication with `active: null`.
- Aspirin and clopidogrel appear often and are antiplatelets.

**Clean in the sample, so untested**

- Document dates match in-text dates.
- "Scheduled for" dates match the structured date.
- Consent patient names match.
- Same-day structured and note vitals never decide a case. They do exist: cases 8, 9, 20, and 38 each have a structured BP at 09:05Z and an H&P BP at 10:05 on one date with different values, but a later reading governs in all four.
- Nothing is dated after the procedure.
- Every case supplies every list field.

**Harness quirks**

- `make determinism` reports 100% decision stability when every run fails the same way.
- The existing tests pin `strict is False` and exactly one model call.

## key cases

Expected results come from the labels. Case N is `case_000NN`.

| case | expected | what it tests |
|---|---|---|
| 16 | READY | arrival BP 189/112 at 08:06, recheck 137/78 at 08:30 in an addendum |
| 32, 46 | NOT_CLEARED | recheck still over 180 systolic |
| 5, 47 | NOT_CLEARED | diastolic 117 in a note newer than normal structured vitals |
| 28 | NOT_CLEARED + REQUIRED_DOCUMENTATION | diastolic exactly 110, and consent not completed |
| 15 | READY | 184/112 is not the latest reading |
| 30 | READY | 37.9 C is 100.2 F |
| 2, 35 | NOT_CLEARED | fever recorded in Celsius in a note |
| 12 | READY | H&P and CBC both exactly 30 days out |
| 14 | REQUIRED_DOCUMENTATION | H&P 31 days out |
| 9 | REQUIRED_TESTING + MISSING_REQUIRED_DATA | HIGH risk CBC 15 days out; no temperature anywhere |
| 17 | REQUIRED_TESTING | HIGH risk with a BMP but no CMP |
| 4, 41 | READY | newest CBC is entered-in-error; the prior one is in window |
| 6 | REQUIRED_TESTING | newest CBC is entered-in-error; the prior one is too old |
| 1, 45 | READY | pre-op H&P filed under a typo'd type; older wellness note ignored |
| 13, 48 | REQUIRED_DOCUMENTATION (+ other) | no H&P anywhere |
| 7, 49 | READY | pending consent superseded by a later signed one |
| 24 | REQUIRED_DOCUMENTATION | signed consent names a different procedure |
| 8, 40 | REQUIRED_DOCUMENTATION | no consent document |
| 18 | ANTICOAGULATION_MANAGEMENT | plan was written for warfarin; patient now takes apixaban |
| 20 | ANTICOAGULATION_MANAGEMENT | pre-op hold documented; post-op deferred |
| 23, 36, 48 | ANTICOAGULATION_MANAGEMENT | consult hands the decision to the surgical team |
| 27, 38 | ANTICOAGULATION_MANAGEMENT | anticoagulant only in a nursing note; no plan |
| 0, 3, 37 | READY | brand-name anticoagulant with a complete plan |
| 19, 43 | READY | warfarin listed but inactive and historically stopped |
| 33 | MISSING_REQUIRED_DATA | `active: null` and the patient can't confirm |
| 25 | READY | aspirin plus clopidogrel; no anticoagulant |
| 11, 44 | MISSING_REQUIRED_DATA | procedure date null |
| 29, 39 | MISSING_REQUIRED_DATA | procedure risk null |

## assumptions

Every clearance criterion cites the appendix. Every ambiguity below states the assumption, the rationale, and the test that pins it. These go into `WRITEUP.md` as they are made.

| ambiguity | proposed assumption | rationale |
|---|---|---|
| Missing temperature | `MISSING_REQUIRED_DATA` | cases 9, 11, 13, 21, 22 |
| Missing procedure date | window checks become `unknown`; presence checks still run | cases 11, 44 |
| Missing risk level | lab window is `unknown`; a missing CBC still fails | cases 29, 39 |
| Day counting | calendar days; day 30 passes | cases 12, 14 |
| Consent and the 30-day window | the window applies to the H&P only | `core.py` policy text |
| Same-day times | structured timestamps and note times are compared as written | notes give no timezone |
| What counts as an H&P (decided) | pre-op evaluations only | cases 13 and 48 say the H&P is "not yet on file for this surgical episode", so a wellness visit is not the document the policy asks for, and counting one would be a false READY. If the only in-window document filed as an H&P is a wellness visit, the evidence says so and the nurse can overrule it |
| **decision:** date only in note text | use it when exactly one explicit date is stated and nothing says pending, or always treat as missing | the labeler trusts note text elsewhere; the sample never tests it |
| **decision:** extraction cache | on-disk cache for iteration, off for determinism runs, or no cache | speeds the loop; has to be disclosed |

## known limits

- A time the quote does not back up (missing from the quote, not a clock time, or a 12-hour time like "2:15 PM") is discarded, not converted. The reading is kept as untimed, so it can still be ordered by date or by its place in the note, but not by clock time against another source.
- An H&P classification from the extractor only counts when the document's own type or first heading calls it an H&P. A real H&P filed and headed as something else would be missed and flagged for follow-up.
- Structured timestamps and note times are compared as written. Notes give no timezone.
- A structured vital is referenced by its source and timestamp, not its list index, so the output does not change with input order.

## steps

### step 0: orient, verify, capture the baseline

- Read `docs/brief.pdf`, `README.md`, `core.py`, `run_baseline.py`, `run_evals.py`, and the tests. Summarize the task back to me in a few bullets.
- Write a throwaway script that re-derives the claims under "what the sample shows." Report anything that does not match.
- Run the existing tests.
- With my go-ahead, run the untouched starter (`make baseline`, `make evals`) and save its outputs and report under `results/baseline/`.
- Done when: I have the starter's score, a list of confirmed and unconfirmed claims, and nothing in the repo's code has changed.

### step 1: observation models and the verifier

- `observations.py`: the extractor's Pydantic models and the `Reading` type the resolver uses.
- `verify.py`: locate a quote in its document (whitespace-insensitive), check stated numbers appear in the quote, convert Celsius, drop and record failures.
- A BP must be written as a pair (157/117 with HR 69 does not verify 157/69), and a BP quote may hold only one BP, so the extractor has to quote the reading's own span.
- An `nn/nn` only counts as a BP when the values are plausible (systolic 50 to 300, diastolic 20 or more and below the systolic), so a date like 03/15 in the quote does not cost a real reading.
- Within one note, readings whose time order disagrees with their text order are dropped.
- Tests: a verbatim quote passes; an indented quote passes; a fabricated quote is dropped; a real quote with the wrong numbers is dropped; a split pair is dropped; a quote with two BPs is dropped; a note with times out of text order is dropped.
- Done when: tests pass with no model calls.

### step 2: case 16 through the replay path

- `resolve.py` for vitals; `rules.py` with all checks in their simplest correct form; the response builder; `pipeline.evaluate`.
- A hand-written observations fixture for case 16, authored from the chart.
- Done when all of these pass:
  - Resolution selects 137/78 and records that the 08:30 reading superseded the 08:06 arrival reading.
  - Rule 4's BP check passes and the case returns `READY` with `issues: []`.
  - The test runs with zero model calls, and reversing the input order of structured vitals and observations gives an identical output.
  - Removing the recheck from the fixture gives `NOT_CLEARED` (a real quote must not be accepted as the latest by default).

### step 3: the extractor

- `extract.py`: one call, `client.responses.parse` with the strict schema, one shared client with a timeout and bounded retries, a compact chart as input, refusals and incomplete responses raised.
- A live test, skipped without a key: extraction on case 16 returns 189/112 at 08:06 and 137/78 at 08:30, each with a quote that verifies.
- Done when: the live test passes and case 16 returns `READY` end to end with a real model call.

### step 4: rule 3 and plan matching

- Anticoagulant status per drug, plan-to-drug matching, pre-op and post-op labels carried through, rule 3 deciding completeness.
- Fixtures and tests for cases 18, 20, 33, then 19, 25, 27, and 3.
- Done when: each case's replay test gives the expected decision and categories, and case 18 fails specifically because the plan names a drug the patient no longer takes.

### step 5: rules 1 and 2 hardened

- Tests on both sides of every threshold: H&P day 30 and 31; CBC day 30 and 31; CBC and CMP day 14 and 15; systolic 179 and 180; diastolic 109 and 110; temperature 100.4 and 100.5 F; 38.0 C.
- Named cases: 4, 6, 17, 41 for labs; 1, 13, 14 for H&P; 7, 8, 24 for consent; 11, 29 for missing inputs.
- Done when: every policy rule has at least one test that changes the decision.

### step 6: the response

- Case 28 in three states: as written is `NOT_CLEARED` with both issues; add a normal later BP and it is `NEEDS_FOLLOW_UP` with the consent issue only; add a signed consent and it is `READY`. Assert the remaining issues at each state.
- Unknown split into chart gap and processing failure, visible in the output.
- A dropped decisive fact leaves its check unresolved; a verified exclusion survives it.
- Done when: those tests pass and I have reviewed three sample outputs, one per decision.

### step 7: wire it in

- `triage_submission(...)` calls extract, then `pipeline.evaluate`. The starter stays callable as `baseline.py` for comparison.
- Update the existing tests that pin the old behavior.
- A behavioral test: a failed or refused model call yields unresolved checks and `NEEDS_FOLLOW_UP`, never `READY` and never a crash.
- Done when: `make test` passes.

### step 8: evaluate

- Run the final system and the baseline back to back on the same model, three runs each.
- Report:
  - Decision and category match.
  - 8 of 8 exclusions caught.
  - How many of the 33 non-READY cases came back `READY`.
  - Latest BP correct in the 21 note-only cases.
  - Every evidence quote verbatim in its document.
- Attribute every miss to read, verify, resolve, or rules. A miss on an unambiguous rule is a bug to fix.
- Done when: results are saved under `results/` and each miss has an owner.

### step 9: determinism

- Five varied cases, 10 runs each, cache off.
- Report valid outputs, decision agreement, issue agreement, and exact-output agreement separately. Only valid outputs count.
- Done when: the numbers are saved, and any instability is traced to its cause.

### step 10: experiments (if time)

- One extraction call versus three focused calls.
- Pipeline with verification turned off.
- About 8 perturbed cases: a paraphrase should not change the decision; changing the underlying fact should.

### step 11: handoff

- `WRITEUP.md`: approach, why the split, evidence design, results with the baseline comparison, assumptions, what I would do next.
- README: run instructions from a fresh checkout.
- Done when: from a fresh clone a reviewer can run one case, run the tests, and find the baseline-versus-final results.

### step 12: intake (last)

- Preserve whether a list field was omitted or supplied empty, and route both runners through the same intake path.

## definition of done

| area | bar |
|---|---|
| normalizer | 13 lab pairs map correctly; 3 `entered-in-error` excluded; BMP never accepted as CMP |
| resolver | cases 16, 18, 20, 33 resolve as described; independent of input order |
| rules | every threshold tested on both sides; every rule has a decision-changing test; case 28 three-state test passes |
| extraction | 8 of 8 exclusions; 0 of 33 non-READY returned `READY`; latest BP right in 21 note-only cases |
| verification | every output quote verbatim; fabricated quote dropped; dropped decisive fact leaves the check unresolved |
| evidence | every issue has a source, date, and quote or value; missing items list sources checked; every rule in `checks[]` |
| api | strict schema, shared client, timeout, retries; a failed call never yields `READY` |
| determinism | 10 of 10 valid and identical on 5 cases, or the real number and its cause |
| handoff | fresh-checkout run works; write-up, README, and session export are in place |
