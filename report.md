# Conversation-to-Spec Technical Report

- Student ID: `YOUR_STUDENT_ID`
- PDF filename for submission: `YOUR_STUDENT_ID.pdf`

## Summary (<= 250 words)
This project builds a Python-only, CLI-based prototype that converts messy client conversations into structured software requirement drafts. The core problem is to extract functional requirements, non-functional requirements, constraints, assumptions, open questions, and follow-up questions with evidence links to utterance IDs while minimizing hallucination. The methodology uses a modular pipeline: utterance parsing, model-based extraction, schema normalization/validation (Pydantic), post-processing, and quantitative evaluation across model backends. Hugging Face models and mock mode are supported through a common runner interface. To increase robustness under weak models, the system applies compact prompts for short-context models, retry/repair logic, LLM-based utterance rescue when chunk-level generation fails, and sanity reclassification for obvious mislabels (for example, question-like text, uncertainty statements, and hard constraints). Evaluation reports micro precision/recall/F1 for requirement extraction, type macro-F1, evidence scores, hallucination rate, open-question recall, schema validity rates, and latency. Current baseline evaluation (mock model) shows strong schema validity but poor extraction quality, indicating that model capability, prompt discipline, and dataset quality remain dominant factors. The latest pipeline revisions significantly improve practical categorization reliability on observed failure cases (especially with `flan-t5-small`) while preserving explainable traceability via evidence IDs and decision logs. The main conclusion is that robust structured extraction requires both LLM-centered reasoning and strict runtime guardrails.

## Introduction
Client conversations are often informal, ambiguous, and incomplete. Junior PMs and student team leaders frequently miss requirements or misinterpret vague statements, causing scope drift and rework.

This project addresses that issue with an automated “Conversation-to-Spec” pipeline. The objective is to produce a structured draft from raw transcript text, preserve traceability through utterance evidence (`U1`, `U2`, ...), and separate uncertain statements into assumptions/open questions instead of hallucinating hard requirements.

Primary objectives:
1. Build a modular Python CLI pipeline with pluggable model backends.
2. Enforce machine-parseable structured outputs via Pydantic schemas.
3. Support model comparison and quantitative evaluation.
4. Record and learn from real failure modes through detailed trial-and-error logs.

## Methods
### 1) Data preprocessing
- Input format: `.txt` / `.md` conversation files.
- Parsing strategy:
  - Speaker-prefixed lines (`Client:`, `PM:`, `User:`, `Developer:`, etc.) are parsed directly.
  - Unlabeled lines are role-inferred (`Client` / `PM` / `Unknown`) with context heuristics.
- Each utterance is assigned an ID (`U1...Un`) for evidence traceability.

### 2) Models used
- `google/flan-t5-small` (short context, low cost, unstable structured generation)
- `microsoft/Phi-3.5-mini-instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-0.5B-Instruct`
- `mock` (deterministic test baseline)

### 3) Experimental pipeline
1. Parse transcript into utterances.
2. Chunk utterances by token budget.
3. Run chunk-level structured extraction prompt.
4. Retry with repair prompts if invalid output.
5. Merge chunk results.
6. LLM-driven ambiguity triage (`question_decisions`: `needs_follow_up`, `already_asked`, `resolved`).
7. Optional consolidation pass.
8. If chunk extraction totally fails, run LLM utterance-level rescue.
9. Validate output using Pydantic and export JSON + Markdown.

### 4) Analysis and metrics
- Requirement extraction precision/recall/F1 (micro; normalized and exact).
- Requirement type macro-F1.
- Evidence linking F1 and Jaccard.
- Hallucination rate.
- Open question recall.
- JSON parse success and Pydantic validation success.
- Inference latency and token statistics.

### 5) GitHub repository
- Repository link: `https://github.com/<YOUR_ACCOUNT>/<YOUR_REPO>`
- If private: invite collaborator `ssuai`.

## Results
### Quantitative baseline (current saved eval artifacts)
Source: `eval_output/comparison_table.md`, `eval_output/comparison_results.json`.

| Model | Req F1 (Norm) | Type Macro-F1 | Evidence F1 | Hallucination Rate | Open Q Recall | JSON Validity | Pydantic Validity |
|---|---:|---:|---:|---:|---:|---:|---:|
| mock | 0.036 | 0.024 | 0.018 | 0.963 | 0.000 | 1.000 | 1.000 |

Interpretation:
- Schema validity is excellent in mock mode.
- Semantic quality is poor, confirming the need for stronger model behavior and stricter prompt/runtime controls.

### Practical behavior after robustness updates
Observed on `sample_conversation_2.txt` with `flan-t5-small`:
- Functional requirements: 3
- Non-functional requirements: 1
- Constraints: 1
- Assumptions: 1
- Open questions: 1
- Question decisions included (`resolved` for already asked-and-answered PM question).

This addresses earlier misclassification where question/constraint/uncertainty lines were incorrectly grouped as non-functional requirements.

## Discussion
The project demonstrates that structured extraction is not only a prompting problem but a reliability engineering problem. Small models often fail with long schema prompts, produce partial tokens, or return malformed payloads. Relying on single-pass generation is therefore fragile.

The most effective changes were:
1. Compact prompt profile for short-context models.
2. Forced LLM rescue path on total chunk failure.
3. Sanity correction for obvious category errors.
4. Explicit question-decision modeling to avoid redundant follow-up questions.

Limitations remain:
- Small models still have weak semantic precision.
- Current evaluation set is limited and should be expanded.
- Semantic matching in evaluation is mostly exact/normalized; paraphrase-aware scoring is needed.

## Conclusion
This prototype successfully implements a Python-only CLI pipeline for Conversation-to-Spec generation with schema validation, evidence linking, and quantitative evaluation support. The work highlights a key lesson: robust requirement extraction requires both LLM reasoning and strict runtime safeguards. Recent updates improved failure recovery and category correctness in realistic failure scenarios, especially for weaker models. Future work should focus on stronger local models, larger annotated datasets, and deeper evaluation for ambiguity handling and evidence quality.

## References
1. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
2. Hugging Face Transformers documentation: https://huggingface.co/docs/transformers
3. Pydantic documentation: https://docs.pydantic.dev/
4. Project repository code and evaluation artifacts (`app/`, `tests/`, `eval_output/`).

---

## Detailed Trial-and-Error Log (Past)

### Log Entry 1
- Symptom: `Token indices sequence length is longer than the specified maximum sequence length for this model (2883 > 512)`.
- Root cause: Prompt + schema + conversation exceeded `flan-t5-small` context capacity.
- Action: Added chunking and later compact prompt profile for short-context models.
- Outcome: Prompt token usage dropped (e.g., ~258 tokens in compact mode), reducing immediate truncation failures.

### Log Entry 2
- Symptom: `The following generation flags are not valid and may be ignored: ['temperature']`.
- Root cause: Generation config mismatch for certain model/task combinations.
- Action: Kept deterministic generation strategy and monitored effective flags.
- Outcome: Warning acknowledged; not a hard failure but tracked as configuration debt.

### Log Entry 3
- Symptom: `MPS backend out of memory`.
- Root cause: Large model + long generation attempts on MPS memory budget.
- Action: Shifted to smaller/compact prompting, reduced context pressure, retried with safer settings.
- Outcome: Reduced OOM frequency during iterative runs.

### Log Entry 4
- Symptom: Repeated parse failure with empty/partial outputs, including `needs_follow_up>` only.
- Root cause: Weak model output collapse under strict/long schema prompt.
- Action: Added compact prompt profile and all-chunk-failure rescue via utterance-level LLM calls.
- Outcome: Pipeline now recovers and produces valid `spec.json` instead of hard-failing.

### Log Entry 5
- Symptom: Schema validation errors like missing `text` fields in map-style outputs.
- Root cause: Model returned key-value buckets instead of required item objects.
- Action: Strengthened extractor normalization for map/list/string variants and evidence inference.
- Outcome: More malformed outputs become recoverable normalized payloads.

### Log Entry 6
- Symptom: Outputs with zero extracted items (`None` sections only).
- Root cause: Weak generation quality and strict parsing failures.
- Action: Added utterance-level rescue classification pass and merge logic.
- Outcome: Non-empty structured outputs are frequently recoverable.

### Log Entry 7
- Symptom: Misclassification of question/constraint/uncertainty lines as non-functional requirements.
- Root cause: Small-model category drift.
- Action: Added sanity postprocess:
  - question-like text -> `open_questions`
  - uncertainty (`maybe`, `later`) -> `assumptions`
  - hard policy/permission constraints -> `constraints`
- Outcome: Category quality improved on observed failure cases.

### Log Entry 8
- Symptom: Redundant follow-up questions even when PM already asked and client answered.
- Root cause: Missing explicit decision model for follow-up necessity.
- Action: Added `question_decisions` schema and decision-aware follow-up filtering.
- Outcome: `resolved/already_asked` items no longer generate redundant follow-up questions.

### Log Entry 9
- Symptom: Generic fallback follow-up question appeared even when not needed.
- Root cause: Automatic fallback follow-up generation was always active.
- Action: Added `auto_follow_up_fallback` control; default disabled for LLM-first mode.
- Outcome: Follow-up list now primarily reflects model decisions (`needs_follow_up`).

### Log Entry 10
- Symptom: Role labels missing in unlabeled transcripts.
- Root cause: Original parser assumed explicit speaker prefixes.
- Action: Added role inference for unlabeled lines and alias normalization (including Korean labels like `개발자`, `고객`).
- Outcome: Improved speaker-role separation and downstream interpretation quality.

---

## Ongoing Trial-and-Error Logging Policy (Future)
From now on, append every significant failure under this section using the template below.

### Required logging rules
1. Record every failure run with timestamp and model name.
2. Include exact error message (copy-paste), not paraphrase only.
3. Document root cause hypothesis and verification method.
4. Record patch location (file + line or function).
5. Record measurable effect (success/fail, latency, metric delta).

### Future Log Template
```markdown
### Log Entry <N>
- Date/Time:
- Model:
- Command:
- Symptom (exact message):
- Root cause hypothesis:
- Verification steps:
- Code changes applied:
- Outcome after patch:
- Remaining risk:
```

### Next Empty Entry Slot
### Log Entry <N+1>
- Date/Time:
- Model:
- Command:
- Symptom (exact message):
- Root cause hypothesis:
- Verification steps:
- Code changes applied:
- Outcome after patch:
- Remaining risk:
