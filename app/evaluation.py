from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from app.parser import parse_conversation_text
from app.pipeline import PipelineDiagnostics, generate_spec_from_utterances
from app.schemas import SpecOutput
from app.utils import compute_prf, ensure_directory, mean, normalize_text, safe_divide, slugify

CATEGORY_FUNCTIONAL = "functional"
CATEGORY_NON_FUNCTIONAL = "non_functional"
CATEGORY_CONSTRAINT = "constraint"
CATEGORY_ASSUMPTION = "assumption"
CATEGORY_OPEN_QUESTION = "open_question"

ALL_CATEGORIES = [
    CATEGORY_FUNCTIONAL,
    CATEGORY_NON_FUNCTIONAL,
    CATEGORY_CONSTRAINT,
    CATEGORY_ASSUMPTION,
    CATEGORY_OPEN_QUESTION,
]


class EvalSample(BaseModel):
    sample_id: str
    conversation: str
    gold: SpecOutput


@dataclass
class EvaluationCounts:
    req_tp: int = 0
    req_fp: int = 0
    req_fn: int = 0

    req_exact_tp: int = 0
    req_exact_fp: int = 0
    req_exact_fn: int = 0

    evidence_tp: int = 0
    evidence_fp: int = 0
    evidence_fn: int = 0

    evidence_jaccard_sum: float = 0.0
    evidence_jaccard_count: int = 0

    hallucinated_items: int = 0
    total_predicted_requirements: int = 0

    open_question_captured: int = 0
    open_question_gold_total: int = 0


def _flatten_spec_items(spec: SpecOutput) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    for item in spec.functional_requirements:
        items.append(
            {
                "category": CATEGORY_FUNCTIONAL,
                "text": item.text,
                "evidence": item.evidence,
            }
        )
    for item in spec.non_functional_requirements:
        items.append(
            {
                "category": CATEGORY_NON_FUNCTIONAL,
                "text": item.text,
                "evidence": item.evidence,
            }
        )
    for item in spec.constraints:
        items.append(
            {
                "category": CATEGORY_CONSTRAINT,
                "text": item.text,
                "evidence": item.evidence,
            }
        )
    for item in spec.assumptions:
        items.append(
            {
                "category": CATEGORY_ASSUMPTION,
                "text": item.text,
                "evidence": item.evidence,
            }
        )
    for item in spec.open_questions:
        items.append(
            {
                "category": CATEGORY_OPEN_QUESTION,
                "text": item.text,
                "evidence": item.evidence,
            }
        )

    return items


def _requirement_text_set(spec: SpecOutput, match_mode: str = "normalized") -> set[str]:
    texts = [item.text for item in (spec.functional_requirements + spec.non_functional_requirements)]

    output: set[str] = set()
    for text in texts:
        raw = text.strip()
        if not raw:
            continue

        if match_mode == "exact":
            output.add(raw)
        elif match_mode == "normalized":
            normalized = normalize_text(raw)
            if normalized:
                output.add(normalized)
        else:
            raise ValueError(f"Unsupported match mode: {match_mode}")

    return output


def _text_to_category_map(spec: SpecOutput, match_mode: str = "normalized") -> dict[str, str]:
    text_to_category: dict[str, str] = {}
    for item in _flatten_spec_items(spec):
        raw = str(item["text"]).strip()
        if not raw:
            continue

        key = raw if match_mode == "exact" else normalize_text(raw)
        if not key:
            continue

        text_to_category.setdefault(key, item["category"])
    return text_to_category


def _text_to_evidence_map(spec: SpecOutput, match_mode: str = "normalized") -> dict[tuple[str, str], set[str]]:
    evidence_map: dict[tuple[str, str], set[str]] = {}
    for item in _flatten_spec_items(spec):
        raw = str(item["text"]).strip()
        key_text = raw if match_mode == "exact" else normalize_text(raw)
        if not key_text:
            continue

        key = (item["category"], key_text)
        evidence_map[key] = set(item["evidence"])
    return evidence_map


def compute_requirement_extraction_counts(
    pred: SpecOutput,
    gold: SpecOutput,
    match_mode: str = "normalized",
) -> tuple[int, int, int]:
    pred_set = _requirement_text_set(pred, match_mode=match_mode)
    gold_set = _requirement_text_set(gold, match_mode=match_mode)

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


def compute_type_classification_counts(pred: SpecOutput, gold: SpecOutput) -> dict[str, dict[str, int]]:
    pred_map = _text_to_category_map(pred, match_mode="normalized")
    gold_map = _text_to_category_map(gold, match_mode="normalized")
    all_texts = set(pred_map) | set(gold_map)

    counts: dict[str, dict[str, int]] = {
        category: {"tp": 0, "fp": 0, "fn": 0} for category in ALL_CATEGORIES
    }

    for text_key in all_texts:
        pred_label = pred_map.get(text_key)
        gold_label = gold_map.get(text_key)

        for category in ALL_CATEGORIES:
            if pred_label == category and gold_label == category:
                counts[category]["tp"] += 1
            elif pred_label == category and gold_label != category:
                counts[category]["fp"] += 1
            elif gold_label == category and pred_label != category:
                counts[category]["fn"] += 1

    return counts


def compute_evidence_linking_counts(pred: SpecOutput, gold: SpecOutput) -> tuple[int, int, int]:
    pred_map = _text_to_evidence_map(pred, match_mode="normalized")
    gold_map = _text_to_evidence_map(gold, match_mode="normalized")
    keys = set(pred_map) | set(gold_map)

    tp = fp = fn = 0
    for key in keys:
        pred_evidence = pred_map.get(key, set())
        gold_evidence = gold_map.get(key, set())

        tp += len(pred_evidence & gold_evidence)
        fp += len(pred_evidence - gold_evidence)
        fn += len(gold_evidence - pred_evidence)

    return tp, fp, fn


def compute_evidence_jaccard(pred: SpecOutput, gold: SpecOutput) -> tuple[float, int]:
    pred_map = _text_to_evidence_map(pred, match_mode="normalized")
    gold_map = _text_to_evidence_map(gold, match_mode="normalized")
    keys = set(pred_map) | set(gold_map)

    if not keys:
        return 0.0, 0

    score_sum = 0.0
    count = 0
    for key in keys:
        pred_evidence = pred_map.get(key, set())
        gold_evidence = gold_map.get(key, set())
        union = pred_evidence | gold_evidence
        if not union:
            continue
        score_sum += len(pred_evidence & gold_evidence) / len(union)
        count += 1

    return score_sum, count


def compute_hallucination_counts(pred: SpecOutput, gold: SpecOutput) -> tuple[int, int]:
    pred_set = _requirement_text_set(pred, match_mode="normalized")
    gold_set = _requirement_text_set(gold, match_mode="normalized")
    unsupported = len(pred_set - gold_set)
    return unsupported, len(pred_set)


def compute_open_question_recall_counts(pred: SpecOutput, gold: SpecOutput) -> tuple[int, int]:
    pred_set = {normalize_text(item.text) for item in pred.open_questions if normalize_text(item.text)}
    gold_set = {normalize_text(item.text) for item in gold.open_questions if normalize_text(item.text)}
    captured = len(pred_set & gold_set)
    return captured, len(gold_set)


def _empty_prediction_like(utterances: list[dict[str, str]]) -> SpecOutput:
    return SpecOutput(
        project_summary="Model output was invalid.",
        functional_requirements=[],
        non_functional_requirements=[],
        constraints=[],
        assumptions=[],
        open_questions=[],
        follow_up_questions=[],
        utterances=utterances,
    )


def _classify_error(error_text: str | None) -> str:
    if not error_text:
        return "none"

    lowered = error_text.lower()
    if "json" in lowered:
        return "json_error"
    if "validation" in lowered or "pydantic" in lowered:
        return "schema_error"
    if "out of memory" in lowered or "oom" in lowered:
        return "oom"
    if "generation failed" in lowered:
        return "generation_failure"
    return "other"


def _failure_tags(
    pred_spec: SpecOutput,
    gold_spec: SpecOutput,
    diagnostics: PipelineDiagnostics,
    extraction_error: str | None,
) -> list[str]:
    tags: list[str] = []

    if not diagnostics.first_pass_json_parse_ok:
        tags.append("json_parse_retry_needed")
    if not diagnostics.first_pass_pydantic_validation_ok:
        tags.append("schema_retry_needed")
    if extraction_error:
        tags.append(_classify_error(extraction_error))

    unsupported, total_predicted = compute_hallucination_counts(pred_spec, gold_spec)
    hallucination_rate = safe_divide(unsupported, total_predicted)
    if hallucination_rate >= 0.5 and total_predicted > 0:
        tags.append("high_hallucination")

    open_captured, open_total = compute_open_question_recall_counts(pred_spec, gold_spec)
    if open_total > 0 and open_captured == 0:
        tags.append("open_question_miss")

    if not pred_spec.functional_requirements and not pred_spec.non_functional_requirements:
        tags.append("no_requirements_predicted")

    deduped: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        deduped.append(tag)
    return deduped


def load_eval_dataset(dataset_path: str | Path) -> list[EvalSample]:
    path = Path(dataset_path)
    raw_payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, list):
        raise ValueError("Evaluation dataset root must be a list of samples")

    samples: list[EvalSample] = []
    for item in raw_payload:
        samples.append(EvalSample.model_validate(item))
    return samples


def evaluate_model_on_dataset(
    dataset_path: str | Path,
    runner,
    prompt_config: dict,
    eval_output_dir: str | Path,
    model_label: str,
) -> dict[str, Any]:
    samples = load_eval_dataset(dataset_path)
    model_slug = slugify(model_label)
    base_output_dir = ensure_directory(Path(eval_output_dir) / model_slug)
    prediction_dir = ensure_directory(base_output_dir / "predictions")

    counts = EvaluationCounts()
    type_counts: dict[str, dict[str, int]] = {
        category: {"tp": 0, "fp": 0, "fn": 0} for category in ALL_CATEGORIES
    }

    parse_success_count = 0
    validation_success_count = 0
    first_pass_parse_success_count = 0
    first_pass_validation_success_count = 0

    failure_cluster_counts: dict[str, int] = {}

    sample_results: list[dict[str, Any]] = []
    latencies: list[float] = []
    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []

    for sample in samples:
        parsed_utterances = parse_conversation_text(sample.conversation)
        utterance_dicts = [utterance.model_dump() for utterance in parsed_utterances]

        raw_output = ""
        extraction_error = None

        try:
            pred_spec, generation, diagnostics = generate_spec_from_utterances(
                utterance_dicts=utterance_dicts,
                runner=runner,
                prompt_config=prompt_config,
            )
            raw_output = generation.raw_text
            latencies.append(generation.latency_sec)
            if generation.prompt_tokens is not None:
                prompt_tokens.append(generation.prompt_tokens)
            if generation.completion_tokens is not None:
                completion_tokens.append(generation.completion_tokens)
        except Exception as exc:  # noqa: BLE001
            pred_spec = _empty_prediction_like(utterance_dicts)
            diagnostics = PipelineDiagnostics(
                json_parse_ok=False,
                pydantic_validation_ok=False,
                first_pass_json_parse_ok=False,
                first_pass_pydantic_validation_ok=False,
                errors=[str(exc)],
            )
            extraction_error = f"Generation failed: {exc}"
            generation = None

        json_ok = diagnostics.json_parse_ok
        validation_ok = diagnostics.pydantic_validation_ok
        first_json_ok = diagnostics.first_pass_json_parse_ok
        first_validation_ok = diagnostics.first_pass_pydantic_validation_ok

        if json_ok:
            parse_success_count += 1
        if validation_ok:
            validation_success_count += 1
        if first_json_ok:
            first_pass_parse_success_count += 1
        if first_validation_ok:
            first_pass_validation_success_count += 1

        gold_spec = sample.gold

        req_tp, req_fp, req_fn = compute_requirement_extraction_counts(pred_spec, gold_spec, match_mode="normalized")
        counts.req_tp += req_tp
        counts.req_fp += req_fp
        counts.req_fn += req_fn

        req_exact_tp, req_exact_fp, req_exact_fn = compute_requirement_extraction_counts(
            pred_spec,
            gold_spec,
            match_mode="exact",
        )
        counts.req_exact_tp += req_exact_tp
        counts.req_exact_fp += req_exact_fp
        counts.req_exact_fn += req_exact_fn

        evidence_tp, evidence_fp, evidence_fn = compute_evidence_linking_counts(pred_spec, gold_spec)
        counts.evidence_tp += evidence_tp
        counts.evidence_fp += evidence_fp
        counts.evidence_fn += evidence_fn

        jaccard_sum, jaccard_count = compute_evidence_jaccard(pred_spec, gold_spec)
        counts.evidence_jaccard_sum += jaccard_sum
        counts.evidence_jaccard_count += jaccard_count

        unsupported, predicted_total = compute_hallucination_counts(pred_spec, gold_spec)
        counts.hallucinated_items += unsupported
        counts.total_predicted_requirements += predicted_total

        open_captured, open_total = compute_open_question_recall_counts(pred_spec, gold_spec)
        counts.open_question_captured += open_captured
        counts.open_question_gold_total += open_total

        sample_type_counts = compute_type_classification_counts(pred_spec, gold_spec)
        for category in ALL_CATEGORIES:
            type_counts[category]["tp"] += sample_type_counts[category]["tp"]
            type_counts[category]["fp"] += sample_type_counts[category]["fp"]
            type_counts[category]["fn"] += sample_type_counts[category]["fn"]

        req_precision, req_recall, req_f1 = compute_prf(req_tp, req_fp, req_fn)

        failure_tags = _failure_tags(
            pred_spec=pred_spec,
            gold_spec=gold_spec,
            diagnostics=diagnostics,
            extraction_error=extraction_error,
        )
        for tag in failure_tags:
            failure_cluster_counts[tag] = failure_cluster_counts.get(tag, 0) + 1

        sample_record = {
            "sample_id": sample.sample_id,
            "json_parse_ok": json_ok,
            "pydantic_validation_ok": validation_ok,
            "first_pass_json_parse_ok": first_json_ok,
            "first_pass_pydantic_validation_ok": first_validation_ok,
            "error": extraction_error,
            "requirement_precision": req_precision,
            "requirement_recall": req_recall,
            "requirement_f1": req_f1,
            "hallucination_rate": safe_divide(unsupported, predicted_total),
            "failure_tags": failure_tags,
            "latency_sec": generation.latency_sec if generation is not None else None,
            "chunks_used": diagnostics.chunks_used,
            "retries_used": diagnostics.retries_used,
        }
        sample_results.append(sample_record)

        sample_slug = slugify(sample.sample_id)
        (prediction_dir / f"{sample_slug}_raw.txt").write_text(raw_output, encoding="utf-8")
        (prediction_dir / f"{sample_slug}_pred.json").write_text(
            json.dumps(pred_spec.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    req_precision, req_recall, req_f1 = compute_prf(counts.req_tp, counts.req_fp, counts.req_fn)
    req_exact_precision, req_exact_recall, req_exact_f1 = compute_prf(
        counts.req_exact_tp,
        counts.req_exact_fp,
        counts.req_exact_fn,
    )

    category_f1_scores: dict[str, float] = {}
    for category in ALL_CATEGORIES:
        _, _, category_f1 = compute_prf(
            type_counts[category]["tp"],
            type_counts[category]["fp"],
            type_counts[category]["fn"],
        )
        category_f1_scores[category] = category_f1

    type_macro_f1 = mean(category_f1_scores.values())

    evidence_precision, evidence_recall, evidence_f1 = compute_prf(
        counts.evidence_tp,
        counts.evidence_fp,
        counts.evidence_fn,
    )
    evidence_jaccard = safe_divide(counts.evidence_jaccard_sum, counts.evidence_jaccard_count)

    hallucination_rate = safe_divide(
        counts.hallucinated_items,
        counts.total_predicted_requirements,
    )
    open_question_recall = safe_divide(
        counts.open_question_captured,
        counts.open_question_gold_total,
    )

    aggregate_metrics: dict[str, Any] = {
        "requirement_precision": req_precision,
        "requirement_recall": req_recall,
        "requirement_f1": req_f1,
        "requirement_exact_precision": req_exact_precision,
        "requirement_exact_recall": req_exact_recall,
        "requirement_exact_f1": req_exact_f1,
        "type_classification_macro_f1": type_macro_f1,
        "type_f1_by_category": category_f1_scores,
        "evidence_precision": evidence_precision,
        "evidence_recall": evidence_recall,
        "evidence_f1": evidence_f1,
        "evidence_jaccard": evidence_jaccard,
        "hallucination_rate": hallucination_rate,
        "open_question_recall": open_question_recall,
        "json_parse_success_rate": safe_divide(parse_success_count, len(samples)),
        "pydantic_validation_success_rate": safe_divide(validation_success_count, len(samples)),
        "first_pass_json_parse_success_rate": safe_divide(first_pass_parse_success_count, len(samples)),
        "first_pass_pydantic_validation_success_rate": safe_divide(first_pass_validation_success_count, len(samples)),
        "avg_latency_sec": mean(latencies),
        "avg_prompt_tokens": mean(prompt_tokens) if prompt_tokens else None,
        "avg_completion_tokens": mean(completion_tokens) if completion_tokens else None,
        "num_samples": len(samples),
        "failure_clusters": failure_cluster_counts,
    }

    result_payload = {
        "model": model_label,
        "aggregate_metrics": aggregate_metrics,
        "per_sample": sample_results,
    }

    (base_output_dir / "metrics.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return result_payload


def render_comparison_table(model_results: list[dict[str, Any]]) -> str:
    if not model_results:
        return "No evaluation results available."

    lines = [
        "| Model | Req F1 (Norm) | Req F1 (Exact) | Type Macro-F1 | Evidence F1 | Evidence Jaccard | Hallucination Rate | Open Q Recall | JSON Validity | Pydantic Validity | Avg Latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for result in model_results:
        metrics = result.get("aggregate_metrics", {})
        lines.append(
            "| {model} | {req_f1:.3f} | {req_exact_f1:.3f} | {type_f1:.3f} | {evidence_f1:.3f} | {evidence_jaccard:.3f} | {hallucination:.3f} | {open_recall:.3f} | {json_valid:.3f} | {pydantic_valid:.3f} | {latency:.3f} |".format(
                model=result.get("model", "unknown"),
                req_f1=metrics.get("requirement_f1", 0.0),
                req_exact_f1=metrics.get("requirement_exact_f1", 0.0),
                type_f1=metrics.get("type_classification_macro_f1", 0.0),
                evidence_f1=metrics.get("evidence_f1", 0.0),
                evidence_jaccard=metrics.get("evidence_jaccard", 0.0),
                hallucination=metrics.get("hallucination_rate", 0.0),
                open_recall=metrics.get("open_question_recall", 0.0),
                json_valid=metrics.get("json_parse_success_rate", 0.0),
                pydantic_valid=metrics.get("pydantic_validation_success_rate", 0.0),
                latency=metrics.get("avg_latency_sec", 0.0),
            )
        )

    return "\n".join(lines)


def load_gold_spec_for_validation(sample_payload: dict[str, Any]) -> SpecOutput:
    try:
        return SpecOutput.model_validate(sample_payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid gold spec payload: {exc}") from exc
