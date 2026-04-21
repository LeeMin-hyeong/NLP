from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import json
import os
import re
import sys
import threading
import time
from typing import Optional

from app.prompt_builder import build_prompt
from app.utils import normalize_text


@dataclass
class GenerationResult:
    model_name: str
    raw_text: str
    latency_sec: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


@dataclass
class ClassificationResult:
    label: str
    score: float
    scores: dict[str, float]
    latency_sec: float


class BaseModelRunner(ABC):
    def __init__(self, runner_name: str) -> None:
        self.runner_name = runner_name

    @abstractmethod
    def generate_structured_output(self, utterances: list[dict], prompt_config: dict) -> str:
        """Generate raw model text intended to be valid JSON."""

    def generate_with_metadata(self, utterances: list[dict], prompt_config: dict) -> GenerationResult:
        start_time = time.perf_counter()
        raw_text = self.generate_structured_output(utterances=utterances, prompt_config=prompt_config)
        latency_sec = time.perf_counter() - start_time
        return GenerationResult(
            model_name=self.runner_name,
            raw_text=raw_text,
            latency_sec=latency_sec,
        )

    def get_input_token_budget(self, prompt_config: dict) -> int:
        user_max_input_tokens = prompt_config.get("max_input_tokens")
        if isinstance(user_max_input_tokens, int) and user_max_input_tokens > 0:
            return user_max_input_tokens
        return 2048

    @staticmethod
    def _progress_enabled(prompt_config: dict) -> bool:
        return bool(prompt_config.get("progress_logging", True))

    @staticmethod
    def _inline_progress_enabled(prompt_config: dict) -> bool:
        return bool(prompt_config.get("single_line_progress", True))

    def _progress_log(self, prompt_config: dict, message: str) -> None:
        if not self._progress_enabled(prompt_config):
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{self.runner_name}] {message}", flush=True)


class BaseClassificationRunner(ABC):
    def __init__(self, runner_name: str) -> None:
        self.runner_name = runner_name

    @abstractmethod
    def classify(self, text: str, labels: list[str]) -> ClassificationResult:
        """Classify text into one of labels."""


class MockClassificationRunner(BaseClassificationRunner):
    def __init__(self) -> None:
        super().__init__(runner_name="mock-classifier")

    def classify(self, text: str, labels: list[str]) -> ClassificationResult:
        start = time.perf_counter()
        lowered = text.lower()
        candidate_scores: dict[str, float] = {label: 0.01 for label in labels}

        def add_if_contains(label_key: str, hints: tuple[str, ...], boost: float) -> None:
            for label in labels:
                if label_key not in label.lower():
                    continue
                if any(hint in lowered for hint in hints):
                    candidate_scores[label] += boost

        add_if_contains("functional", ("must", "need", "allow", "create", "update", "join", "publish"), 0.9)
        add_if_contains("non", ("easy to use", "simple", "fast", "user-friendly", "secure", "reliable"), 0.9)
        add_if_contains("constraint", ("deadline", "budget", "only", "must use", "cannot", "can't"), 0.9)
        add_if_contains("assumption", ("maybe", "later", "for now", "not decided", "if possible"), 0.9)
        add_if_contains("question", ("?", "clarify", "who", "what", "when", "how", "which"), 0.9)

        best_label = max(labels, key=lambda label: candidate_scores.get(label, 0.0))
        total = sum(candidate_scores.values()) or 1.0
        normalized = {label: score / total for label, score in candidate_scores.items()}
        score = normalized.get(best_label, 0.0)
        return ClassificationResult(
            label=best_label,
            score=score,
            scores=normalized,
            latency_sec=time.perf_counter() - start,
        )


class HuggingFaceZeroShotClassificationRunner(BaseClassificationRunner):
    def __init__(self, model_name: str) -> None:
        super().__init__(runner_name=model_name)
        self.model_name = model_name
        self._classifier = None

    def _load(self) -> None:
        if self._classifier is not None:
            return

        import torch
        from transformers import pipeline

        device = -1
        if torch.cuda.is_available():
            device = 0
        # MPS can be unstable in zero-shot pipeline across versions; keep CPU fallback for consistency.
        self._classifier = pipeline(
            task="zero-shot-classification",
            model=self.model_name,
            device=device,
        )

    def classify(self, text: str, labels: list[str]) -> ClassificationResult:
        self._load()
        if self._classifier is None:
            raise RuntimeError("Zero-shot classifier not initialized")

        start = time.perf_counter()
        payload = self._classifier(
            sequences=text,
            candidate_labels=labels,
            multi_label=False,
        )
        latency = time.perf_counter() - start

        ranked_labels = payload.get("labels", []) or []
        ranked_scores = payload.get("scores", []) or []
        if not ranked_labels:
            fallback_label = labels[0] if labels else "functional requirement"
            return ClassificationResult(label=fallback_label, score=0.0, scores={}, latency_sec=latency)

        scores = {
            str(label): float(score)
            for label, score in zip(ranked_labels, ranked_scores)
        }
        top_label = str(ranked_labels[0])
        top_score = float(ranked_scores[0]) if ranked_scores else 0.0
        return ClassificationResult(
            label=top_label,
            score=top_score,
            scores=scores,
            latency_sec=latency,
        )


class MockModelRunner(BaseModelRunner):
    FUNCTIONAL_HINTS = (
        "need",
        "needs",
        "must",
        "should",
        "allow",
        "allows",
        "can",
        "want",
        "wants",
        "create",
        "update",
        "edit",
        "delete",
        "book",
        "reserve",
        "schedule",
        "manage",
        "view",
        "track",
        "notify",
        "reminder",
        "login",
        "sign in",
        "post",
        "publish",
        "request",
        "approve",
    )
    NFR_HINTS = (
        "fast",
        "easy to use",
        "simple",
        "clean",
        "modern",
        "user-friendly",
        "secure",
        "reliable",
        "responsive",
        "performance",
        "few seconds",
    )
    CONSTRAINT_HINTS = (
        "budget",
        "deadline",
        "must use",
        "cannot",
        "can't",
        "within",
        "only",
        "web only",
        "on-prem",
        "on prem",
        "school servers",
        "three month",
        "3 month",
        "not ios",
        "not android",
    )
    UNCERTAINTY_HINTS = (
        "maybe",
        "later",
        "nice to have",
        "not decided",
        "not decided yet",
        "if possible",
        "for now",
        "would be good",
    )

    def __init__(self) -> None:
        super().__init__(runner_name="mock")

    @staticmethod
    def _contains_any(text: str, hints: tuple[str, ...]) -> bool:
        return any(token in text for token in hints)

    @staticmethod
    def _default_question_from_note(note: str) -> str:
        stem = note.strip().rstrip(".")
        if stem.endswith("?"):
            return stem
        return f"Can you clarify: {stem}?"

    def generate_structured_output(self, utterances: list[dict], prompt_config: dict) -> str:
        ambiguous_hints = tuple(token.lower() for token in prompt_config.get("ambiguous_phrases", []))

        functional_items: list[dict] = []
        non_functional_items: list[dict] = []
        constraints: list[dict] = []
        assumptions: list[dict] = []
        open_questions: list[dict] = []
        follow_up_questions: list[str] = []
        question_decisions: list[dict] = []

        dedupe_guard: set[tuple[str, str]] = set()
        question_dedupe_guard: set[str] = set()

        def append_item(bucket_name: str, bucket: list[dict], text: str, evidence_id: str) -> None:
            key = (bucket_name, normalize_text(text))
            if key in dedupe_guard:
                return
            dedupe_guard.add(key)
            bucket.append({"text": text.strip(), "evidence": [evidence_id]})

        def append_question_decision(
            text: str,
            decision: str,
            evidence_id: str,
            suggested_follow_up: str | None = None,
        ) -> None:
            key = f"{decision}|{normalize_text(text)}"
            if not text.strip() or key in question_dedupe_guard:
                return
            question_dedupe_guard.add(key)
            question_decisions.append(
                {
                    "text": text.strip(),
                    "decision": decision,
                    "suggested_follow_up": suggested_follow_up,
                    "evidence": [evidence_id],
                }
            )

        for utterance in utterances:
            text = utterance["text"].strip()
            lower_text = text.lower()
            evidence_id = utterance["id"]
            speaker = str(utterance.get("speaker", "")).strip().lower()

            if not text:
                continue

            if "?" in text:
                append_item(
                    "open_question",
                    open_questions,
                    text,
                    evidence_id,
                )
                if speaker in {"pm", "developer", "dev", "engineer", "team lead", "assistant", "analyst", "개발자"}:
                    append_question_decision(
                        text=text,
                        decision="already_asked",
                        evidence_id=evidence_id,
                    )
                else:
                    append_question_decision(
                        text=text,
                        decision="needs_follow_up",
                        evidence_id=evidence_id,
                        suggested_follow_up=text,
                    )
                continue

            if self._contains_any(lower_text, self.CONSTRAINT_HINTS):
                append_item("constraint", constraints, text, evidence_id)

            if self._contains_any(lower_text, self.UNCERTAINTY_HINTS):
                append_item("assumption", assumptions, text, evidence_id)
                append_question_decision(
                    text=text,
                    decision="needs_follow_up",
                    evidence_id=evidence_id,
                    suggested_follow_up=f"Can you clarify: {text.rstrip('.')}?",
                )

            ambiguous_detected = self._contains_any(lower_text, ambiguous_hints) if ambiguous_hints else False
            if self._contains_any(lower_text, self.NFR_HINTS):
                if ambiguous_detected and not re.search(r"\d", lower_text):
                    append_item(
                        "open_question",
                        open_questions,
                        f"Ambiguous quality target needs clarification: {text}",
                        evidence_id,
                    )
                    append_question_decision(
                        text=text,
                        decision="needs_follow_up",
                        evidence_id=evidence_id,
                        suggested_follow_up=f"How should we measure this quality requirement: {text.rstrip('.')}?",
                    )
                elif "easy" in lower_text or "simple" in lower_text or "user-friendly" in lower_text:
                    append_question_decision(
                        text=text,
                        decision="needs_follow_up",
                        evidence_id=evidence_id,
                        suggested_follow_up=f"What measurable acceptance criteria define: {text.rstrip('.')}?",
                    )
                append_item("non_functional", non_functional_items, text, evidence_id)
                continue

            if self._contains_any(lower_text, self.FUNCTIONAL_HINTS):
                append_item("functional", functional_items, text, evidence_id)

        for idx, item in enumerate(functional_items, start=1):
            item["id"] = f"FR{idx}"
            item["priority"] = "medium"

        for idx, item in enumerate(non_functional_items, start=1):
            item["id"] = f"NFR{idx}"
            item["priority"] = "medium"

        if not follow_up_questions:
            for note in open_questions[:5]:
                follow_up_questions.append(self._default_question_from_note(note["text"]))

        if not follow_up_questions and assumptions:
            follow_up_questions.append("Which assumed items should be in scope for the first release?")

        if not follow_up_questions:
            for item in question_decisions:
                if item["decision"] != "needs_follow_up":
                    continue
                suggested = (item.get("suggested_follow_up") or "").strip()
                if suggested:
                    follow_up_questions.append(suggested)

        if utterances:
            speakers = sorted({utterance["speaker"] for utterance in utterances})
            project_summary = (
                f"Draft requirements extracted from {len(utterances)} utterances involving "
                f"{', '.join(speakers)}."
            )
        else:
            project_summary = "No utterances were provided."

        payload = {
            "project_summary": project_summary,
            "functional_requirements": functional_items,
            "non_functional_requirements": non_functional_items,
            "constraints": constraints,
            "assumptions": assumptions,
            "open_questions": open_questions,
            "follow_up_questions": follow_up_questions,
            "question_decisions": question_decisions,
            "utterances": utterances,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


class HuggingFaceModelRunner(BaseModelRunner):
    def __init__(
        self,
        model_name: str,
        generation_config: Optional[dict] = None,
        task: Optional[str] = None,
        context_window_hint: Optional[int] = None,
    ) -> None:
        super().__init__(runner_name=model_name)
        self.model_name = model_name
        self.generation_config = generation_config or {}
        self.task = task
        self._context_window_hint = int(context_window_hint) if context_window_hint else 4096

        self._model = None
        self._tokenizer = None
        self._device = None
        self._device_label = "cpu"
        self._context_window = self._context_window_hint

    def _resolve_device(self):
        import torch

        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            return torch.device("mps"), "mps"

        return torch.device("cpu"), "cpu"

    @staticmethod
    def _infer_context_window(config, tokenizer) -> int:
        candidates: list[int] = []

        tokenizer_max = getattr(tokenizer, "model_max_length", None)
        if isinstance(tokenizer_max, int) and 0 < tokenizer_max <= 32768:
            candidates.append(tokenizer_max)

        for field_name in ("max_position_embeddings", "n_positions", "max_sequence_length"):
            value = getattr(config, field_name, None)
            if isinstance(value, int) and 0 < value <= 32768:
                candidates.append(value)

        if not candidates:
            return 2048

        # Pick the strictest known bound to avoid index errors.
        return max(128, min(candidates))

    @staticmethod
    def _is_oom_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "out of memory" in text or "cuda out of memory" in text or "mps backend out of memory" in text

    def _clear_device_cache(self) -> None:
        import torch

        if self._device_label == "cuda" and hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()
        if self._device_label == "mps":
            mps_mod = getattr(torch, "mps", None)
            if mps_mod is not None and hasattr(mps_mod, "empty_cache"):
                mps_mod.empty_cache()

    def _move_model_to_cpu(self) -> None:
        import torch

        if self._model is None:
            return
        self._model = self._model.to(torch.device("cpu"))
        self._device = torch.device("cpu")
        self._device_label = "cpu"

    def _load_pipeline(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )

        config = AutoConfig.from_pretrained(self.model_name)
        task = self.task or (
            "text2text-generation" if getattr(config, "is_encoder_decoder", False) else "text-generation"
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        device, device_label = self._resolve_device()
        dtype = torch.float16 if device_label in {"cuda", "mps"} else torch.float32

        model_kwargs = {"dtype": dtype}
        try:
            if task == "text2text-generation":
                model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **model_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        except Exception:  # noqa: BLE001
            if task == "text2text-generation":
                model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            else:
                model = AutoModelForCausalLM.from_pretrained(self.model_name)

        model = model.to(device)
        model.eval()

        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._device_label = device_label
        self._context_window = self._infer_context_window(config=config, tokenizer=tokenizer)
        self.task = task

    def get_input_token_budget(self, prompt_config: dict) -> int:
        model_context = self._context_window if self._context_window else self._context_window_hint
        if model_context <= 0:
            model_context = self._context_window_hint

        user_max_input_tokens = prompt_config.get("max_input_tokens")
        if isinstance(user_max_input_tokens, int) and user_max_input_tokens > 0:
            model_context = min(model_context, user_max_input_tokens)

        max_new_tokens = int(prompt_config.get("generation", {}).get("max_new_tokens", self.generation_config.get("max_new_tokens", 256)))

        if self.task == "text-generation":
            return max(128, model_context - max_new_tokens - 16)

        return max(128, model_context)

    def _post_process_output(self, prompt: str, generated_text: str) -> str:
        text = generated_text.strip()
        if self.task == "text-generation" and text.startswith(prompt):
            text = text[len(prompt) :].strip()
        return text

    def generate_structured_output(self, utterances: list[dict], prompt_config: dict) -> str:
        result = self.generate_with_metadata(utterances=utterances, prompt_config=prompt_config)
        return result.raw_text

    def generate_with_metadata(self, utterances: list[dict], prompt_config: dict) -> GenerationResult:
        import torch

        is_first_load = self._model is None or self._tokenizer is None
        if is_first_load:
            self._progress_log(prompt_config, "Loading tokenizer/model weights...")
        self._load_pipeline()
        if is_first_load:
            self._progress_log(prompt_config, f"Model loaded on device={self._device_label}.")

        prompt = build_prompt(utterances=utterances, prompt_config=prompt_config)

        generation_kwargs = {
            "max_new_tokens": 512,
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
        }
        generation_kwargs.update(self.generation_config)
        generation_kwargs.update(prompt_config.get("generation", {}))

        if self._tokenizer is not None:
            if self._tokenizer.pad_token_id is not None:
                generation_kwargs.setdefault("pad_token_id", self._tokenizer.pad_token_id)
            if self._tokenizer.eos_token_id is not None:
                generation_kwargs.setdefault("eos_token_id", self._tokenizer.eos_token_id)

        do_sample = bool(generation_kwargs.get("do_sample", False))
        if not do_sample:
            generation_kwargs.pop("temperature", None)
            generation_kwargs.pop("top_p", None)
            generation_kwargs.pop("top_k", None)

        if self._tokenizer is None or self._model is None or self._device is None:
            raise RuntimeError("Model runner not initialized properly")

        max_input_tokens = self.get_input_token_budget(prompt_config=prompt_config)

        tokenized = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        model_inputs = {key: value.to(self._device) for key, value in tokenized.items()}
        prompt_token_count = int(tokenized["input_ids"].shape[-1])
        max_new_tokens = int(generation_kwargs.get("max_new_tokens", 0))
        self._progress_log(
            prompt_config,
            (
                f"Starting generation "
                f"(prompt_tokens={prompt_token_count}, max_input_tokens={max_input_tokens}, "
                f"max_new_tokens={max_new_tokens}, device={self._device_label})"
            ),
        )

        if self._device_label == "mps":
            generation_kwargs.setdefault("use_cache", False)

        heartbeat_sec_raw = prompt_config.get("progress_heartbeat_sec", 10)
        try:
            heartbeat_sec = max(2, int(heartbeat_sec_raw))
        except (TypeError, ValueError):
            heartbeat_sec = 10

        stop_heartbeat = threading.Event()
        inline_progress = self._inline_progress_enabled(prompt_config) and sys.stdout.isatty()
        inline_heartbeat_printed = False
        inline_last_len = 0

        def inline_heartbeat_log(elapsed: int) -> None:
            nonlocal inline_heartbeat_printed
            nonlocal inline_last_len
            timestamp = datetime.now().strftime("%H:%M:%S")
            line = f"[{timestamp}] [{self.runner_name}] Generation in progress... {elapsed}s elapsed"
            padded = line
            if len(line) < inline_last_len:
                padded += " " * (inline_last_len - len(line))
            sys.stdout.write("\r" + padded)
            sys.stdout.flush()
            inline_last_len = len(line)
            inline_heartbeat_printed = True

        def finalize_inline_heartbeat_line() -> None:
            nonlocal inline_heartbeat_printed
            if inline_progress and inline_heartbeat_printed:
                sys.stdout.write("\n")
                sys.stdout.flush()
                inline_heartbeat_printed = False

        def heartbeat() -> None:
            elapsed = 0
            while not stop_heartbeat.wait(heartbeat_sec):
                elapsed += heartbeat_sec
                if inline_progress:
                    inline_heartbeat_log(elapsed)
                else:
                    self._progress_log(prompt_config, f"Generation in progress... {elapsed}s elapsed")

        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        if self._progress_enabled(prompt_config):
            heartbeat_thread.start()

        start_time = time.perf_counter()
        try:
            try:
                with torch.no_grad():
                    generated = self._model.generate(**model_inputs, **generation_kwargs)
            finally:
                stop_heartbeat.set()
                if heartbeat_thread.is_alive():
                    heartbeat_thread.join(timeout=0.2)
                finalize_inline_heartbeat_line()
        except RuntimeError as exc:
            if not self._is_oom_error(exc) or self._device_label == "cpu":
                raise

            self._progress_log(prompt_config, "OOM detected on accelerator; retrying generation on CPU.")
            self._clear_device_cache()
            self._move_model_to_cpu()
            cpu_inputs = {key: value.to(self._device) for key, value in tokenized.items()}
            retry_kwargs = dict(generation_kwargs)
            retry_kwargs["max_new_tokens"] = min(int(retry_kwargs.get("max_new_tokens", 256)), 256)
            with torch.no_grad():
                generated = self._model.generate(**cpu_inputs, **retry_kwargs)
        latency_sec = time.perf_counter() - start_time
        self._progress_log(prompt_config, f"Generation completed in {latency_sec:.2f}s.")

        if generated.ndim != 2 or generated.shape[0] == 0:
            generated_text = ""
            completion_token_count = 0
        elif self.task == "text-generation":
            prompt_token_count = model_inputs["input_ids"].shape[-1]
            completion_ids = generated[0][prompt_token_count:]
            generated_text = self._tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            completion_token_count = int(completion_ids.shape[-1]) if completion_ids.ndim == 1 else 0
        else:
            completion_ids = generated[0]
            generated_text = self._tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            completion_token_count = int(completion_ids.shape[-1]) if completion_ids.ndim == 1 else 0

        cleaned_output = self._post_process_output(prompt=prompt, generated_text=generated_text)

        prompt_tokens = int(tokenized["input_ids"].shape[-1])
        completion_tokens = completion_token_count

        return GenerationResult(
            model_name=self.runner_name,
            raw_text=cleaned_output,
            latency_sec=latency_sec,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


def available_model_keys(models_config: dict) -> list[str]:
    return list(models_config.get("models", {}).keys())


def resolve_generation_config(models_config: dict, model_key: str) -> dict:
    defaults = dict(models_config.get("generation_defaults", {}))
    model_entry = models_config.get("models", {}).get(model_key, {})
    defaults.update(model_entry.get("generation", {}))
    return defaults


def create_model_runner(
    model_key: Optional[str],
    models_config: dict,
    use_mock: bool = False,
) -> BaseModelRunner:
    if use_mock:
        return MockModelRunner()

    models = models_config.get("models", {})
    if model_key is None:
        model_key = models_config.get("default_model")

    if model_key is None:
        raise ValueError("No model key provided and no default model configured")

    model_entry = models.get(model_key)
    default_context_window_hint = int(models_config.get("default_context_window_hint", 4096))

    if model_entry is None:
        generation = dict(models_config.get("generation_defaults", {}))
        return HuggingFaceModelRunner(
            model_name=model_key,
            generation_config=generation,
            context_window_hint=default_context_window_hint,
        )

    model_type = model_entry.get("type", "huggingface").lower()
    if model_type == "mock":
        return MockModelRunner()

    if model_type != "huggingface":
        raise ValueError(f"Unsupported model type: {model_type}")

    model_name = model_entry.get("model_name", model_key)
    generation_config = resolve_generation_config(models_config=models_config, model_key=model_key)
    task = model_entry.get("task")
    context_window_hint = int(model_entry.get("context_window_hint", default_context_window_hint))
    return HuggingFaceModelRunner(
        model_name=model_name,
        generation_config=generation_config,
        task=task,
        context_window_hint=context_window_hint,
    )


def create_classification_runner(prompt_config: dict, use_mock: bool = False) -> BaseClassificationRunner:
    if use_mock or bool(prompt_config.get("classification_use_mock", False)):
        return MockClassificationRunner()

    model_name = str(
        prompt_config.get(
            "classification_model_name",
            "MoritzLaurer/ModernBERT-large-zeroshot-v2.0",
        )
    ).strip()
    if not model_name:
        return MockClassificationRunner()

    return HuggingFaceZeroShotClassificationRunner(model_name=model_name)
