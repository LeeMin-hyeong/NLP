from __future__ import annotations

import json
import random
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseModelRunner(ABC):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.last_generation_info: dict[str, Any] = {}

    @abstractmethod
    def generate(self, prompt: str, generation_config: dict) -> str:
        raise NotImplementedError

    def close(self) -> None:
        return None


class HFModelRunner(BaseModelRunner):
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name=model_name)
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._device = None

    def _load_tokenizer(self, auto_tokenizer: Any) -> Any:
        try:
            return auto_tokenizer.from_pretrained(self.model_name)
        except AttributeError as exc:
            # Gemma 4 can surface extra_special_tokens as a list during tokenizer
            # initialization. Retrying with an explicit empty mapping avoids that
            # incompatible path without changing other models.
            if "keys" not in str(exc):
                raise
            return auto_tokenizer.from_pretrained(self.model_name, extra_special_tokens={})

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional runtime env
            raise RuntimeError(
                "Transformers and torch are required for HFModelRunner."
            ) from exc

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = self._load_tokenizer(AutoTokenizer)
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        model_kwargs: dict[str, Any] = {"dtype": dtype}
        if self._device == "cuda":
            model_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if self._device == "cpu":
            self._model.to("cpu")

        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def generate(self, prompt: str, generation_config: dict) -> str:
        self._ensure_loaded()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None

        seed = generation_config.get("seed")
        if seed is not None:
            try:
                seed_int = int(seed)
                random.seed(seed_int)
                self._torch.manual_seed(seed_int)
                if self._torch.cuda.is_available():
                    self._torch.cuda.manual_seed_all(seed_int)
            except Exception:
                # Seed is best-effort and should not fail generation.
                pass

        started = time.perf_counter()
        prompt_text = prompt
        chat_template = getattr(self._tokenizer, "chat_template", None)
        if chat_template:
            try:
                prompt_text = self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt_text = prompt

        encoded = self._tokenizer(prompt_text, return_tensors="pt")
        encoded = {
            key: value.to(self._model.device)
            for key, value in encoded.items()
        }

        temperature = float(generation_config.get("temperature", 0.0))
        do_sample = bool(generation_config.get("do_sample", False))
        if temperature <= 0:
            do_sample = False

        with self._torch.no_grad():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=int(generation_config.get("max_new_tokens", 900)),
                temperature=temperature,
                top_p=float(generation_config.get("top_p", 1.0)),
                do_sample=do_sample,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        prompt_tokens = int(encoded["input_ids"].shape[-1])
        completion_token_ids = generated[0][prompt_tokens:]
        completion_tokens = int(completion_token_ids.shape[-1])
        output = self._tokenizer.decode(completion_token_ids, skip_special_tokens=True).strip()
        stop_sequences = generation_config.get("stop_sequences", [])
        if isinstance(stop_sequences, list):
            earliest_cut = None
            for stop in stop_sequences:
                if not stop:
                    continue
                idx = output.find(str(stop))
                if idx >= 0 and (earliest_cut is None or idx < earliest_cut):
                    earliest_cut = idx
            if earliest_cut is not None:
                output = output[:earliest_cut].strip()
        elapsed = time.perf_counter() - started

        self.last_generation_info = {
            "model_name": self.model_name,
            "latency_sec": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        return output


class MockModelRunner(BaseModelRunner):
    FUNCTIONAL_HINTS = (
        "need",
        "should be able",
        "allow",
        "let",
        "reserve",
        "book",
        "admin",
        "dashboard",
        "track",
        "update",
        "create",
        "manage",
    )
    NFR_HINTS = (
        "fast",
        "performance",
        "responsive",
        "mobile",
        "modern",
        "clean",
        "simple",
        "easy",
        "secure",
        "reliable",
        "accessible",
    )
    FUTURE_HINTS = ("later", "future", "phase 2", "eventually", "maybe")
    AMBIGUITY_HINTS = ("clean", "modern", "simple", "easy", "probably", "maybe")
    CONSTRAINT_HINTS = (
        "not in version one",
        "not in v1",
        "not part of the first release",
        "first release",
        "web only",
        "no app",
        "within",
        "deadline",
        "budget",
        "limited budget",
        "only staff",
        "only admin",
        "only administrators",
        "must launch",
    )

    def __init__(self) -> None:
        super().__init__(model_name="mock")

    @staticmethod
    def _extract_units(prompt: str) -> list[tuple[str, str]]:
        pattern = re.compile(r"^(U\d+)\s+\|\s+(.+)$", flags=re.MULTILINE)
        return [(m.group(1), m.group(2).strip()) for m in pattern.finditer(prompt)]

    @staticmethod
    def _extract_first_json_object(text: str) -> str | None:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    @classmethod
    def _extract_json_after_label(cls, prompt: str, label: str) -> dict[str, Any]:
        idx = prompt.find(label)
        if idx < 0:
            return {}
        tail = prompt[idx + len(label) :]
        block = cls._extract_first_json_object(tail)
        if not block:
            return {}
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
        return {}

    @staticmethod
    def _rewrite_requirement(text: str, prefix: str) -> str:
        normalized = text.strip().rstrip(".")
        lowered = normalized.lower()

        if lowered.startswith("we need "):
            rest = normalized[8:].strip()
            return f"{prefix} provide {rest}."
        if lowered.startswith("i need "):
            rest = normalized[7:].strip()
            return f"{prefix} provide {rest}."
        if lowered.startswith("we want "):
            rest = normalized[8:].strip()
            return f"{prefix} provide {rest}."
        if lowered.startswith("i want "):
            rest = normalized[7:].strip()
            return f"{prefix} provide {rest}."
        if lowered.startswith("customers should be able to "):
            rest = normalized[28:].strip()
            return f"{prefix} allow customers to {rest}."
        if lowered.startswith("users should be able to "):
            rest = normalized[24:].strip()
            return f"{prefix} allow users to {rest}."
        if lowered.startswith("staff must be able to "):
            rest = normalized[22:].strip()
            return f"{prefix} allow staff to {rest}."
        if lowered.startswith("the site must "):
            rest = normalized[14:].strip()
            return f"{prefix} {rest}."
        if lowered.startswith("it should "):
            rest = normalized[10:].strip()
            return f"{prefix} {rest}."
        if lowered.startswith("the app should "):
            rest = normalized[15:].strip()
            return f"{prefix} {rest}."
        if lowered.startswith("most visitors will use phones"):
            return (
                f"{prefix} provide a mobile-first experience with fast loading for phone users."
            )
        if lowered.startswith("the design should feel "):
            rest = normalized[23:].strip()
            return f"{prefix} follow a {rest} visual style."

        if not normalized:
            normalized = "support the described behavior"
        return f"{prefix} {normalized[0].lower() + normalized[1:]}."

    @staticmethod
    def _rewrite_constraint(text: str) -> str:
        normalized = text.strip().rstrip(".")
        lowered = normalized.lower()
        if "may add" in lowered or "might add" in lowered:
            feature = normalized
            feature = re.sub(r"^(we|i)\s+(may|might)\s+add\s+", "", feature, flags=re.IGNORECASE)
            feature = feature.strip()
            if feature:
                return f"The feature '{feature}' shall not be included in the initial release."
        if "web only" in lowered or ("website" in lowered and "not" in lowered and "app" in lowered):
            return "The initial solution shall be delivered as a web application only."
        if "budget" in lowered:
            return "The initial scope should remain limited to fit the available project budget."
        if "within" in lowered or "must launch" in lowered or "deadline" in lowered:
            return "The implementation scope shall be planned to meet the stated delivery timeline."
        if "only staff" in lowered or "only admin" in lowered or "only administrators" in lowered:
            return "Administrative functions shall be restricted to authorized staff or administrators."
        return f"The project shall satisfy this boundary: {normalized}."

    @staticmethod
    def _build_summary(units: list[tuple[str, str]]) -> str:
        if not units:
            return (
                "The conversation describes a software project but lacks clear requirements. "
                "Clarification is required before implementation."
            )
        first = units[0][1].rstrip(".")
        summary = (
            f"The conversation outlines a software project focused on {first.lower()}. "
            "It includes feature requests, quality expectations, and unresolved points "
            "that require clarification before implementation."
        )
        return summary

    def _stage_1_candidates(self, units: list[tuple[str, str]]) -> dict[str, Any]:
        candidates: list[dict[str, Any]] = []
        cid = 1
        for uid, text in units:
            lowered = text.lower()
            if any(hint in lowered for hint in self.FUNCTIONAL_HINTS):
                candidates.append(
                    {
                        "id": f"C{cid}",
                        "kind": "possible_requirement",
                        "text": text.rstrip("."),
                        "source_units": [uid],
                    }
                )
                cid += 1
            if any(hint in lowered for hint in self.NFR_HINTS):
                candidates.append(
                    {
                        "id": f"C{cid}",
                        "kind": "possible_quality_expectation",
                        "text": text.rstrip("."),
                        "source_units": [uid],
                    }
                )
                cid += 1
            if any(hint in lowered for hint in self.FUTURE_HINTS):
                candidates.append(
                    {
                        "id": f"C{cid}",
                        "kind": "possible_future_scope",
                        "text": text.rstrip("."),
                        "source_units": [uid],
                    }
                )
                cid += 1
            if any(hint in lowered for hint in self.CONSTRAINT_HINTS):
                candidates.append(
                    {
                        "id": f"C{cid}",
                        "kind": "possible_constraint",
                        "text": text.rstrip("."),
                        "source_units": [uid],
                    }
                )
                cid += 1
            if "?" in text or any(hint in lowered for hint in self.AMBIGUITY_HINTS):
                candidates.append(
                    {
                        "id": f"C{cid}",
                        "kind": "possible_ambiguity",
                        "text": text.rstrip("."),
                        "source_units": [uid],
                    }
                )
                cid += 1
                candidates.append(
                    {
                        "id": f"C{cid}",
                        "kind": "possible_followup_trigger",
                        "text": text.rstrip("."),
                        "source_units": [uid],
                    }
                )
                cid += 1
        return {"candidates": candidates}

    def _stage_2_classify(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        out: list[dict[str, Any]] = []
        for item in candidates:
            cid = str(item.get("id", "")).strip() or "C0"
            text = str(item.get("text", "")).strip()
            kind = str(item.get("kind", "")).strip()
            source_units = [str(x).strip() for x in item.get("source_units", []) if str(x).strip()]
            lowered = text.lower()

            final_type = "discard"
            reason = "Not useful for final specification."
            has_future = any(h in lowered for h in self.FUTURE_HINTS)
            has_constraint = any(h in lowered for h in self.CONSTRAINT_HINTS)

            if kind == "possible_constraint":
                if has_future and not (
                    "not in version one" in lowered
                    or "not in v1" in lowered
                    or "not part of the first release" in lowered
                    or "web only" in lowered
                    or "no app" in lowered
                ):
                    final_type = "note"
                    reason = "Future possibility is noted, but no explicit current boundary is fixed."
                elif has_constraint:
                    final_type = "constraint"
                    reason = "Explicit project boundary or implementation limitation is stated."
                elif any(h in lowered for h in self.AMBIGUITY_HINTS):
                    final_type = "open_question"
                    reason = "Constraint intent exists but details are under-specified."
                else:
                    final_type = "constraint"
                    reason = "Represents a project boundary."
            elif kind == "possible_future_scope" or has_future:
                final_type = "note"
                reason = "Future/optional scope should be tracked as note."
            elif kind == "possible_followup_trigger":
                final_type = "follow_up_trigger"
                reason = "Requires concrete clarification before implementation."
            elif kind == "possible_ambiguity" or "?" in text:
                final_type = "open_question"
                reason = "Ambiguity should stay unresolved until clarified."
            elif kind == "possible_quality_expectation":
                if any(h in lowered for h in self.AMBIGUITY_HINTS):
                    final_type = "open_question"
                    reason = "Quality expectation is vague and needs clarification."
                else:
                    final_type = "non_functional_requirement"
                    reason = "Describes quality expectations."
            elif kind == "possible_requirement":
                if any(h in lowered for h in self.AMBIGUITY_HINTS):
                    final_type = "open_question"
                    reason = "Potential requirement is too vague."
                elif has_constraint:
                    final_type = "constraint"
                    reason = "Statement encodes a clear boundary rather than pure behavior."
                elif any(h in lowered for h in self.NFR_HINTS) and not any(
                    h in lowered for h in ("allow", "reserve", "book", "update", "create", "manage")
                ):
                    final_type = "non_functional_requirement"
                    reason = "Behavior is mostly quality-oriented."
                else:
                    final_type = "functional_requirement"
                    reason = "Represents system behavior/capability."

            out.append(
                {
                    "id": cid,
                    "final_type": final_type,
                    "reason": reason,
                    "source_units": source_units,
                }
            )
        return {"classified_candidates": out}

    def _stage_3_rewrite(self, classified_candidates: list[dict[str, Any]]) -> dict[str, Any]:
        rewritten: list[dict[str, Any]] = []
        rid = 1
        for item in classified_candidates:
            ctype = str(item.get("final_type", "")).strip()
            if ctype not in {"functional_requirement", "non_functional_requirement", "constraint"}:
                continue
            text = str(item.get("reason_text", "")).strip() or str(item.get("text", "")).strip()
            if not text:
                text = str(item.get("reason", "")).strip()
            if not text:
                continue
            if ctype == "functional_requirement":
                rewritten_text = self._rewrite_requirement(text, "The system shall")
            elif ctype == "non_functional_requirement":
                rewritten_text = self._rewrite_requirement(text, "The system should")
            else:
                rewritten_text = self._rewrite_constraint(text)
            rewritten.append(
                {
                    "id": f"R{rid}",
                    "type": ctype,
                    "text": rewritten_text,
                    "source_units": [str(x).strip() for x in item.get("source_units", []) if str(x).strip()],
                }
            )
            rid += 1
        return {"rewritten_items": rewritten}

    def _stage_4_open_questions(
        self,
        classified_candidates: list[dict[str, Any]],
        rewritten_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        questions: list[dict[str, Any]] = []
        for item in classified_candidates:
            ftype = str(item.get("final_type", "")).strip()
            if ftype not in {"open_question", "follow_up_trigger", "constraint"}:
                continue
            source_units = [str(x).strip() for x in item.get("source_units", []) if str(x).strip()]
            reason = str(item.get("reason", "")).strip().rstrip(".")
            text = str(item.get("text", "")).strip().lower()
            if ftype == "constraint" and ("budget" in text or "budget" in reason.lower()):
                q = "What budget boundary should be treated as a hard constraint for the initial release?"
            elif ftype == "constraint" and ("within" in text or "deadline" in text or "launch" in text):
                q = "What exact launch date or deadline is required for the first release?"
            elif ftype == "constraint" and ("only staff" in text or "only admin" in text):
                q = "Which actions must be restricted to staff or administrators?"
            elif ftype == "constraint":
                q = "Can you clarify this boundary as a hard initial-release constraint?"
            elif "design" in reason.lower() or "style" in reason.lower():
                q = "What concrete design references define the expected style?"
            else:
                q = "Could you clarify the unresolved intent for this item?"
            if source_units:
                questions.append({"text": q, "source_units": source_units})

        if not questions and rewritten_items:
            questions.append(
                {
                    "text": "Which requirement details are still unresolved for the first release?",
                    "source_units": [rewritten_items[0].get("source_units", ["U1"])[0]],
                }
            )
        return {"open_questions": questions}

    def _stage_5_followups(
        self,
        classified_candidates: list[dict[str, Any]],
        rewritten_items: list[dict[str, Any]],
        open_questions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        questions: list[dict[str, Any]] = []
        for item in classified_candidates:
            ftype = str(item.get("final_type", "")).strip()
            if ftype not in {"follow_up_trigger", "constraint", "note"}:
                continue
            source_units = [str(x).strip() for x in item.get("source_units", []) if str(x).strip()]
            text = str(item.get("text", "")).strip().lower()
            reason = str(item.get("reason", "")).strip().lower()

            if ftype == "note":
                q = "Should this future-scope item be included in the first release plan?"
            elif "budget" in text or "budget" in reason:
                q = "What budget range should we plan against for the first release?"
            elif "within" in text or "deadline" in text or "launch" in text:
                q = "What delivery date should be treated as the committed target?"
            elif "only staff" in text or "only admin" in text:
                q = "Which exact admin actions require restricted access controls?"
            else:
                q = "What decision is needed next to proceed with implementation?"

            if source_units:
                questions.append({"text": q, "source_units": source_units})

        for item in open_questions:
            text = str(item.get("text", "")).strip().rstrip("?")
            source_units = [str(x).strip() for x in item.get("source_units", []) if str(x).strip()]
            if not text or not source_units:
                continue
            questions.append(
                {
                    "text": f"What decision should we make to resolve: {text}?",
                    "source_units": source_units,
                }
            )

        if not questions and rewritten_items:
            questions.append(
                {
                    "text": "What is the highest-priority requirement for the first release?",
                    "source_units": [rewritten_items[0].get("source_units", ["U1"])[0]],
                }
            )
        return {"follow_up_questions": questions}

    def _stage_6_summary(
        self,
        units: list[tuple[str, str]],
        rewritten_items: list[dict[str, Any]],
        open_questions: list[dict[str, Any]],
        notes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        feature_count = len(
            [x for x in rewritten_items if str(x.get("type", "")).strip() in {"functional_requirement", "non_functional_requirement"}]
        )
        constraint_count = len(
            [x for x in rewritten_items if str(x.get("type", "")).strip() == "constraint"]
        )
        open_count = len(open_questions)
        note_count = len(notes)
        if units:
            anchor = units[0][1].rstrip(".")
        else:
            anchor = "the discussed software project"
        summary = (
            f"The conversation describes a project focused on {anchor.lower()}. "
            f"It currently includes {feature_count} drafted requirements and {constraint_count} explicit constraints. "
            f"There are {open_count} unresolved points and {note_count} scope notes that should be confirmed before implementation."
        )
        return {"project_summary": summary}

    # Backward-compatible alias for old stage numbering.
    def _stage_5_summary(
        self,
        units: list[tuple[str, str]],
        rewritten_items: list[dict[str, Any]],
        open_questions: list[dict[str, Any]],
        notes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._stage_6_summary(units, rewritten_items, open_questions, notes)

    def generate(self, prompt: str, generation_config: dict) -> str:
        started = time.perf_counter()
        units = self._extract_units(prompt)

        if "CHAIN_STAGE:1_CANDIDATE_EXTRACTION" in prompt:
            payload = self._stage_1_candidates(units)
            output = json.dumps(payload, indent=2)
            elapsed = time.perf_counter() - started
            self.last_generation_info = {
                "model_name": self.model_name,
                "latency_sec": elapsed,
                "prompt_tokens": None,
                "completion_tokens": None,
                "generation_config": generation_config,
            }
            return output

        if "CHAIN_STAGE:2_CANDIDATE_CLASSIFICATION" in prompt:
            parsed = self._extract_json_after_label(prompt, "Candidates JSON:")
            candidates = parsed.get("candidates", []) if isinstance(parsed, dict) else []
            payload = self._stage_2_classify(candidates)
            output = json.dumps(payload, indent=2)
            elapsed = time.perf_counter() - started
            self.last_generation_info = {
                "model_name": self.model_name,
                "latency_sec": elapsed,
                "prompt_tokens": None,
                "completion_tokens": None,
                "generation_config": generation_config,
            }
            return output

        if "CHAIN_STAGE:3_REQUIREMENT_REWRITING" in prompt:
            parsed = self._extract_json_after_label(prompt, "Classified candidates JSON:")
            classified = parsed.get("classified_candidates", []) if isinstance(parsed, dict) else []
            payload = self._stage_3_rewrite(classified)
            output = json.dumps(payload, indent=2)
            elapsed = time.perf_counter() - started
            self.last_generation_info = {
                "model_name": self.model_name,
                "latency_sec": elapsed,
                "prompt_tokens": None,
                "completion_tokens": None,
                "generation_config": generation_config,
            }
            return output

        if "CHAIN_STAGE:4_OPEN_QUESTION_GENERATION" in prompt:
            parsed_cls = self._extract_json_after_label(prompt, "Classified candidates JSON:")
            parsed_rw = self._extract_json_after_label(prompt, "Rewritten items JSON:")
            classified = (
                parsed_cls.get("classified_candidates", []) if isinstance(parsed_cls, dict) else []
            )
            rewritten = parsed_rw.get("rewritten_items", []) if isinstance(parsed_rw, dict) else []
            payload = self._stage_4_open_questions(classified, rewritten)
            output = json.dumps(payload, indent=2)
            elapsed = time.perf_counter() - started
            self.last_generation_info = {
                "model_name": self.model_name,
                "latency_sec": elapsed,
                "prompt_tokens": None,
                "completion_tokens": None,
                "generation_config": generation_config,
            }
            return output

        if "CHAIN_STAGE:5_FOLLOWUP_GENERATION" in prompt:
            parsed_cls = self._extract_json_after_label(prompt, "Classified candidates JSON:")
            parsed_rw = self._extract_json_after_label(prompt, "Rewritten items JSON:")
            parsed_oq = self._extract_json_after_label(prompt, "Open questions JSON:")
            classified = (
                parsed_cls.get("classified_candidates", []) if isinstance(parsed_cls, dict) else []
            )
            rewritten = parsed_rw.get("rewritten_items", []) if isinstance(parsed_rw, dict) else []
            open_questions = parsed_oq.get("open_questions", []) if isinstance(parsed_oq, dict) else []
            payload = self._stage_5_followups(classified, rewritten, open_questions)
            output = json.dumps(payload, indent=2)
            elapsed = time.perf_counter() - started
            self.last_generation_info = {
                "model_name": self.model_name,
                "latency_sec": elapsed,
                "prompt_tokens": None,
                "completion_tokens": None,
                "generation_config": generation_config,
            }
            return output

        if "CHAIN_STAGE:6_PROJECT_SUMMARY" in prompt:
            parsed_rw = self._extract_json_after_label(prompt, "Rewritten items JSON:")
            parsed_oq = self._extract_json_after_label(prompt, "Open questions JSON:")
            parsed_notes = self._extract_json_after_label(prompt, "Notes JSON:")
            rewritten = parsed_rw.get("rewritten_items", []) if isinstance(parsed_rw, dict) else []
            open_questions = parsed_oq.get("open_questions", []) if isinstance(parsed_oq, dict) else []
            notes = parsed_notes.get("notes", []) if isinstance(parsed_notes, dict) else []
            payload = self._stage_6_summary(units, rewritten, open_questions, notes)
            output = json.dumps(payload, indent=2)
            elapsed = time.perf_counter() - started
            self.last_generation_info = {
                "model_name": self.model_name,
                "latency_sec": elapsed,
                "prompt_tokens": None,
                "completion_tokens": None,
                "generation_config": generation_config,
            }
            return output

        # Backward-compatible stage labels.
        if "CHAIN_STAGE:4_FOLLOWUP_GENERATION" in prompt:
            parsed_cls = self._extract_json_after_label(prompt, "Classified candidates JSON:")
            parsed_rw = self._extract_json_after_label(prompt, "Rewritten items JSON:")
            classified = (
                parsed_cls.get("classified_candidates", []) if isinstance(parsed_cls, dict) else []
            )
            rewritten = parsed_rw.get("rewritten_items", []) if isinstance(parsed_rw, dict) else []
            payload = self._stage_5_followups(classified, rewritten, [])
            output = json.dumps(payload, indent=2)
            elapsed = time.perf_counter() - started
            self.last_generation_info = {
                "model_name": self.model_name,
                "latency_sec": elapsed,
                "prompt_tokens": None,
                "completion_tokens": None,
                "generation_config": generation_config,
            }
            return output

        if "CHAIN_STAGE:5_PROJECT_SUMMARY" in prompt:
            parsed_rw = self._extract_json_after_label(prompt, "Rewritten items JSON:")
            parsed_oq = self._extract_json_after_label(prompt, "Open questions JSON:")
            parsed_notes = self._extract_json_after_label(prompt, "Notes JSON:")
            rewritten = parsed_rw.get("rewritten_items", []) if isinstance(parsed_rw, dict) else []
            open_questions = parsed_oq.get("open_questions", []) if isinstance(parsed_oq, dict) else []
            notes = parsed_notes.get("notes", []) if isinstance(parsed_notes, dict) else []
            payload = self._stage_6_summary(units, rewritten, open_questions, notes)
            output = json.dumps(payload, indent=2)
            elapsed = time.perf_counter() - started
            self.last_generation_info = {
                "model_name": self.model_name,
                "latency_sec": elapsed,
                "prompt_tokens": None,
                "completion_tokens": None,
                "generation_config": generation_config,
            }
            return output

        fr_items: list[dict[str, Any]] = []
        nfr_items: list[dict[str, Any]] = []
        constraint_items: list[dict[str, Any]] = []
        open_questions: list[dict[str, Any]] = []
        follow_ups: list[dict[str, Any]] = []
        notes: list[dict[str, Any]] = []

        for uid, text in units:
            lowered = text.lower()
            is_question = "?" in text
            has_functional_hint = any(hint in lowered for hint in self.FUNCTIONAL_HINTS)
            has_nfr_hint = any(hint in lowered for hint in self.NFR_HINTS)
            has_future_hint = any(hint in lowered for hint in self.FUTURE_HINTS)
            has_ambiguity_hint = any(hint in lowered for hint in self.AMBIGUITY_HINTS)
            has_constraint_hint = any(hint in lowered for hint in self.CONSTRAINT_HINTS)

            if is_question:
                open_questions.append({"text": text.rstrip(), "source_units": [uid]})
                follow_ups.append(
                    {
                        "text": f"Please confirm the decision for: {text.rstrip('?')}.",
                        "source_units": [uid],
                    }
                )
                continue

            if has_functional_hint:
                fr_items.append(
                    {
                        "id": f"FR{len(fr_items) + 1}",
                        "text": self._rewrite_requirement(text, "The system shall"),
                        "source_units": [uid],
                    }
                )

            if has_nfr_hint:
                nfr_items.append(
                    {
                        "id": f"NFR{len(nfr_items) + 1}",
                        "text": self._rewrite_requirement(text, "The system should"),
                        "source_units": [uid],
                    }
                )

            if has_future_hint:
                notes.append(
                    {
                        "text": f"Future-scope item: {text.rstrip('.')}.",
                        "source_units": [uid],
                    }
                )
                follow_ups.append(
                    {
                        "text": "Should this future-scope item be included in the current release?",
                        "source_units": [uid],
                    }
                )

            if has_constraint_hint and not has_future_hint:
                constraint_items.append(
                    {
                        "id": f"CON{len(constraint_items) + 1}",
                        "text": self._rewrite_constraint(text),
                        "source_units": [uid],
                    }
                )
                follow_ups.append(
                    {
                        "text": "Can you confirm this boundary as a hard constraint for the initial release?",
                        "source_units": [uid],
                    }
                )

            if has_ambiguity_hint and not has_future_hint:
                open_questions.append(
                    {
                        "text": f"Clarify expected outcome for: {text.rstrip('.')}.",
                        "source_units": [uid],
                    }
                )
                follow_ups.append(
                    {
                        "text": "Can you provide measurable acceptance criteria for this expectation?",
                        "source_units": [uid],
                    }
                )

        if not fr_items and units:
            uid, text = units[0]
            fr_items.append(
                {
                    "id": "FR1",
                    "text": self._rewrite_requirement(text, "The system shall"),
                    "source_units": [uid],
                }
            )

        if not follow_ups and units:
            follow_ups.append(
                {
                    "text": "What is the highest-priority requirement for the first release?",
                    "source_units": [units[0][0]],
                }
            )

        payload = {
            "project_summary": self._build_summary(units),
            "functional_requirements": fr_items,
            "non_functional_requirements": nfr_items,
            "constraints": constraint_items,
            "open_questions": open_questions,
            "follow_up_questions": follow_ups,
            "notes": notes,
            "conversation_units": [{"id": uid, "text": text} for uid, text in units],
        }

        output = json.dumps(payload, indent=2)
        elapsed = time.perf_counter() - started
        self.last_generation_info = {
            "model_name": self.model_name,
            "latency_sec": elapsed,
            "prompt_tokens": None,
            "completion_tokens": None,
            "generation_config": generation_config,
        }
        return output
