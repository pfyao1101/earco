import json
import os
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple
import re
from pathlib import Path

from utils import normalize_text, write_jsonl

def build_rca_data_processor(dataset_base_cls: Any) -> Any:
    class RCADataProcessor(dataset_base_cls):
        def __init__(self) -> None:
            self._judge_client = None
            self._judge_cache: Dict[Tuple[str, str], bool] = {}
            self._judge_enabled = os.getenv("RCA_USE_LLM_JUDGE", "true").strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
            self._judge_model = os.getenv(
                "RCA_JUDGE_MODEL",
                os.getenv("OPENAI_MODEL_NAME", os.getenv("SLM_MODEL", "")),
            ).strip()
            self._judge_temperature = float(os.getenv("RCA_JUDGE_TEMPERATURE", "0") or 0.0)

        @staticmethod
        def _extract_last_number(text: str) -> Optional[str]:
            numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
            return numbers[-1] if numbers else None

        @staticmethod
        def _is_pure_number(text: str) -> bool:
            return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", normalize_text(text).replace(",", "")))

        @staticmethod
        def _get_sample_text(sample: Dict[str, Any], keys: Tuple[str, ...]) -> str:
            for key in keys:
                value = sample.get(key)
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    return text
            return ""

        @staticmethod
        def _strip_think_blocks(text: str) -> str:
            return normalize_text(re.sub(r"(?is)<think>.*?</think>", " ", text))

        @classmethod
        def _strip_leading_labels(cls, text: str) -> str:
            cleaned = normalize_text(text)
            cleaned = re.sub(
                r"^(?:root cause|troubleshooting steps|diagnosis|conclusion|final answer|answer|reasoning process)\s*[:\-]?\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(r"^(?:fortd)\b\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"^\d+\s*[:.)-]\s*", "", cleaned)
            return normalize_text(cleaned)

        @classmethod
        def _extract_final_section(cls, text: str) -> str:
            cleaned = cls._strip_think_blocks(text)
            if not cleaned:
                return ""

            marker_patterns = (
                r"(?i)\broot cause\b\s*[:\-]?",
                r"(?i)\btroubleshooting steps\b\s*[:\-]?",
                r"(?i)\bdiagnosis\b\s*[:\-]?",
                r"(?i)\bconclusion\b\s*[:\-]?",
                r"(?i)\bfinal answer\b\s*[:\-]?",
            )
            last_match = None
            for marker_pattern in marker_patterns:
                for match in re.finditer(marker_pattern, cleaned):
                    if last_match is None or match.start() > last_match.start():
                        last_match = match

            extracted = cleaned[last_match.end():] if last_match else cleaned
            extracted = cls._strip_leading_labels(extracted)
            return extracted

        @staticmethod
        def _comparison_text(text: str) -> str:
            cleaned = normalize_text(text).lower()
            cleaned = re.sub(r"[^\w\s]+", " ", cleaned, flags=re.UNICODE)
            return normalize_text(cleaned)

        @staticmethod
        def _parse_judge_response(content: str) -> Optional[bool]:
            if not content:
                return None

            payload: Dict[str, Any] = {}
            try:
                payload = json.loads(content)
            except Exception:
                match = re.search(r"\{.*\}", content, flags=re.DOTALL)
                if match:
                    try:
                        payload = json.loads(match.group(0))
                    except Exception:
                        payload = {}
                else:
                    lowered = content.strip().lower()
                    if lowered.startswith("yes") or "true" in lowered:
                        return True
                    if lowered.startswith("no") or "false" in lowered:
                        return False
                    return None

            value = payload.get("is_correct")
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "yes", "correct", "same"}:
                    return True
                if lowered in {"false", "no", "incorrect", "different"}:
                    return False
            return None

        def _get_judge_client(self) -> Any:
            if not self._judge_enabled or not self._judge_model:
                return None

            if self._judge_client is not None:
                return self._judge_client

            api_base = os.getenv("OPENAI_API_BASE")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_base or not api_key:
                return None

            try:
                from openai import OpenAI
            except Exception:
                return None

            self._judge_client = OpenAI(api_key=api_key, base_url=api_base.rstrip("/"))
            return self._judge_client

        def _judge_answer(self, predicted: str, gold: str) -> Optional[bool]:
            cache_key = (predicted, gold)
            if cache_key in self._judge_cache:
                return self._judge_cache[cache_key]

            client = self._get_judge_client()
            if client is None:
                return None

            system_prompt = (
                "You are a strict evaluator for root cause analysis answers. "
                "Decide whether the predicted answer and the reference answer describe the same primary root cause or diagnosis."
            )
            user_prompt = (
                "Judge the two RCA answers below. Ignore differences in wording, order, and formatting. "
                "The reference answer may contain reasoning and troubleshooting steps; focus on the core root cause or diagnosis. "
                "Mark correct only if the predicted answer expresses the same underlying root cause as the reference answer. "
                "Mark incorrect if the root cause differs, is too vague, or contradicts the reference.\n\n"
                "Return JSON only with this schema:\n"
                '{"is_correct": true/false, "reason": "short explanation"}\n\n'
                f"Predicted answer:\n<<<{predicted}>>>\n\n"
                f"Reference answer:\n<<<{gold}>>>"
            )

            try:
                response = client.chat.completions.create(
                    model=self._judge_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self._judge_temperature,
                )
                content = response.choices[0].message.content if response.choices else None
                decision = self._parse_judge_response(content or "")
                if decision is not None:
                    self._judge_cache[cache_key] = decision
                return decision
            except Exception:
                return None

        def dataset_to_jsonl(self, dataset_jsonl: str, task: str = "", **kwargs: Any) -> None:
            _ = task
            dataset = kwargs.get("dataset", [])
            rows: List[Dict[str, Any]] = []
            for sample in dataset:
                question = normalize_text(self._get_sample_text(sample, ("question", "input", "phenomenon")))
                answer = self._get_sample_text(sample, ("answer", "output", "label"))
                if not question or not answer:
                    continue
                rows.append(
                    {
                        self.QUESTION_LITERAL: question,
                        self.ANSWER_WITH_REASON_LITERAL: answer,
                        self.FINAL_ANSWER_LITERAL: self.extract_final_answer(answer),
                    }
                        )
            write_jsonl(Path(dataset_jsonl), rows)

        def extract_final_answer(self, answer: str) -> str:
            if not answer:
                return self.INVALID_ANS

            pattern = re.compile(
                rf"{re.escape(self.ANSWER_START)}(.*?){re.escape(self.ANSWER_END)}",
                re.DOTALL,
            )
            match = pattern.search(answer)
            if match:
                extracted = normalize_text(match.group(1))
                return extracted if extracted else self.INVALID_ANS

            extracted = self._extract_final_section(answer)
            return extracted if extracted else self.INVALID_ANS

        def access_answer(self, llm_output: str, gt_answer: str) -> Tuple[bool, Any]:
            judge_decision = self._judge_answer(llm_output, gt_answer)
            if judge_decision is not None:
                return judge_decision, normalize_text(llm_output)

            predicted = self.extract_final_answer(llm_output)
            gold = self.extract_final_answer(gt_answer)

            if predicted == self.INVALID_ANS or gold == self.INVALID_ANS:
                return False, predicted

            if self._is_pure_number(predicted) and self._is_pure_number(gold):
                pred_num = self._extract_last_number(predicted)
                gold_num = self._extract_last_number(gold)
                if pred_num is not None and gold_num is not None:
                    return pred_num == gold_num, predicted

            pred_norm = self._comparison_text(predicted)
            gold_norm = self._comparison_text(gold)
            if not pred_norm or not gold_norm:
                return False, predicted

            if pred_norm == gold_norm:
                return True, predicted

            similarity = SequenceMatcher(None, pred_norm, gold_norm).ratio()
            if similarity >= 0.85:
                return True, predicted

            pred_tokens = set(pred_norm.split())
            gold_tokens = set(gold_norm.split())
            if pred_tokens and gold_tokens:
                overlap = len(pred_tokens & gold_tokens) / max(len(pred_tokens), len(gold_tokens))
                if overlap >= 0.8:
                    return True, predicted

            return False, predicted
    return RCADataProcessor()