"""
eARCO reproduction pipeline (RAG + PromptWizard + SLM API).

This script provides a runnable pipeline:
1) Process training data and build a retrieval index.
2) Configure PromptWizard parameters and optimize instruction.
3) Assemble final RAG prompt with optimized instruction/expert profile.
4) Send the composed prompt to an OpenAI-compatible SLM endpoint.
"""

from __future__ import annotations
import argparse
import json 
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import yaml
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv(override=True)


from utils import read_jsonl, IncidentRecord, write_jsonl
from faiss_RAG import RAGRetriever
from SLMCLient import SLMClient
from RCADataProcessor import build_rca_data_processor
import template

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="eARCO baseline pipeline")
	parser.add_argument("--question", type=str, default="")
	parser.add_argument("--run-test", action="store_true")
	# data related args
	parser.add_argument("--train-path", type=Path, default=Path("data/train.jsonl"))
	parser.add_argument("--test-path", type=Path, default=Path("data/test.jsonl"))
	parser.add_argument("--output", type=Path, default=Path("outputs/predictions.jsonl"))
	# slm related args
	parser.add_argument("--slm-model", type=str, default=os.getenv("SLM_MODEL"))
	parser.add_argument("--api-base", type=str, default=os.getenv("OPENAI_API_BASE"))
	parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"))
	parser.add_argument("--temperature", type=float, default=0.0)
	# RAG related args
	parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
	parser.add_argument("--top-k", type=int, default=10)
	# PromptWizard related args
	parser.add_argument("--disable-promptwizard", action="store_true")
	parser.add_argument("--pw-mutate-refine-iterations", type=int, default=3)
	parser.add_argument("--pw-mutation-rounds", type=int, default=3)
	parser.add_argument("--pw-refine-task-eg-iterations", type=int, default=3)
	parser.add_argument("--pw-question-batch-size", type=int, default=5)
	parser.add_argument("--pw-min-correct-count", type=int, default=3)
	parser.add_argument("--pw-few-shot-count", type=int, default=5)
	parser.add_argument("--pw-seen-set-size", type=int, default=25)
	return parser.parse_args()

@dataclass
class PipelineConfig:
	train_path: Path = Path("data/train.jsonl")
	test_path: Path = Path("data/test.jsonl")
	# slm related configs
	slm_model: str = os.getenv("SLM_MODEL")
	api_base: Optional[str] = os.getenv("OPENAI_API_BASE")
	api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
	temperature: float = 0.0
	# RAG related configs
	embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
	top_k: int = 10
	# promptwizard related configs
	promptwizard_enabled: bool = True
	promptwizard_mutate_refine_iterations: int = 3
	promptwizard_mutation_rounds: int = 3
	promptwizard_refine_task_eg_iterations: int = 3
	promptwizard_question_batch_size: int = 5
	promptwizard_min_correct_count: int = 3
	promptwizard_few_shot_count: int = 5
	promptwizard_seen_set_size: int = 25
	promptwizard_use_examples: bool = True
	promptwizard_generate_reasoning: bool = True
	promptwizard_resolve_tie_criteria: str = "max"
	promptwizard_task_description: str = (
		"You are an expert incident analyst specializing in root cause analysis for storage, network, and distributed systems faults."
	)
	promptwizard_base_instruction: str = "Think step by step and keep the root-cause diagnosis concise."
	promptwizard_answer_format: str = (
		"For each question, provide concise reasoning and wrap only the final root-cause statement "
		"between <ANS_START> and <ANS_END>."
	)
	promptwizard_work_dir: Path = Path("outputs/promptwizard")
	promptwizard_prompt_config_template: Path = Path(
		"configs/earco_promptopt_config.yaml"
	)
	promptwizard_setup_config_template: Path = Path(
		"configs/earco_setup_config.yaml"
	)


class PromptWizardAdapter:
	"""PromptWizard adapter for scenario-3 style prompt optimization."""

	def __init__(self, config: PipelineConfig) -> None:
		self.config = config
		self.project_root = Path(__file__).resolve().parent
		self.best_prompt: Optional[str] = None
		self.expert_profile: Optional[str] = None
		self.error_message: Optional[str] = None

	@staticmethod
	def _load_yaml(path: Path) -> Dict[str, Any]:
		if not path.exists():
			return {}
		with path.open("r", encoding="utf-8") as f:
			data = yaml.safe_load(f)
		return data if isinstance(data, dict) else {}

	@staticmethod
	def _dump_yaml(path: Path, data: Dict[str, Any]) -> None:
		path.parent.mkdir(parents=True, exist_ok=True)
		with path.open("w", encoding="utf-8") as f:
			yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)

	def _load_promptwizard_modules(self) -> Tuple[Any, Any]:
		promptwizard_root = self.project_root / "PromptWizard"
		if not promptwizard_root.exists():
			raise FileNotFoundError(
				f"PromptWizard directory not found: {promptwizard_root}"
			)

		promptwizard_root_str = str(promptwizard_root)
		if promptwizard_root_str not in sys.path:
			sys.path.insert(0, promptwizard_root_str)

		from promptwizard.glue.promptopt.instantiate import GluePromptOpt
		from promptwizard.glue.promptopt.techniques.common_logic import DatasetSpecificProcessing

		return GluePromptOpt, DatasetSpecificProcessing

	def _build_processor(self, dataset_base_cls: Any) -> Any:
		return build_rca_data_processor(dataset_base_cls)

	def _build_promptopt_config(self, dataset_size: int) -> Dict[str, Any]:
		cfg = self._load_yaml(self.project_root / self.config.promptwizard_prompt_config_template)

		seen_set_size = max(1, min(self.config.promptwizard_seen_set_size, dataset_size))
		few_shot_count = max(0, min(self.config.promptwizard_few_shot_count, seen_set_size))

		default_cfg: Dict[str, Any] = {
			"prompt_technique_name": "critique_n_refine",
			"unique_model_id": self.config.slm_model,
			"style_variation": 10,
			"questions_batch_size": self.config.promptwizard_question_batch_size,
			"min_correct_count": self.config.promptwizard_min_correct_count,
			"max_eval_batches": 6,
			"top_n": 1,
			"mutation_rounds": self.config.promptwizard_mutation_rounds,
			"refine_instruction": True,
			"mutate_refine_iterations": self.config.promptwizard_mutate_refine_iterations,
			"refine_task_eg_iterations": self.config.promptwizard_refine_task_eg_iterations,
			"task_description": self.config.promptwizard_task_description,
			"base_instruction": self.config.promptwizard_base_instruction,
			"answer_format": self.config.promptwizard_answer_format,
			"seen_set_size": seen_set_size,
			"few_shot_count": few_shot_count,
			"generate_reasoning": self.config.promptwizard_generate_reasoning,
			"generate_expert_identity": True,
			"generate_intent_keywords": True,
			"num_train_examples": seen_set_size,
		}

		merged = {**default_cfg, **cfg}
		merged["prompt_technique_name"] = "critique_n_refine"
		merged["unique_model_id"] = self.config.slm_model
		merged["questions_batch_size"] = self.config.promptwizard_question_batch_size
		merged["min_correct_count"] = self.config.promptwizard_min_correct_count
		merged["mutation_rounds"] = self.config.promptwizard_mutation_rounds
		merged["mutate_refine_iterations"] = self.config.promptwizard_mutate_refine_iterations
		merged["refine_task_eg_iterations"] = self.config.promptwizard_refine_task_eg_iterations
		merged["task_description"] = self.config.promptwizard_task_description
		merged["base_instruction"] = self.config.promptwizard_base_instruction
		merged["answer_format"] = self.config.promptwizard_answer_format
		merged["seen_set_size"] = seen_set_size
		merged["few_shot_count"] = few_shot_count
		merged["generate_reasoning"] = self.config.promptwizard_generate_reasoning
		merged["num_train_examples"] = seen_set_size

		return merged

	def _build_setup_config(self) -> Dict[str, Any]:
		cfg = self._load_yaml(self.project_root / self.config.promptwizard_setup_config_template)

		default_cfg: Dict[str, Any] = {
			"assistant_llm": {"prompt_opt": self.config.slm_model},
			"dir_info": {
				"base_dir": str((self.project_root / self.config.promptwizard_work_dir / "logs").as_posix()),
				"log_dir_name": "glue_logs",
			},
			"experiment_name": "earco",
			"mode": "offline",
			"description": "PromptWizard optimization for eARCO",
		}

		merged = {**default_cfg, **cfg}
		merged["assistant_llm"] = {"prompt_opt": self.config.slm_model}
		dir_info = merged.get("dir_info", {})
		if not isinstance(dir_info, dict):
			dir_info = {}
		dir_info["base_dir"] = str((self.project_root / self.config.promptwizard_work_dir / "logs").as_posix())
		dir_info.setdefault("log_dir_name", "glue_logs")
		merged["dir_info"] = dir_info
		merged["experiment_name"] = "earco"
		merged["mode"] = "offline"

		return merged

	def prepare(self, records: List[IncidentRecord]) -> None:
		if not self.config.promptwizard_enabled:
			return

		if not records:
			self.error_message = "PromptWizard skipped: empty train records."
			return

		if not self.config.api_base or not self.config.api_key:
			self.error_message = "PromptWizard skipped: API_BASE/API_KEY not configured."
			return

		try:
			os.environ["OPENAI_MODEL_NAME"] = self.config.slm_model
			GluePromptOpt, DatasetSpecificProcessing = self._load_promptwizard_modules()
			processor = self._build_processor(DatasetSpecificProcessing)

			work_dir = self.project_root / self.config.promptwizard_work_dir
			work_dir.mkdir(parents=True, exist_ok=True)

			train_for_pw = work_dir / "train_promptwizard.jsonl"
			processor.dataset_to_jsonl(
				str(train_for_pw),
				task="earco",
				dataset=[{"question": r.question, "answer": r.answer} for r in records],
			)

			promptopt_cfg_path = self.project_root / self.config.promptwizard_prompt_config_template
			setup_cfg_path = self.project_root / self.config.promptwizard_setup_config_template

			self._dump_yaml(promptopt_cfg_path, self._build_promptopt_config(len(records)))
			self._dump_yaml(setup_cfg_path, self._build_setup_config())

			gp = GluePromptOpt(
				str(promptopt_cfg_path),
				str(setup_cfg_path),
				str(train_for_pw),
				processor,
			)
			self.best_prompt, self.expert_profile = gp.get_best_prompt(
				use_examples=self.config.promptwizard_use_examples,
				run_without_train_examples=False,
				generate_synthetic_examples=False,
				resolve_tie_criteria=self.config.promptwizard_resolve_tie_criteria,
			)
		except Exception as exc:
			self.error_message = f"PromptWizard optimization failed: {exc}"

	def optimize_prompt(
		self,
		prompt: str,
		question: str,
		retrieved_examples: List[Tuple[IncidentRecord, float]],
	) -> str:
		_ = question, retrieved_examples
		if not self.best_prompt:
			return "", prompt

		system_prompt: List[str] = []
		user_prompt: List[str] = []
		if self.expert_profile:
			system_prompt.append(f"## Role{self.expert_profile}")
		system_prompt.append(template.SYSTEM_INSTRUCTION_TEMPLATE)
		user_prompt.append(f"## Instruction\n{self.best_prompt}")
		user_prompt.append(prompt)
		return "\n\n".join(system_prompt), "\n\n".join(user_prompt)

	def status(self) -> Dict[str, Any]:
		return {
			"enabled": self.config.promptwizard_enabled,
			"optimized": bool(self.best_prompt),
			"error": self.error_message,
		}


def build_prompt(question: str, retrieved_examples: List[Tuple[IncidentRecord, float]]) -> str:
	shots: List[str] = []
	for i, (rec, dist) in enumerate(retrieved_examples, start=1):
		shots.append(
			"\n".join(
				[
					f"Example {i} (retrieval distance={dist:.4f}):",
					f"Q: {rec.question}",
					f"A: {rec.answer}",
				]
			)
		)

	shot_block = "\n\n".join(shots)
	return (
		"## Retrieved examples \n Use the retrieved examples as references and produce a clear RCA conclusion.\n\n"
		f"{shot_block}\n\n"
		f"## Target Incident Question:\nQ: {question}\n\n"
		"Provide concise reasoning and wrap only the final root-cause statement in <ANS_START> and <ANS_END>."
	)

class EARCOPipeline:
	def __init__(self, config: PipelineConfig) -> None:
		self.config = config
		self.retriever = RAGRetriever(config.embedding_model)
		self.promptwizard = PromptWizardAdapter(config)
		
		self._slm_client = SLMClient(
			api_base=config.api_base,
			api_key=config.api_key,
			model=config.slm_model,
			temperature=config.temperature,
		)

	def prepare(self) -> None:
		rows = read_jsonl(self.config.train_path)
		records = [
			IncidentRecord(question=row.get("input", ""), answer=row.get("output", ""))
			for row in rows
		]
		records = [r for r in records if r.question and r.answer]
		self.retriever.build(records)
		self.promptwizard.prepare(records)

	def _run_sequential(self, question: str) -> Dict[str, Any]:
		retrieved = self.retriever.search(question, self.config.top_k)
		prompt = build_prompt(question, retrieved)
		system_prompt, user_prompt = self.promptwizard.optimize_prompt(prompt, question, retrieved)

		result: Dict[str, Any] = {
			"question": question,
			"retrieved": [
				{"question": rec.question, "answer": rec.answer, "distance": dist}
				for rec, dist in retrieved
			],
			"promptwizard": self.promptwizard.status(),
			"system_prompt": system_prompt,
			"user_prompt": user_prompt,
		}

		if self._slm_client is None:
			raise RuntimeError(
				"SLM client not configured. Provide API_BASE and API_KEY (or CLI args)."
			)
		result["prediction"] = self._slm_client.generate(system_prompt, user_prompt)
		return result

	def run_one(self, question: str) -> Dict[str, Any]:
		return self._run_sequential(question)

	def run_testset(
		self,
	) -> List[Dict[str, Any]]:
		rows = read_jsonl(self.config.test_path)

		outputs: List[Dict[str, Any]] = []
		for row in tqdm(rows, desc="Running testset"):
			q = row.get("input", "")
			if not q:
				continue
			result = self.run_one(q)
			result["gold_answer"] = row.get("output", "")
			outputs.append(result)
		return outputs


def main() -> None:
	args = parse_args()
	config = PipelineConfig(
		train_path=args.train_path,
		test_path=args.test_path,
		slm_model=args.slm_model,
		api_base=args.api_base,
		api_key=args.api_key,
		temperature=args.temperature,
		embedding_model=args.embedding_model,
		top_k=args.top_k,
		promptwizard_enabled=not args.disable_promptwizard,
		promptwizard_mutate_refine_iterations=args.pw_mutate_refine_iterations,
		promptwizard_mutation_rounds=args.pw_mutation_rounds,
		promptwizard_refine_task_eg_iterations=args.pw_refine_task_eg_iterations,
		promptwizard_question_batch_size=args.pw_question_batch_size,
		promptwizard_min_correct_count=args.pw_min_correct_count,
		promptwizard_few_shot_count=args.pw_few_shot_count,
		promptwizard_seen_set_size=args.pw_seen_set_size,
	)

	pipeline = EARCOPipeline(config)
	pipeline.prepare()

	if args.question:
		result = pipeline.run_one(args.question)
		print(json.dumps(result, ensure_ascii=False, indent=2))
		return

	if args.run_test:
		results = pipeline.run_testset()
		write_jsonl(args.output, results)
		print(f"Saved {len(results)} records to: {args.output}")
		return

	print("No action specified. Use --question or --run-test.")


if __name__ == "__main__":
	main()