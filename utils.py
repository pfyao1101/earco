from typing import Any, Dict, List, Optional, Tuple, TypedDict
from pathlib import Path
import json 
from dataclasses import dataclass

@dataclass
class IncidentRecord:
	question: str
	answer: str


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
	if not path.exists():
		raise FileNotFoundError(f"JSONL file not found: {path}")
	rows: List[Dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			rows.append(json.loads(line))
	return rows


def normalize_text(text: str) -> str:
	return " ".join(text.strip().split())


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		for row in rows:
			f.write(json.dumps(row, ensure_ascii=False) + "\n")