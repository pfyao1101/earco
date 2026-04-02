import faiss
import numpy as np
from utils import normalize_text, IncidentRecord
from typing import Any, List, Optional, Tuple
from sentence_transformers import SentenceTransformer

class RAGRetriever:
	def __init__(self, model_name: str) -> None:
		self.embedder = SentenceTransformer(model_name)
		self.records: List[IncidentRecord] = []
		self.embeddings: Optional[np.ndarray] = None
		self.index: Optional[Any] = None

	@staticmethod
	def _to_doc_text(record: IncidentRecord) -> str:
		# Keep close to paper style: combine incident title/summary equivalent.
		return f"Question: {normalize_text(record.question)}\nAnswer: {normalize_text(record.answer)}"

	def build(self, records: List[IncidentRecord]) -> None:
		if not records:
			raise ValueError("No records provided to build retriever index.")

		self.records = records
		corpus = [self._to_doc_text(rec) for rec in records]
		vectors = self.embedder.encode(corpus, convert_to_numpy=True, show_progress_bar=True)
		vectors = vectors.astype("float32")
		self.embeddings = vectors

		# index = faiss.IndexFlatL2(vectors.shape[1])
		dim, measure = vectors.shape[1], faiss.METRIC_L2
		param = 'Flat'
		index = faiss.index_factory(dim, param, measure)
		index.add(vectors)
		self.index = index

	def search(self, query: str, top_k: int) -> List[Tuple[IncidentRecord, float]]:
		if not self.records or self.embeddings is None:
			raise RuntimeError("Retriever index is not built. Call build() first.")

		top_k = min(top_k, len(self.records))
		q_vec = self.embedder.encode([normalize_text(query)], convert_to_numpy=True).astype("float32")
		
		distances, indices = self.index.search(q_vec, top_k)
		result: List[Tuple[IncidentRecord, float]] = []
		for idx, dist in zip(indices[0], distances[0]):
			result.append((self.records[int(idx)], float(dist)))
		return result
	
if __name__ == "__main__":
	records = [
		IncidentRecord(question="What is the capital of France?", answer="Paris"),
		IncidentRecord(question="What is the largest mammal?", answer="Blue whale"),
		IncidentRecord(question="Who wrote '1984'?", answer="George Orwell"),
	]
	retriever = RAGRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
	retriever.build(records)
	results = retriever.search("What is the capital city of France?", top_k=2)
	for rec, dist in results:
		print(f"Question: {rec.question}, Answer: {rec.answer}, Distance: {dist:.4f}")