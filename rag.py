import re
from pathlib import Path
from typing import List, Dict
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from .config import DOCUMENTS_DIR, TOP_K, CHUNK_SIZE, CHUNK_OVERLAP

class LocalRAG:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents: List[Dict] = []
        self.embeddings = None
        self.load_documents()

    def load_documents(self):
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        files = list(DOCUMENTS_DIR.glob("*.txt")) + list(DOCUMENTS_DIR.glob("*.md")) + list(DOCUMENTS_DIR.glob("*.pdf"))
        for file_path in files:
            text = self._read_file(file_path)
            if not text.strip():
                continue
            for index, chunk in enumerate(self._chunk_text(text)):
                self.documents.append({"id": f"{file_path.name}_{index}", "source": file_path.name, "chunk": chunk})
        if self.documents:
            self.embeddings = self.model.encode(
                [item["chunk"] for item in self.documents],
                normalize_embeddings=True
            )

    def _read_file(self, file_path: Path) -> str:
        if file_path.suffix.lower() == ".pdf":
            reader = PdfReader(str(file_path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        return file_path.read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _chunk_text(self, text: str) -> List[str]:
        text = self._clean_text(text)
        chunks, start = [], 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)
        return chunks

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        if not self.documents:
            return []
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(self.embeddings, query_embedding)
        indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for index in indices:
            item = dict(self.documents[index])
            item["score"] = float(scores[index])
            results.append(item)
        return results

    def context(self, query: str, top_k: int = TOP_K) -> Dict:
        results = self.search(query, top_k)
        return {
            "context": "\n\n".join(item["chunk"] for item in results),
            "sources": [{"source": item["source"], "score": round(item["score"], 4)} for item in results],
            "results": results,
        }

rag = LocalRAG()
