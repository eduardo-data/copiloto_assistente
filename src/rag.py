from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    source: str
    text: str


class LocalRAG:
    def __init__(self, docs_dir: str = "data/docs") -> None:
        self.docs_dir = Path(docs_dir)
        self.chunks = self._load_chunks()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks]) if self.chunks else None

    def _load_chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        for path in self.docs_dir.glob("**/*"):
            if path.suffix.lower() not in {".txt", ".md"}:
                continue
            text = path.read_text(encoding="utf-8")
            parts = [p.strip() for p in text.split("\n\n") if p.strip()]
            chunks.extend(Chunk(source=path.name, text=part) for part in parts)
        return chunks

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        if not self.chunks or self.matrix is None:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        indexes = scores.argsort()[::-1][:top_k]
        return [
            {"source": self.chunks[i].source, "text": self.chunks[i].text, "score": float(scores[i])}
            for i in indexes
            if scores[i] > 0
        ]
