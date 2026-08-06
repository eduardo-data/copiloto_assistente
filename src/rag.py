from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str
    start: int
    end: int


class LocalRAG:
    def __init__(
        self,
        docs_dir: str = "data/docs",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        chunk_size: int = 900,
        chunk_overlap: int = 180,
    ) -> None:
        self.docs_dir = Path(docs_dir)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(embedding_model)
        self.chunks: list[Chunk] = []
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.rebuild()

    def rebuild(self) -> None:
        self.chunks = self._load_chunks()
        if not self.chunks:
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            return

        self.embeddings = self.model.encode(
            [chunk.text for chunk in self.chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def _load_chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        for path in sorted(self.docs_dir.glob("**/*")):
            if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
                continue

            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue

            relative_source = str(path.relative_to(self.docs_dir)).replace("\\", "/")
            for index, (part, start, end) in enumerate(self._split_text(text)):
                chunks.append(
                    Chunk(
                        chunk_id=f"{relative_source}::{index}",
                        source=relative_source,
                        text=part,
                        start=start,
                        end=end,
                    )
                )
        return chunks

    def _split_text(self, text: str) -> list[tuple[str, int, int]]:
        normalized = "\n".join(line.rstrip() for line in text.splitlines())
        parts: list[tuple[str, int, int]] = []
        start = 0

        while start < len(normalized):
            target_end = min(start + self.chunk_size, len(normalized))
            end = target_end

            if target_end < len(normalized):
                search_start = start + max(1, self.chunk_size // 2)
                candidates = [
                    normalized.rfind("\n\n", search_start, target_end),
                    normalized.rfind("\n", search_start, target_end),
                    normalized.rfind(". ", search_start, target_end),
                    normalized.rfind("; ", search_start, target_end),
                ]
                best_break = max(candidates)
                if best_break > start:
                    end = best_break + 1

            part = normalized[start:end].strip()
            if part:
                parts.append((part, start, end))

            if end >= len(normalized):
                break
            start = max(end - self.chunk_overlap, start + 1)

        return parts

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        clean_query = query.strip()
        if not clean_query or not self.chunks or self.embeddings.size == 0:
            return []

        query_embedding = self.model.encode(
            [clean_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype(np.float32)

        scores = self.embeddings @ query_embedding
        indexes = np.argsort(scores)[::-1][: max(1, top_k)]

        return [
            {
                "chunk_id": self.chunks[index].chunk_id,
                "source": self.chunks[index].source,
                "text": self.chunks[index].text,
                "score": float(scores[index]),
                "start": self.chunks[index].start,
                "end": self.chunks[index].end,
            }
            for index in indexes
        ]

    def corpus_preview(self, max_chunks: int = 20) -> str:
        selected = self.chunks[:max_chunks]
        return "\n\n".join(
            f"FONTE: {chunk.source}\n{chunk.text}" for chunk in selected
        )
