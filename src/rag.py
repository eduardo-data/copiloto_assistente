from __future__ import annotations

import re
from collections import defaultdict
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
    kind: str
    section: str
    entities: tuple[str, ...]


class LocalRAG:
    """RAG local com chunking estruturado, busca vetorial e expansão por grafo.

    O modo ``naive`` retorna os chunks com maior similaridade semântica.
    O modo ``graph`` usa esses chunks como sementes e expande a recuperação para
    chunks relacionados por sequência, seção ou entidades compartilhadas.
    """

    TABLE_SEPARATOR_RE = re.compile(
        r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
    )
    HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    MONEY_RE = re.compile(
        r"(?:R\$\s?\d{1,3}(?:\.\d{3})*(?:,\d{2})?|\d+(?:,\d{2})?\s?(?:reais|real))",
        re.IGNORECASE,
    )
    CODE_RE = re.compile(r"\b[A-Z]{2,}[A-Z0-9_-]*\d+[A-Z0-9_-]*\b")
    TITLE_ENTITY_RE = re.compile(
        r"\b(?:[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç-]+(?:\s+|$)){1,4}"
    )

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
        self._chunk_index: dict[str, int] = {}
        self._adjacency: dict[str, set[str]] = defaultdict(set)
        self._edge_reasons: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.rebuild()

    def rebuild(self) -> None:
        self.chunks = self._load_chunks()
        self._chunk_index = {chunk.chunk_id: index for index, chunk in enumerate(self.chunks)}
        self._build_graph()

        if not self.chunks:
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            return

        self.embeddings = self.model.encode(
            [self._embedding_text(chunk) for chunk in self.chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def _embedding_text(self, chunk: Chunk) -> str:
        return (
            f"Documento: {chunk.source}\n"
            f"Seção: {chunk.section or 'Sem seção'}\n"
            f"Tipo: {chunk.kind}\n{chunk.text}"
        )

    def _load_chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        for path in sorted(self.docs_dir.glob("**/*")):
            if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
                continue

            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue

            relative_source = str(path.relative_to(self.docs_dir)).replace("\\", "/")
            document_chunks = self._structured_split(text)
            for index, item in enumerate(document_chunks):
                chunks.append(
                    Chunk(
                        chunk_id=f"{relative_source}::{index}",
                        source=relative_source,
                        text=item["text"],
                        start=item["start"],
                        end=item["end"],
                        kind=item["kind"],
                        section=item["section"],
                        entities=tuple(self._extract_entities(item["text"])),
                    )
                )
        return chunks

    def _structured_split(self, text: str) -> list[dict[str, str | int]]:
        """Divide o documento respeitando seções, tabelas e blocos de código.

        Tabelas Markdown e blocos cercados por ``` são atômicos: nunca são
        quebrados, mesmo quando ultrapassam ``chunk_size``.
        """
        normalized = "\n".join(line.rstrip() for line in text.splitlines())
        lines = normalized.splitlines(keepends=True)
        blocks: list[dict[str, str | int]] = []
        current_section = ""
        cursor = 0
        index = 0
        paragraph_lines: list[str] = []
        paragraph_start = 0

        def flush_paragraph() -> None:
            nonlocal paragraph_lines, paragraph_start
            raw = "".join(paragraph_lines).strip()
            if raw:
                blocks.append(
                    {
                        "text": raw,
                        "start": paragraph_start,
                        "end": paragraph_start + len("".join(paragraph_lines)),
                        "kind": "text",
                        "section": current_section,
                    }
                )
            paragraph_lines = []

        while index < len(lines):
            line = lines[index]
            stripped = line.strip()
            heading = self.HEADING_RE.match(stripped)

            if heading:
                flush_paragraph()
                current_section = heading.group(2).strip()
                blocks.append(
                    {
                        "text": stripped,
                        "start": cursor,
                        "end": cursor + len(line),
                        "kind": "heading",
                        "section": current_section,
                    }
                )
                cursor += len(line)
                index += 1
                paragraph_start = cursor
                continue

            if stripped.startswith("```"):
                flush_paragraph()
                block_start = cursor
                fenced = [line]
                cursor += len(line)
                index += 1
                while index < len(lines):
                    fenced_line = lines[index]
                    fenced.append(fenced_line)
                    cursor += len(fenced_line)
                    index += 1
                    if fenced_line.strip().startswith("```"):
                        break
                blocks.append(
                    {
                        "text": "".join(fenced).strip(),
                        "start": block_start,
                        "end": cursor,
                        "kind": "code",
                        "section": current_section,
                    }
                )
                paragraph_start = cursor
                continue

            is_table = (
                "|" in stripped
                and index + 1 < len(lines)
                and bool(self.TABLE_SEPARATOR_RE.match(lines[index + 1].strip()))
            )
            if is_table:
                flush_paragraph()
                table_start = cursor
                table_lines = [line, lines[index + 1]]
                cursor += len(line) + len(lines[index + 1])
                index += 2
                while index < len(lines):
                    table_line = lines[index]
                    if "|" not in table_line.strip() or not table_line.strip():
                        break
                    table_lines.append(table_line)
                    cursor += len(table_line)
                    index += 1
                blocks.append(
                    {
                        "text": "".join(table_lines).strip(),
                        "start": table_start,
                        "end": cursor,
                        "kind": "table",
                        "section": current_section,
                    }
                )
                paragraph_start = cursor
                continue

            if not paragraph_lines:
                paragraph_start = cursor
            paragraph_lines.append(line)
            cursor += len(line)
            index += 1

            if not stripped:
                flush_paragraph()
                paragraph_start = cursor

        flush_paragraph()

        chunks: list[dict[str, str | int]] = []
        text_buffer: list[dict[str, str | int]] = []

        def flush_text_buffer() -> None:
            nonlocal text_buffer
            if not text_buffer:
                return
            combined = "\n\n".join(str(block["text"]) for block in text_buffer).strip()
            section = str(text_buffer[0]["section"])
            start = int(text_buffer[0]["start"])
            end = int(text_buffer[-1]["end"])
            for part, local_start, local_end in self._split_long_text(combined):
                chunks.append(
                    {
                        "text": part,
                        "start": start + local_start,
                        "end": min(start + local_end, end),
                        "kind": "text",
                        "section": section,
                    }
                )
            text_buffer = []

        for block in blocks:
            if block["kind"] in {"table", "code"}:
                flush_text_buffer()
                chunks.append(block)
                continue

            if text_buffer and (
                block["section"] != text_buffer[0]["section"]
                or sum(len(str(item["text"])) for item in text_buffer)
                + len(str(block["text"]))
                > self.chunk_size
            ):
                flush_text_buffer()
            text_buffer.append(block)

        flush_text_buffer()
        return chunks

    def _split_long_text(self, text: str) -> list[tuple[str, int, int]]:
        if len(text) <= self.chunk_size:
            return [(text.strip(), 0, len(text))]

        parts: list[tuple[str, int, int]] = []
        start = 0
        while start < len(text):
            target_end = min(start + self.chunk_size, len(text))
            end = target_end
            if target_end < len(text):
                search_start = start + max(1, self.chunk_size // 2)
                candidates = [
                    text.rfind("\n\n", search_start, target_end),
                    text.rfind("\n", search_start, target_end),
                    text.rfind(". ", search_start, target_end),
                    text.rfind("; ", search_start, target_end),
                ]
                best_break = max(candidates)
                if best_break > start:
                    end = best_break + 1
            part = text[start:end].strip()
            if part:
                parts.append((part, start, end))
            if end >= len(text):
                break
            start = max(end - self.chunk_overlap, start + 1)
        return parts

    def _extract_entities(self, text: str) -> list[str]:
        entities: set[str] = set()
        entities.update(match.strip() for match in self.MONEY_RE.findall(text))
        entities.update(self.CODE_RE.findall(text))
        for match in self.TITLE_ENTITY_RE.findall(text):
            entity = " ".join(match.split()).strip(" -:,.|")
            if len(entity) >= 4:
                entities.add(entity)
        return sorted(entities)[:40]

    def _add_edge(self, left: str, right: str, reason: str) -> None:
        if left == right:
            return
        self._adjacency[left].add(right)
        self._adjacency[right].add(left)
        key = tuple(sorted((left, right)))
        self._edge_reasons[key].add(reason)

    def _build_graph(self) -> None:
        self._adjacency = defaultdict(set)
        self._edge_reasons = defaultdict(set)
        by_source: dict[str, list[Chunk]] = defaultdict(list)
        by_section: dict[tuple[str, str], list[Chunk]] = defaultdict(list)
        by_entity: dict[str, list[Chunk]] = defaultdict(list)

        for chunk in self.chunks:
            by_source[chunk.source].append(chunk)
            by_section[(chunk.source, chunk.section)].append(chunk)
            for entity in chunk.entities:
                by_entity[entity.casefold()].append(chunk)

        for source_chunks in by_source.values():
            for left, right in zip(source_chunks, source_chunks[1:]):
                self._add_edge(left.chunk_id, right.chunk_id, "sequência")

        for section_chunks in by_section.values():
            for index, left in enumerate(section_chunks):
                for right in section_chunks[index + 1 : index + 5]:
                    self._add_edge(left.chunk_id, right.chunk_id, "mesma seção")

        for entity, entity_chunks in by_entity.items():
            if len(entity) < 3 or len(entity_chunks) > 30:
                continue
            for index, left in enumerate(entity_chunks):
                for right in entity_chunks[index + 1 : index + 8]:
                    self._add_edge(left.chunk_id, right.chunk_id, f"entidade: {entity}")

    def _semantic_scores(self, query: str) -> np.ndarray:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype(np.float32)
        return self.embeddings @ query_embedding

    def search(self, query: str, top_k: int = 5, mode: str = "naive") -> list[dict]:
        clean_query = query.strip()
        if not clean_query or not self.chunks or self.embeddings.size == 0:
            return []

        scores = self._semantic_scores(clean_query)
        normalized_mode = mode.strip().lower()
        if normalized_mode == "graph":
            return self._graph_search(scores, top_k)
        return self._naive_search(scores, top_k)

    def _result(self, index: int, score: float, origin: str, relations: list[str] | None = None) -> dict:
        chunk = self.chunks[index]
        return {
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "text": chunk.text,
            "score": float(score),
            "start": chunk.start,
            "end": chunk.end,
            "kind": chunk.kind,
            "section": chunk.section,
            "entities": list(chunk.entities),
            "retrieval_origin": origin,
            "relations": relations or [],
        }

    def _naive_search(self, scores: np.ndarray, top_k: int) -> list[dict]:
        indexes = np.argsort(scores)[::-1][: max(1, top_k)]
        return [self._result(int(index), float(scores[index]), "similaridade vetorial") for index in indexes]

    def _graph_search(self, scores: np.ndarray, top_k: int) -> list[dict]:
        seed_count = min(len(self.chunks), max(2, top_k))
        seed_indexes = np.argsort(scores)[::-1][:seed_count]
        candidates: dict[int, tuple[float, str, list[str]]] = {}

        for seed_index in seed_indexes:
            seed_index = int(seed_index)
            seed_chunk = self.chunks[seed_index]
            candidates[seed_index] = (
                float(scores[seed_index]),
                "semente vetorial",
                [],
            )
            for neighbor_id in self._adjacency.get(seed_chunk.chunk_id, set()):
                neighbor_index = self._chunk_index[neighbor_id]
                edge_key = tuple(sorted((seed_chunk.chunk_id, neighbor_id)))
                reasons = sorted(self._edge_reasons.get(edge_key, set()))
                graph_score = max(
                    float(scores[neighbor_index]),
                    float(scores[seed_index]) * 0.82,
                )
                previous = candidates.get(neighbor_index)
                if previous is None or graph_score > previous[0]:
                    candidates[neighbor_index] = (
                        graph_score,
                        "expansão do grafo",
                        reasons,
                    )

        ordered = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)
        return [
            self._result(index, score, origin, relations)
            for index, (score, origin, relations) in ordered[: max(1, top_k)]
        ]

    def graph_stats(self) -> dict[str, int]:
        edge_count = sum(len(neighbors) for neighbors in self._adjacency.values()) // 2
        entity_count = len({entity.casefold() for chunk in self.chunks for entity in chunk.entities})
        return {
            "chunk_nodes": len(self.chunks),
            "edges": edge_count,
            "entities": entity_count,
        }

    def corpus_preview(self, max_chunks: int = 20) -> str:
        selected = self.chunks[:max_chunks]
        return "\n\n".join(
            f"FONTE: {chunk.source}\nSEÇÃO: {chunk.section}\nTIPO: {chunk.kind}\n{chunk.text}"
            for chunk in selected
        )
