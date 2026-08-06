"""Componentes do Copiloto Assistente.

Este módulo aplica uma proteção de consulta ao ``LocalRAG``: quando a interface
envia uma consulta composta por pergunta atual e histórico, somente a pergunta
atual é transformada em embedding. O histórico continua disponível para o LLM,
mas não contamina o ranking vetorial da nova pergunta.
"""

from __future__ import annotations

import re
from typing import Any

from src.rag import LocalRAG

_CURRENT_QUESTION_RE = re.compile(
    r"PERGUNTA ATUAL(?: DO CLIENTE)?\s*:\s*(.+?)(?=\n\s*\n[A-ZÁÉÍÓÚÂÊÔÃÕÇ _-]+:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_ORIGINAL_SEARCH = LocalRAG.search


def _extract_current_question(query: str) -> str:
    """Extrai a pergunta atual de uma consulta composta.

    Caso a consulta não use o formato estruturado da interface, o texto original
    é mantido para preservar compatibilidade com chamadas diretas ao retriever.
    """
    clean_query = query.strip()
    match = _CURRENT_QUESTION_RE.search(clean_query)
    if not match:
        return clean_query
    current_question = match.group(1).strip()
    return current_question or clean_query


def _question_focused_search(
    self: LocalRAG,
    query: str,
    top_k: int = 5,
    mode: str = "naive",
) -> list[dict[str, Any]]:
    """Executa a recuperação usando somente a pergunta atual no embedding."""
    embedding_query = _extract_current_question(query)
    results = _ORIGINAL_SEARCH(self, embedding_query, top_k=top_k, mode=mode)
    for result in results:
        result["embedding_query"] = embedding_query
    return results


LocalRAG.search = _question_focused_search

__all__ = ["LocalRAG"]
