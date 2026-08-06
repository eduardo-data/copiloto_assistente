from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    azure_endpoint: str = os.getenv("AZURE_STRUCTURING_ENDPOINT", "").strip()
    azure_api_key: str = os.getenv("AZURE_STRUCTURING_KEY", "").strip()
    azure_deployment: str = os.getenv(
        "AZURE_STRUCTURING_DEPLOYMENT", "gpt-4o-mini-2"
    ).strip()
    azure_api_version: str = os.getenv(
        "AZURE_STRUCTURING_VERSION_COMPLETIONS", "2024-02-15-preview"
    ).strip()
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ).strip()
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "180"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    max_turns: int = int(os.getenv("MAX_TURNS", "12"))

    def validate(self) -> None:
        missing: list[str] = []

        if not self.azure_endpoint:
            missing.append("AZURE_STRUCTURING_ENDPOINT")
        if not self.azure_api_key:
            missing.append("AZURE_STRUCTURING_KEY")
        if not self.azure_deployment:
            missing.append("AZURE_STRUCTURING_DEPLOYMENT")
        if not self.azure_api_version:
            missing.append("AZURE_STRUCTURING_VERSION_COMPLETIONS")
        if not self.embedding_model:
            missing.append("EMBEDDING_MODEL")

        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE deve ser maior que zero.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP deve ser menor que CHUNK_SIZE.")

        if missing:
            raise ValueError("Variáveis ausentes no .env: " + ", ".join(missing))
