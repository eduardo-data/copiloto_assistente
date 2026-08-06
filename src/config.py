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

    top_k: int = int(os.getenv("TOP_K", "4"))
    max_turns: int = int(os.getenv("MAX_TURNS", "8"))

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

        if missing:
            raise ValueError("Variáveis ausentes no .env: " + ", ".join(missing))
