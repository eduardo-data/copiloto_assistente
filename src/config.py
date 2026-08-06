from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT_5", "")
    azure_api_key: str = os.getenv("AZURE_OPENAI_API_KEY_PRIMARY_5", "")
    azure_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION_5", "2025-04-01-preview")
    azure_model: str = os.getenv("AZURE_OPENAI_MODEL_5", "gpt-5.4-mini-ptu")
    top_k: int = int(os.getenv("TOP_K", "4"))
    max_turns: int = int(os.getenv("MAX_TURNS", "8"))

    def validate(self) -> None:
        missing = []
        if not self.azure_endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT_5")
        if not self.azure_api_key:
            missing.append("AZURE_OPENAI_API_KEY_PRIMARY_5")
        if not self.azure_model:
            missing.append("AZURE_OPENAI_MODEL_5")
        if missing:
            raise ValueError("Variáveis ausentes no .env: " + ", ".join(missing))
