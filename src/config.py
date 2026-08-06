from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT_5", "").strip()
    azure_api_key: str = os.getenv("AZURE_OPENAI_API_KEY_PRIMARY_5", "").strip()
    azure_api_version: str = os.getenv(
        "AZURE_OPENAI_API_VERSION_5", "2025-04-01-preview"
    ).strip()
    azure_model: str = os.getenv(
        "AZURE_OPENAI_MODEL_5", "gpt-5.4-mini-ptu"
    ).strip()

    llm_transport: str = os.getenv("LLM_TRANSPORT", "azure_sdk").strip().lower()
    direct_url: str = os.getenv("AZURE_OPENAI_DIRECT_URL_5", "").strip()
    direct_auth_type: str = os.getenv("AZURE_OPENAI_AUTH_TYPE_5", "api-key").strip().lower()
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))

    top_k: int = int(os.getenv("TOP_K", "4"))
    max_turns: int = int(os.getenv("MAX_TURNS", "8"))

    def validate(self) -> None:
        missing: list[str] = []

        if not self.azure_api_key:
            missing.append("AZURE_OPENAI_API_KEY_PRIMARY_5")
        if not self.azure_model:
            missing.append("AZURE_OPENAI_MODEL_5")

        if self.llm_transport == "direct":
            if not self.direct_url:
                missing.append("AZURE_OPENAI_DIRECT_URL_5")
        elif self.llm_transport == "azure_sdk":
            if not self.azure_endpoint:
                missing.append("AZURE_OPENAI_ENDPOINT_5")
        else:
            raise ValueError(
                "LLM_TRANSPORT deve ser 'azure_sdk' ou 'direct'."
            )

        if missing:
            raise ValueError("Variáveis ausentes no .env: " + ", ".join(missing))
