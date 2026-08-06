from __future__ import annotations

from openai import AzureOpenAI

from src.config import Settings


class AzureLLM:
    def __init__(self, settings: Settings) -> None:
        settings.validate()
        self.settings = settings
        self.client = AzureOpenAI(
            azure_endpoint=settings.azure_endpoint,
            api_key=settings.azure_api_key,
            api_version=settings.azure_api_version,
        )

    def complete(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.settings.azure_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""
