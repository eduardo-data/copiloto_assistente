from __future__ import annotations

from typing import Any

import requests
from openai import AzureOpenAI

from src.config import Settings


class AzureLLM:
    def __init__(self, settings: Settings) -> None:
        settings.validate()
        self.settings = settings
        self.client: AzureOpenAI | None = None

        if settings.llm_transport == "azure_sdk":
            self.client = AzureOpenAI(
                azure_endpoint=settings.azure_endpoint,
                api_key=settings.azure_api_key,
                api_version=settings.azure_api_version,
                timeout=settings.request_timeout_seconds,
            )

    def complete(self, system: str, user: str) -> str:
        if self.settings.llm_transport == "direct":
            return self._complete_direct(system, user)
        return self._complete_azure_sdk(system, user)

    def _complete_azure_sdk(self, system: str, user: str) -> str:
        if self.client is None:
            raise RuntimeError("Cliente Azure OpenAI não foi inicializado.")

        response = self.client.chat.completions.create(
            model=self.settings.azure_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""

    def _complete_direct(self, system: str, user: str) -> str:
        headers = {"Content-Type": "application/json"}

        if self.settings.direct_auth_type == "bearer":
            headers["Authorization"] = f"Bearer {self.settings.azure_api_key}"
        else:
            headers["api-key"] = self.settings.azure_api_key

        payload = {
            "model": self.settings.azure_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        response = requests.post(
            self.settings.direct_url,
            headers=headers,
            json=payload,
            timeout=self.settings.request_timeout_seconds,
        )

        if not response.ok:
            content_type = response.headers.get("content-type", "")
            detail = response.text.strip()
            if "application/json" in content_type:
                try:
                    detail = str(response.json())
                except ValueError:
                    pass
            raise RuntimeError(
                f"Gateway retornou HTTP {response.status_code}: {detail[:1000]}"
            )

        try:
            data: dict[str, Any] = response.json()
            return data["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "O gateway respondeu, mas o JSON não segue o formato "
                "choices[0].message.content. Resposta: "
                f"{response.text[:1000]}"
            ) from exc
