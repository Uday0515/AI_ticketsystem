from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

load_dotenv()


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, provider: str | None = None, timeout: float = 30):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "mock")).lower()
        self.timeout = timeout
        self.model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    @property
    def configured(self) -> bool:
        return self.provider == "mock" or bool(os.getenv("GROQ_API_KEY"))

    def complete(self, messages: list[dict[str, str]], temperature: float = 0) -> str:
        if self.provider == "mock":
            return self._mock(messages)
        if self.provider != "groq":
            raise LLMError(f"Unsupported LLM_PROVIDER: {self.provider}")

        try:
            headers = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY', '')}"}
            return self._request(
                "https://api.groq.com/openai/v1/chat/completions",
                messages,
                temperature,
                headers,
            )
        except HTTPError as exc:
            if exc.code == 403:
                message = (
                    "Groq rejected the request (HTTP 403). Check that GROQ_API_KEY "
                    "is valid and that the account can use the configured model."
                )
            else:
                message = f"{self.provider} request failed: {exc}"
            raise LLMError(message) from exc
        except (URLError, TimeoutError, ValueError) as exc:
            raise LLMError(f"{self.provider} request failed: {exc}") from exc

    def reachable(self) -> bool:
        if self.provider == "mock":
            return True
        try:
            if not self.configured:
                return False
            return True
        except (HTTPError, URLError, TimeoutError):
            return False

    def _request(
        self,
        url: str,
        messages: list[dict[str, str]],
        temperature: float,
        headers: dict[str, str],
    ) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
        ).encode()
        request_headers = {"Content-Type": "application/json", **headers}
        request = Request(url, data=body, headers=request_headers, method="POST")
        with urlopen(request, timeout=self.timeout) as response:
            payload: dict[str, Any] = json.loads(response.read().decode())
        return str(payload["choices"][0]["message"]["content"])

    def _mock(self, messages: list[dict[str, str]]) -> str:
        prompt = messages[-1]["content"].lower()
        if "generate one pandas" in prompt or "repair" in prompt:
            if "currently open" in prompt:
                expression = "{'count': int((df['status'] == 'Open').sum())}"
            elif "resolved the most" in prompt:
                expression = "df[(df['status'] == 'Resolved') & (df['created_at'].dt.year == 2024) & (df['created_at'].dt.month == 3)].groupby('agent_id').size().sort_values(ascending=False).head(1).to_dict()"
            elif "critical" in prompt and "12 hours" in prompt:
                expression = (
                    "df[(df['priority'] == 'Critical') & "
                    "((~df['is_resolved']) | "
                    "(df['resolution_time_hrs'] > 12))]"
                    "[['ticket_id', 'status', 'resolution_time_hrs']]"
                )
            elif "average customer rating" in prompt:
                expression = "{'average_rating': float(df.loc[df['category'] == 'Technical', 'customer_rating'].mean())}"
            else:
                expression = "{'count': 0}"
            return json.dumps({"expression": expression})
        return f"Result: {messages[-1]['content']}"
