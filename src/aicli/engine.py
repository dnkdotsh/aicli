# aicli/engine.py
# aicli: A command-line interface for interacting with AI models.
# Copyright (C) 2025 Dank A. Saurus

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY;
# without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


import abc
from typing import Any

import requests

from .logger import log
from .settings import settings


class AIEngine(abc.ABC):
    """Abstract base class for an AI engine provider."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of the engine (e.g., 'openai')."""
        pass

    @abc.abstractmethod
    def get_chat_url(self, model: str, stream: bool) -> str:
        """Get the API endpoint URL for chat completions."""
        pass

    @abc.abstractmethod
    def build_chat_payload(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None,
        max_tokens: int | None,
        stream: bool,
        model: str,
    ) -> dict[str, Any]:
        """Build the JSON payload for a chat request."""
        pass

    @abc.abstractmethod
    def parse_chat_response(self, response_data: dict[str, Any]) -> str:
        """Extract the assistant's text response from the API response."""
        pass

    @abc.abstractmethod
    def fetch_available_models(self, task: str) -> list[str]:
        """Fetch a list of available models for a given task."""
        pass


class OpenAIEngine(AIEngine):
    """AI Engine implementation for OpenAI."""

    @property
    def name(self) -> str:
        return "openai"

    def get_chat_url(self, model: str, stream: bool) -> str:
        return "https://api.openai.com/v1/chat/completions"

    def build_chat_payload(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None,
        max_tokens: int | None,
        stream: bool,
        model: str,
    ) -> dict[str, Any]:
        all_messages = messages.copy()
        if system_prompt:
            all_messages.insert(0, {"role": "system", "content": system_prompt})

        payload = {"model": model, "messages": all_messages, "stream": stream}

        if stream:
            payload["stream_options"] = {"include_usage": True}

        if max_tokens:
            # Legacy models ('gpt-4' but not 'gpt-4o', 'gpt-3.5-turbo') use 'max_tokens'.
            # Newer and future models default to 'max_completion_tokens' for better compatibility.
            if model.startswith("gpt-3.5-turbo") or (
                model.startswith("gpt-4") and not model.startswith("gpt-4o")
            ):
                payload["max_tokens"] = max_tokens
            else:
                payload["max_completion_tokens"] = max_tokens
        return payload

    def parse_chat_response(self, response_data: dict[str, Any]) -> str:
        # Local import to break circular dependency: engine -> utils -> engine
        from . import utils

        if "choices" in response_data and response_data["choices"]:
            message = response_data["choices"][0].get("message")
            if message:
                return utils.extract_text_from_message(message)
        return ""

    def fetch_available_models(self, task: str) -> list[str]:
        try:
            url = "https://api.openai.com/v1/models"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(
                url, headers=headers, timeout=settings["api_timeout"]
            )
            response.raise_for_status()
            model_list = response.json().get("data", [])
            model_ids = [m["id"] for m in model_list]

            if task == "chat":
                chat_models = [mid for mid in model_ids if mid.startswith("gpt")]
                return sorted(chat_models)
            elif task == "image":
                image_models = [
                    mid
                    for mid in model_ids
                    if mid.startswith("dall-e") or "image" in mid
                ]
                return sorted(image_models)
        except requests.exceptions.RequestException as e:
            log.warning("Could not fetch OpenAI model list (%s).", e)
        return []


class GeminiEngine(AIEngine):
    """AI Engine implementation for Google Gemini."""

    @property
    def name(self) -> str:
        return "gemini"

    def get_chat_url(self, model: str, stream: bool) -> str:
        if stream:
            return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?key={self.api_key}&alt=sse"
        return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"

    def build_chat_payload(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None,
        max_tokens: int | None,
        stream: bool,
        model: str,
    ) -> dict[str, Any]:
        payload = {"contents": messages}
        if system_prompt:
            payload["system_instruction"] = {"parts": [{"text": system_prompt}]}
        if max_tokens:
            payload["generationConfig"] = {"maxOutputTokens": max_tokens}
        return payload

    def parse_chat_response(self, response_data: dict[str, Any]) -> str:
        """Safely extracts text from a Gemini API response."""
        try:
            # Check for content and parts before accessing them
            candidate = response_data.get("candidates", [{}])[0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            if parts and "text" in parts[0]:
                return parts[0]["text"]

            # If no text is found, log the reason but return an empty string
            finish_reason = candidate.get("finishReason", "UNKNOWN")
            if finish_reason not in ["UNKNOWN", None]:
                log.info(
                    "Gemini response finished with reason '%s' but no text content.",
                    finish_reason,
                )
            else:
                log.warning(
                    "Could not extract Gemini response part or finish reason. Full response: %s",
                    response_data,
                )
            return ""
        except (KeyError, IndexError):
            log.warning(
                "Could not parse Gemini response due to unexpected structure. Full response: %s",
                response_data,
            )
            return ""

    def fetch_available_models(self, task: str) -> list[str]:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
            response = requests.get(url, timeout=settings["api_timeout"])
            response.raise_for_status()
            model_list = response.json().get("models", [])
            return sorted(
                [
                    m["name"].replace("models/", "")
                    for m in model_list
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                ]
            )
        except requests.exceptions.RequestException as e:
            log.warning("Could not fetch Gemini model list (%s).", e)
        return []


def get_engine(engine_name: str, api_key: str) -> AIEngine:
    """Factory function to get an engine instance by name."""
    if engine_name == "openai":
        return OpenAIEngine(api_key)
    if engine_name == "gemini":
        return GeminiEngine(api_key)
    raise ValueError(f"Unknown engine: {engine_name}")
