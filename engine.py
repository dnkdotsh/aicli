# aicli/engine.py

import abc
import requests
import sys
from typing import Dict, Any, List, Tuple

import config
import utils

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
    def build_chat_payload(self, messages: List[Dict[str, Any]], system_prompt: str | None, max_tokens: int | None, stream: bool, model: str) -> Dict[str, Any]:
        """Build the JSON payload for a chat request."""
        pass

    @abc.abstractmethod
    def parse_chat_response(self, response_data: Dict[str, Any]) -> str:
        """Extract the assistant's text response from the API response."""
        pass

    @abc.abstractmethod
    def fetch_available_models(self, task: str) -> List[str]:
        """Fetch a list of available models for a given task."""
        pass

class OpenAIEngine(AIEngine):
    """AI Engine implementation for OpenAI."""

    @property
    def name(self) -> str:
        return 'openai'

    def get_chat_url(self, model: str, stream: bool) -> str:
        return "https://api.openai.com/v1/chat/completions"

    def build_chat_payload(self, messages: List[Dict[str, Any]], system_prompt: str | None, max_tokens: int | None, stream: bool, model: str) -> Dict[str, Any]:
        all_messages = messages.copy()
        if system_prompt:
            all_messages.insert(0, {"role": "system", "content": system_prompt})

        payload = {
            "model": model,
            "messages": all_messages,
            "stream": stream
        }

        if max_tokens:
            # Legacy models ('gpt-4' but not 'gpt-4o', 'gpt-3.5-turbo') use 'max_tokens'.
            # Newer and future models default to 'max_completion_tokens' for better compatibility.
            if model.startswith('gpt-3.5-turbo') or (model.startswith('gpt-4') and not model.startswith('gpt-4o')):
                payload['max_tokens'] = max_tokens
            else:
                payload['max_completion_tokens'] = max_tokens
        return payload

    def parse_chat_response(self, response_data: Dict[str, Any]) -> str:
        if 'choices' in response_data:
            return utils.extract_text_from_message(response_data['choices'][0]['message'])
        return ""

    def fetch_available_models(self, task: str) -> List[str]:
        try:
            url = "https://api.openai.com/v1/models"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            model_list = response.json().get('data', [])
            model_ids = [m['id'] for m in model_list]

            if task == 'chat':
                chat_models = [mid for mid in model_ids if mid.startswith('gpt')]
                return sorted(chat_models)
            elif task == 'image':
                image_models = [mid for mid in model_ids if mid.startswith('dall-e') or 'image' in mid]
                return sorted(image_models)
        except requests.exceptions.RequestException as e:
            print(f"\nWarning: Could not fetch OpenAI model list ({e}).", file=sys.stderr)
        return []


class GeminiEngine(AIEngine):
    """AI Engine implementation for Google Gemini."""

    @property
    def name(self) -> str:
        return 'gemini'

    def get_chat_url(self, model: str, stream: bool) -> str:
        if stream:
            return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?key={self.api_key}&alt=sse"
        return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"

    def build_chat_payload(self, messages: List[Dict[str, Any]], system_prompt: str | None, max_tokens: int | None, stream: bool, model: str) -> Dict[str, Any]:
        payload = {"contents": messages}
        if system_prompt:
            payload['system_instruction'] = {'parts': [{'text': system_prompt}]}
        if max_tokens:
            payload['generationConfig'] = {'maxOutputTokens': max_tokens}
        return payload

    def parse_chat_response(self, response_data: Dict[str, Any]) -> str:
        try:
            if 'candidates' in response_data:
                return response_data['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            finish_reason = response_data.get('candidates', [{}])[0].get('finishReason', 'UNKNOWN')
            print(f"Warning: Could not extract Gemini response part. Finish reason: {finish_reason}", file=sys.stderr)
        return ""

    def fetch_available_models(self, task: str) -> List[str]:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()
            model_list = response.json().get('models', [])
            return sorted([m['name'].replace('models/', '') for m in model_list if 'generateContent' in m.get('supportedGenerationMethods', [])])
        except requests.exceptions.RequestException as e:
            print(f"\nWarning: Could not fetch Gemini model list ({e}).", file=sys.stderr)
        return []

def get_engine(engine_name: str, api_key: str) -> AIEngine:
    """Factory function to get an engine instance by name."""
    if engine_name == 'openai':
        return OpenAIEngine(api_key)
    if engine_name == 'gemini':
        return GeminiEngine(api_key)
    raise ValueError(f"Unknown engine: {engine_name}")
