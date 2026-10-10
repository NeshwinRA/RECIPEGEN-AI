"""Synchronous Gemini text generation client."""

from __future__ import annotations

from typing import Protocol

from google import genai
from google.genai import errors, types

from recipegen.config import Settings, get_settings


class GeminiClientError(RuntimeError):
    """Base exception for safe, application-level Gemini client failures."""


class GeminiRequestError(GeminiClientError):
    """The Gemini API rejected the generation request."""


class GeminiServiceError(GeminiClientError):
    """The Gemini API service failed to process the request."""


class GeminiAPIError(GeminiClientError):
    """The Gemini API returned an otherwise unclassified API error."""


class GeminiEmptyResponseError(GeminiClientError):
    """Gemini returned no usable text."""


class _ModelsClient(Protocol):
    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: types.GenerateContentConfig | types.GenerateContentConfigDict | None = None,
    ) -> types.GenerateContentResponse: ...


class _SDKClient(Protocol):
    models: _ModelsClient

    def close(self) -> None: ...


class GeminiClient:
    """Generate text from Gemini with injectable settings and SDK client."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        sdk_client: _SDKClient | None = None,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._client = (
            sdk_client
            if sdk_client is not None
            else genai.Client(api_key=self._settings.gemini_api_key)
        )

    def generate_text(
        self,
        prompt: str,
        *,
        config: types.GenerateContentConfig | types.GenerateContentConfigDict | None = None,
    ) -> str:
        """Generate non-empty text for a prompt, translating SDK API errors safely."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        try:
            response = self._client.models.generate_content(
                model=self._settings.gemini_model,
                contents=prompt,
                config=config,
            )
        except errors.ServerError as error:
            raise GeminiServiceError(
                f"Gemini service error (HTTP {error.code})."
            ) from None
        except errors.ClientError as error:
            raise GeminiRequestError(
                f"Gemini rejected the generation request (HTTP {error.code})."
            ) from None
        except errors.APIError as error:
            raise GeminiAPIError(f"Gemini API error (HTTP {error.code}).") from None

        text = response.text
        if not isinstance(text, str) or not text.strip():
            raise GeminiEmptyResponseError(
                "Gemini returned no usable text; the response may have been empty or blocked."
            )
        return text

    def close(self) -> None:
        """Close the underlying SDK client and its network resources."""

        self._client.close()
