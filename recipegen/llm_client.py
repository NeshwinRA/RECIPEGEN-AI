"""Synchronous Gemini text generation client."""

from __future__ import annotations

import logging
from typing import Protocol

from google import genai
from google.genai import errors, types

from recipegen.config import Settings, get_settings
from recipegen.timing import timed_stage

_LOGGER = logging.getLogger(__name__)


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


class _InteractionsClient(Protocol):
    def create(
        self,
        *,
        model: str,
        input: str,
        response_format: dict[str, object],
    ) -> object: ...


class _InteractionsSDKClient(Protocol):
    interactions: _InteractionsClient

    def close(self) -> None: ...


def _recipe_response_config() -> types.GenerateContentConfig:
    """Build the strict JSON schema used for generated recipe responses."""

    string_array = types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(type=types.Type.STRING),
    )
    integer_or_null = types.Schema(
        type=types.Type.INTEGER,
        nullable=True,
    )
    number_array_or_null = types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(type=types.Type.NUMBER),
        nullable=True,
    )
    response_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "recipe_name": types.Schema(type=types.Type.STRING),
            "ingredients": string_array,
            "steps": string_array,
            "total_time_min": integer_or_null,
            "prep_time_min": integer_or_null,
            "cook_time_min": integer_or_null,
            "servings": integer_or_null,
            "tags": string_array,
            "nutrition": number_array_or_null,
            "source_recipe_ids": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.INTEGER),
            ),
        },
        required=[
            "recipe_name",
            "ingredients",
            "steps",
            "total_time_min",
            "prep_time_min",
            "cook_time_min",
            "servings",
            "tags",
            "nutrition",
            "source_recipe_ids",
        ],
    )
    return types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
    )


def _json_schema_from_sdk_schema(schema: types.Schema) -> dict[str, object]:
    result: dict[str, object] = {}
    if schema.type is not None:
        result["type"] = schema.type.value.lower()
    if schema.nullable is not None:
        result["nullable"] = schema.nullable
    if schema.items is not None:
        result["items"] = _json_schema_from_sdk_schema(schema.items)
    if schema.properties is not None:
        result["properties"] = {
            name: _json_schema_from_sdk_schema(property_schema)
            for name, property_schema in schema.properties.items()
        }
    if schema.required is not None:
        result["required"] = schema.required
    return result


def _recipe_interactions_response_format() -> dict[str, object]:
    """Build the Interactions API text response format from the recipe schema."""

    schema = _recipe_response_config().response_schema
    if not isinstance(schema, types.Schema):
        raise TypeError("Recipe response schema must be a Google GenAI Schema.")
    return {
        "type": "text",
        "mime_type": "application/json",
        "schema": _json_schema_from_sdk_schema(schema),
    }


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
            with timed_stage("gemini_generation"):
                response = self._client.models.generate_content(
                    model=self._settings.gemini_model,
                    contents=prompt,
                    config=config if config is not None else _recipe_response_config(),
                )
        except errors.ServerError as error:
            raise GeminiServiceError(
                f"Gemini service error (HTTP {error.code})."
            ) from None
        except errors.ClientError as error:
            error_code = error.code
            if (
                isinstance(error_code, int)
                and not isinstance(error_code, bool)
                and 100 <= error_code <= 599
            ):
                _LOGGER.warning(
                    "Gemini API client error exception_class=%s http_code=%d",
                    type(error).__name__,
                    error_code,
                )
            else:
                _LOGGER.warning(
                    "Gemini API client error exception_class=%s",
                    type(error).__name__,
                )
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


class GeminiInteractionsClient:
    """Opt-in Gemini client using the Interactions API for structured JSON."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        sdk_client: _InteractionsSDKClient | None = None,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._client = (
            sdk_client
            if sdk_client is not None
            else genai.Client(api_key=self._settings.gemini_api_key)
        )

    def generate_text(self, prompt: str) -> str:
        """Generate JSON text using Interactions while preserving the client contract."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        try:
            with timed_stage("gemini_interactions_generation"):
                response = self._client.interactions.create(
                    model=self._settings.gemini_model,
                    input=prompt,
                    response_format=_recipe_interactions_response_format(),
                )
        except errors.ServerError as error:
            raise GeminiServiceError(
                f"Gemini service error (HTTP {error.code})."
            ) from None
        except errors.ClientError as error:
            error_code = error.code
            if (
                isinstance(error_code, int)
                and not isinstance(error_code, bool)
                and 100 <= error_code <= 599
            ):
                _LOGGER.warning(
                    "Gemini Interactions API client error exception_class=%s http_code=%d",
                    type(error).__name__,
                    error_code,
                )
            else:
                _LOGGER.warning(
                    "Gemini Interactions API client error exception_class=%s",
                    type(error).__name__,
                )
            raise GeminiRequestError(
                f"Gemini rejected the generation request (HTTP {error.code})."
            ) from None
        except errors.APIError as error:
            raise GeminiAPIError(f"Gemini API error (HTTP {error.code}).") from None

        text = getattr(response, "output_text", None)
        if not isinstance(text, str) or not text.strip():
            raise GeminiEmptyResponseError(
                "Gemini returned no usable text; the response may have been empty or blocked."
            )
        return text

    def close(self) -> None:
        """Close the underlying SDK client and its network resources."""

        self._client.close()
