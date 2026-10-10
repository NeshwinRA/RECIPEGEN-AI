"""Tests for the Gemini text-generation client."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors, types

from recipegen.config import Settings
from recipegen.llm_client import (
    GeminiAPIError,
    GeminiClient,
    GeminiEmptyResponseError,
    GeminiRequestError,
    GeminiServiceError,
)


class FakeModelsClient:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.call: dict[str, Any] | None = None

    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: types.GenerateContentConfig | types.GenerateContentConfigDict | None = None,
    ) -> Any:
        self.call = {"model": model, "contents": contents, "config": config}
        if self.error is not None:
            raise self.error
        return self.response


class FakeSDKClient:
    def __init__(self, models: FakeModelsClient) -> None:
        self.models = models
        self.closed = False

    def close(self) -> None:
        self.closed = True


def make_client(
    response: Any = None,
    error: Exception | None = None,
) -> tuple[GeminiClient, FakeSDKClient, FakeModelsClient]:
    models = FakeModelsClient(response=response, error=error)
    sdk_client = FakeSDKClient(models)
    settings = Settings(gemini_api_key="test-only-key", gemini_model="gemini-test")
    return GeminiClient(settings, sdk_client=sdk_client), sdk_client, models


def test_generate_text_uses_configured_model_and_returns_text() -> None:
    client, _, models = make_client(
        response=SimpleNamespace(text="Generated recipe text")
    )
    config = types.GenerateContentConfig(temperature=0.2)

    result = client.generate_text("Suggest a recipe.", config=config)

    assert result == "Generated recipe text"
    assert models.call == {
        "model": "gemini-test",
        "contents": "Suggest a recipe.",
        "config": config,
    }


@pytest.mark.parametrize("text", [None, "", " \n\t"])
def test_empty_or_blocked_response_raises_safe_error(text: str | None) -> None:
    client, _, _ = make_client(response=SimpleNamespace(text=text))

    with pytest.raises(GeminiEmptyResponseError, match="no usable text"):
        client.generate_text("Suggest a recipe.")


@pytest.mark.parametrize("prompt", ["", "  ", "\n\t", None, 123])
def test_invalid_prompt_is_rejected(prompt: Any) -> None:
    client, _, models = make_client(response=SimpleNamespace(text="unused"))

    with pytest.raises(ValueError, match="non-empty string"):
        client.generate_text(prompt)

    assert models.call is None


def test_client_error_is_translated_without_sensitive_details() -> None:
    secret = "test-only-key"
    prompt = "private recipe request"
    sdk_error = errors.ClientError(
        400,
        {"error": {"message": f"leaked {secret} {prompt}"}},
    )
    client, _, _ = make_client(error=sdk_error)

    with pytest.raises(GeminiRequestError) as raised:
        client.generate_text(prompt)

    assert str(raised.value) == "Gemini rejected the generation request (HTTP 400)."
    assert secret not in str(raised.value)
    assert prompt not in str(raised.value)


def test_server_error_is_translated_without_raw_sdk_message() -> None:
    sdk_error = errors.ServerError(
        503,
        {"error": {"message": "sensitive upstream response"}},
    )
    client, _, _ = make_client(error=sdk_error)

    with pytest.raises(GeminiServiceError) as raised:
        client.generate_text("Suggest a recipe.")

    assert str(raised.value) == "Gemini service error (HTTP 503)."
    assert "sensitive upstream response" not in str(raised.value)


def test_other_api_error_is_translated_without_raw_sdk_message() -> None:
    sdk_error = errors.APIError(
        429,
        {"error": {"message": "sensitive quota response"}},
    )
    client, _, _ = make_client(error=sdk_error)

    with pytest.raises(GeminiAPIError) as raised:
        client.generate_text("Suggest a recipe.")

    assert str(raised.value) == "Gemini API error (HTTP 429)."
    assert "sensitive quota response" not in str(raised.value)


def test_close_closes_the_underlying_sdk_client() -> None:
    client, sdk_client, _ = make_client()

    client.close()

    assert sdk_client.closed is True
