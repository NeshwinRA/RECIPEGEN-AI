"""Tests for the Gemini text-generation client."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors, types

from recipegen.config import Settings
from recipegen.llm_client import (
    GeminiAPIError,
    GeminiClient,
    GeminiEmptyResponseError,
    GeminiInteractionsClient,
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


class FakeInteractions:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.call: dict[str, Any] | None = None

    def create(
        self,
        *,
        model: str,
        input: str,
        response_format: dict[str, object],
    ) -> Any:
        self.call = {
            "model": model,
            "input": input,
            "response_format": response_format,
        }
        if self.error is not None:
            raise self.error
        return self.response


class FakeInteractionsSDKClient:
    def __init__(self, interactions: FakeInteractions) -> None:
        self.interactions = interactions
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


def make_interactions_client(
    response: Any = None,
    error: Exception | None = None,
) -> tuple[GeminiInteractionsClient, FakeInteractionsSDKClient, FakeInteractions]:
    interactions = FakeInteractions(response=response, error=error)
    sdk_client = FakeInteractionsSDKClient(interactions)
    settings = Settings(gemini_api_key="test-only-key", gemini_model="gemini-test")
    return (
        GeminiInteractionsClient(settings, sdk_client=sdk_client),
        sdk_client,
        interactions,
    )


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


def test_generate_text_defaults_to_json_response_schema() -> None:
    client, _, models = make_client(
        response=SimpleNamespace(text='{"recipe_name":"Soup"}')
    )

    assert client.generate_text("Suggest a recipe.") == '{"recipe_name":"Soup"}'

    config = models.call["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert config.response_mime_type == "application/json"
    schema = config.response_schema
    assert isinstance(schema, types.Schema)
    assert schema.type == types.Type.OBJECT
    assert schema.additional_properties is None
    assert set(schema.properties) == {
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
    }
    assert set(schema.required) == set(schema.properties)
    assert schema.properties["total_time_min"].nullable is True
    assert schema.properties["nutrition"].nullable is True
    assert schema.properties["source_recipe_ids"].items.type == types.Type.INTEGER


def test_interactions_client_uses_documented_json_response_format() -> None:
    private_prompt = "private recipe request"
    response_text = '{"recipe_name":"Soup"}'
    client, _, interactions = make_interactions_client(
        response=SimpleNamespace(output_text=response_text)
    )

    assert client.generate_text(private_prompt) == response_text

    assert interactions.call == {
        "model": "gemini-test",
        "input": private_prompt,
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": {
                "type": "object",
                "properties": {
                    "recipe_name": {"type": "string"},
                    "ingredients": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "steps": {"type": "array", "items": {"type": "string"}},
                    "total_time_min": {"type": "integer", "nullable": True},
                    "prep_time_min": {"type": "integer", "nullable": True},
                    "cook_time_min": {"type": "integer", "nullable": True},
                    "servings": {"type": "integer", "nullable": True},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "nutrition": {
                        "type": "array",
                        "items": {"type": "number"},
                        "nullable": True,
                    },
                    "source_recipe_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
                "required": [
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
            },
        },
    }


@pytest.mark.parametrize(
    ("sdk_error", "expected_error"),
    [
        (
            errors.ClientError(400, {"error": {"message": "sensitive details"}}),
            GeminiRequestError,
        ),
        (
            errors.ServerError(503, {"error": {"message": "sensitive details"}}),
            GeminiServiceError,
        ),
        (
            errors.APIError(429, {"error": {"message": "sensitive details"}}),
            GeminiAPIError,
        ),
    ],
)
def test_interactions_client_translates_sdk_errors_without_sensitive_details(
    sdk_error: Exception,
    expected_error: type[Exception],
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, _, _ = make_interactions_client(error=sdk_error)

    with caplog.at_level(logging.WARNING, logger="recipegen.llm_client"):
        with pytest.raises(expected_error):
            client.generate_text("private recipe request")

    assert "sensitive details" not in caplog.text
    assert "private recipe request" not in caplog.text
    if isinstance(sdk_error, errors.ClientError):
        assert "exception_class=ClientError" in caplog.text
        assert "http_code=400" in caplog.text


@pytest.mark.parametrize("output_text", [None, "", " \n"])
def test_interactions_client_rejects_empty_output_text(output_text: str | None) -> None:
    client, _, _ = make_interactions_client(
        response=SimpleNamespace(output_text=output_text)
    )

    with pytest.raises(GeminiEmptyResponseError, match="no usable text"):
        client.generate_text("Suggest a recipe.")


def test_generation_timing_does_not_log_prompt_or_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    private_prompt = "private ingredient request"
    private_response = "private generated recipe"
    client, _, _ = make_client(response=SimpleNamespace(text=private_response))

    with caplog.at_level(logging.INFO, logger="recipegen.timing"):
        assert client.generate_text(private_prompt) == private_response

    assert "stage=gemini_generation" in caplog.text
    assert private_prompt not in caplog.text
    assert private_response not in caplog.text


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


def test_client_error_logs_numeric_code_and_translates_without_sensitive_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "test-only-key"
    prompt = "private recipe request"
    sensitive_message = f"leaked {secret} {prompt}"
    sdk_error = errors.ClientError(
        400,
        {"error": {"message": sensitive_message}},
    )
    client, _, _ = make_client(error=sdk_error)

    with caplog.at_level(logging.WARNING, logger="recipegen.llm_client"):
        with pytest.raises(GeminiRequestError) as raised:
            client.generate_text(prompt)

    assert str(raised.value) == "Gemini rejected the generation request (HTTP 400)."
    assert secret not in str(raised.value)
    assert prompt not in str(raised.value)
    assert "Gemini API client error" in caplog.text
    assert "exception_class=ClientError" in caplog.text
    assert "http_code=400" in caplog.text
    assert sensitive_message not in caplog.text
    assert secret not in caplog.text
    assert prompt not in caplog.text


@pytest.mark.parametrize(
    "error_code",
    [None, "400", True, 99, 600],
)
def test_client_error_omits_missing_or_invalid_http_code_safely(
    error_code: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sdk_error = errors.ClientError(
        400,
        {"error": {"message": "sensitive upstream message"}},
    )
    sdk_error.code = error_code
    client, _, _ = make_client(error=sdk_error)

    with caplog.at_level(logging.WARNING, logger="recipegen.llm_client"):
        with pytest.raises(GeminiRequestError):
            client.generate_text("private prompt")

    assert "Gemini API client error" in caplog.text
    assert "exception_class=ClientError" in caplog.text
    assert "http_code=" not in caplog.text
    assert "sensitive upstream message" not in caplog.text
    assert "private prompt" not in caplog.text


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
