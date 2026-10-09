"""Minimal smoke test for the configured Gemini client."""

from __future__ import annotations

import sys
from pathlib import Path

from google import genai

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recipegen.config import get_settings


def main() -> None:
    """Send a one-line prompt to Gemini and print the reply."""

    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents="Say hello in one line",
    )

    reply = getattr(response, "text", None)
    if not reply:
        raise ValueError("Gemini returned an empty response.")

    print(reply)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Gemini request failed: {exc}") from exc
