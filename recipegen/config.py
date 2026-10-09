"""Runtime configuration for RecipeGen AI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from the local environment file."""

    gemini_api_key: str
    gemini_model: str


def get_settings() -> Settings:
    """Load Gemini settings from .env and fail clearly when required values are missing."""

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "").strip()

    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY in .env. Add your Google AI Studio key before running RecipeGen AI."
        )
    if not model:
        raise RuntimeError(
            "Missing GEMINI_MODEL in .env. Use a valid Gemini model name such as gemini-2.5-flash."
        )

    return Settings(gemini_api_key=api_key, gemini_model=model)
