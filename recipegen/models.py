"""Domain models for RecipeGen AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Recipe:
    """Normalized recipe record aligned with RecipeGen AI expectations."""

    recipe_id: int
    recipe_name: str
    ingredients: list[str]
    steps: list[str]
    total_time_min: int | None = None
    prep_time_min: int | None = None
    cook_time_min: int | None = None
    servings: int | None = None
    tags: list[str] = field(default_factory=list)
    nutrition: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert a validated recipe to a dictionary for CSV export."""

        return {
            "recipe_id": self.recipe_id,
            "recipe_name": self.recipe_name,
            "ingredients": self.ingredients,
            "steps": self.steps,
            "total_time_min": self.total_time_min,
            "prep_time_min": self.prep_time_min,
            "cook_time_min": self.cook_time_min,
            "servings": self.servings,
            "tags": self.tags,
            "nutrition": self.nutrition,
        }


@dataclass
class RecipeRequest:
    """User request model for recipe generation from the frontend."""

    ingredients: list[str]
    cuisine: str
    dietary_preference: str
    max_cooking_time: int
    servings: int
    spice_level: str
    equipment: str
    bachelor_mode: bool = False
