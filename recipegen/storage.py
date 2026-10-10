"""SQLite persistence for generated recipe history and favourites."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .models import GeneratedRecipe, GenerationResult, RecipeRequest

DEFAULT_DATABASE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "user" / "recipegen_history.sqlite3"
)
SQLITE_BUSY_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class SavedRecipe:
    """A stored generation with enough data to display it without regeneration."""

    id: int
    recipe: GeneratedRecipe
    request: RecipeRequest
    grounded: bool
    source_recipe_ids: list[int]
    validation_warnings: list[str]
    created_at: str
    history_created_at: str | None = None
    favorite_created_at: str | None = None


@contextmanager
def _connection(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        path,
        timeout=SQLITE_BUSY_TIMEOUT_SECONDS,
    )
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {int(SQLITE_BUSY_TIMEOUT_SECONDS * 1000)}")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS recipes (
                id INTEGER PRIMARY KEY,
                recipe_json TEXT NOT NULL,
                request_json TEXT NOT NULL,
                grounded INTEGER NOT NULL CHECK (grounded IN (0, 1)),
                source_recipe_ids_json TEXT NOT NULL,
                validation_warnings_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS history (
                recipe_id INTEGER PRIMARY KEY REFERENCES recipes(id),
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS favorites (
                recipe_id INTEGER PRIMARY KEY REFERENCES recipes(id),
                created_at TEXT NOT NULL
            );
            """
        )
        yield connection
    finally:
        connection.close()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _load_saved_recipe(
    row: sqlite3.Row,
    *,
    history_created_at: str | None = None,
    favorite_created_at: str | None = None,
) -> SavedRecipe:
    recipe_data = json.loads(row["recipe_json"])
    request_data = json.loads(row["request_json"])
    source_ids = json.loads(row["source_recipe_ids_json"])
    warnings = json.loads(row["validation_warnings_json"])
    if not isinstance(recipe_data, dict) or not isinstance(request_data, dict):
        raise ValueError("Stored recipe and request values must be JSON objects.")
    if not isinstance(source_ids, list) or not isinstance(warnings, list):
        raise ValueError("Stored source IDs and validation warnings must be JSON lists.")
    if any(not isinstance(warning, str) for warning in warnings):
        raise ValueError("Stored validation warnings must contain only strings.")

    return SavedRecipe(
        id=row["id"],
        recipe=GeneratedRecipe(**recipe_data),
        request=RecipeRequest(**request_data),
        grounded=bool(row["grounded"]),
        source_recipe_ids=source_ids,
        validation_warnings=warnings,
        created_at=row["created_at"],
        history_created_at=history_created_at,
        favorite_created_at=favorite_created_at,
    )


def save_generation(
    result: GenerationResult,
    request: RecipeRequest,
    db_path: str | Path = DEFAULT_DATABASE_PATH,
) -> int:
    """Persist a validated recipe generation and create its history entry."""

    if not isinstance(result, GenerationResult):
        raise TypeError("result must be a GenerationResult.")
    if not isinstance(request, RecipeRequest):
        raise TypeError("request must be a RecipeRequest.")
    if not result.validation_report.passed:
        raise ValueError("Cannot save a recipe that did not pass response validation.")
    if result.grounded != bool(result.source_recipe_ids):
        raise ValueError("Grounded status must match whether source recipe IDs are present.")
    if any(
        isinstance(recipe_id, bool) or not isinstance(recipe_id, int) or recipe_id <= 0
        for recipe_id in result.source_recipe_ids
    ):
        raise ValueError("Source recipe IDs must be positive integers.")

    now = _timestamp()
    recipe_json = _dump_json(asdict(result.recipe))
    request_json = _dump_json(asdict(request))
    source_ids_json = _dump_json(result.source_recipe_ids)
    warnings_json = _dump_json(result.validation_report.warnings)

    with _connection(db_path) as connection:
        with connection:
            cursor = connection.execute(
                """
                INSERT INTO recipes (
                    recipe_json, request_json, grounded, source_recipe_ids_json,
                    validation_warnings_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    recipe_json,
                    request_json,
                    int(result.grounded),
                    source_ids_json,
                    warnings_json,
                    now,
                ),
            )
            recipe_id = int(cursor.lastrowid)
            connection.execute(
                "INSERT INTO history (recipe_id, created_at) VALUES (?, ?)",
                (recipe_id, now),
            )
    return recipe_id


def list_history(
    limit: int = 50,
    db_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[SavedRecipe]:
    """Return newest history entries first."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer.")
    with _connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT recipes.*, history.created_at AS history_created_at
            FROM history
            JOIN recipes ON recipes.id = history.recipe_id
            ORDER BY history.created_at DESC, history.recipe_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        _load_saved_recipe(row, history_created_at=row["history_created_at"])
        for row in rows
    ]


def list_favorites(db_path: str | Path = DEFAULT_DATABASE_PATH) -> list[SavedRecipe]:
    """Return saved favourites newest-first, including their recipe and request data."""

    with _connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT recipes.*, favorites.created_at AS favorite_created_at
            FROM favorites
            JOIN recipes ON recipes.id = favorites.recipe_id
            ORDER BY favorites.created_at DESC, favorites.recipe_id DESC
            """
        ).fetchall()
    return [
        _load_saved_recipe(row, favorite_created_at=row["favorite_created_at"])
        for row in rows
    ]


def get_saved_recipe(
    recipe_id: int,
    db_path: str | Path = DEFAULT_DATABASE_PATH,
) -> SavedRecipe | None:
    """Return a saved generation by ID, or None when it does not exist."""

    _validate_recipe_id(recipe_id)
    with _connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT recipes.*,
                   history.created_at AS history_created_at,
                   favorites.created_at AS favorite_created_at
            FROM recipes
            LEFT JOIN history ON history.recipe_id = recipes.id
            LEFT JOIN favorites ON favorites.recipe_id = recipes.id
            WHERE recipes.id = ?
            """,
            (recipe_id,),
        ).fetchone()
    if row is None:
        return None
    return _load_saved_recipe(
        row,
        history_created_at=row["history_created_at"],
        favorite_created_at=row["favorite_created_at"],
    )


def _validate_recipe_id(recipe_id: int) -> None:
    if isinstance(recipe_id, bool) or not isinstance(recipe_id, int) or recipe_id < 1:
        raise ValueError("recipe_id must be a positive integer.")


def _delete_if_unreferenced(connection: sqlite3.Connection, recipe_id: int) -> None:
    connection.execute(
        """
        DELETE FROM recipes
        WHERE id = ?
          AND NOT EXISTS (SELECT 1 FROM history WHERE history.recipe_id = recipes.id)
          AND NOT EXISTS (SELECT 1 FROM favorites WHERE favorites.recipe_id = recipes.id)
        """,
        (recipe_id,),
    )


def set_favorite(
    recipe_id: int,
    saved: bool,
    db_path: str | Path = DEFAULT_DATABASE_PATH,
) -> None:
    """Save or remove a favourite without creating duplicate favourite rows."""

    _validate_recipe_id(recipe_id)
    if not isinstance(saved, bool):
        raise TypeError("saved must be a boolean.")
    now = _timestamp()
    with _connection(db_path) as connection:
        with connection:
            if saved:
                connection.execute(
                    "INSERT OR IGNORE INTO favorites (recipe_id, created_at) VALUES (?, ?)",
                    (recipe_id, now),
                )
            else:
                connection.execute("DELETE FROM favorites WHERE recipe_id = ?", (recipe_id,))
                _delete_if_unreferenced(connection, recipe_id)


def delete_history_entry(recipe_id: int, db_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
    """Delete one history reference and retain recipes still used as favourites."""

    _validate_recipe_id(recipe_id)
    with _connection(db_path) as connection:
        with connection:
            connection.execute("DELETE FROM history WHERE recipe_id = ?", (recipe_id,))
            _delete_if_unreferenced(connection, recipe_id)


def clear_history(db_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
    """Clear history and delete only recipes that are not saved favourites."""

    with _connection(db_path) as connection:
        with connection:
            connection.execute("DELETE FROM history")
            connection.execute(
                """
                DELETE FROM recipes
                WHERE NOT EXISTS (
                    SELECT 1 FROM favorites WHERE favorites.recipe_id = recipes.id
                )
                """
            )
