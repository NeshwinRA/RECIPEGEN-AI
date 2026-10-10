"""Streamlit interface for RecipeGen AI."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime

import streamlit as st

from recipegen.llm_client import GeminiClientError
from recipegen.models import GenerationResult, RecipeRequest
from recipegen.pipeline import (
    RecipeGenerationError,
    RecipeResponseValidationError,
    generate_recipe,
)
from recipegen.timing import timed_stage
from recipegen import storage
from recipegen.storage import SavedRecipe

DATABASE_PATH = storage.DEFAULT_DATABASE_PATH
PAGES = ("🍲 New Recipe", "🕘 History", "⭐ Saved Recipes")
_LOGGER = logging.getLogger(__name__)


def _display_time(label: str, value: int | None) -> None:
    display_value = f"{value} minutes" if value is not None else "Not provided"
    st.write(f"**{label}:** {display_value}")


def _creation_label(created_at: str | None) -> str:
    if not created_at:
        return "Date not available"
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError:
        return created_at
    return created.astimezone().strftime("%b %d, %Y %I:%M %p")


def _display_recipe(result: GenerationResult | SavedRecipe, recipe_id: int | None = None) -> None:
    recipe = result.recipe
    grounded = result.grounded
    source_recipe_ids = result.source_recipe_ids
    warnings = (
        result.validation_report.warnings
        if isinstance(result, GenerationResult)
        else result.validation_warnings
    )
    if isinstance(result, SavedRecipe):
        recipe_id = result.id

    st.subheader(recipe.recipe_name)
    time_columns = st.columns(4)
    with time_columns[0]:
        _display_time("Total time", recipe.total_time_min)
    with time_columns[1]:
        _display_time("Preparation time", recipe.prep_time_min)
    with time_columns[2]:
        _display_time("Cooking time", recipe.cook_time_min)
    with time_columns[3]:
        servings = str(recipe.servings) if recipe.servings is not None else "Not provided"
        st.write(f"**Servings:** {servings}")

    st.markdown("#### Ingredients")
    for ingredient in recipe.ingredients:
        st.markdown(f"- {ingredient}")

    st.markdown("#### Instructions")
    for step_number, step in enumerate(recipe.steps, start=1):
        st.markdown(f"{step_number}. {step}")

    if recipe.tags:
        st.markdown("#### Tags")
        st.write(", ".join(recipe.tags))

    if recipe.nutrition:
        st.markdown("#### Nutrition")
        st.write(recipe.nutrition)

    if grounded:
        source_ids = ", ".join(str(item) for item in source_recipe_ids)
        st.info(
            "This recipe is grounded in retrieved Food.com dataset records. "
            f"Source recipe IDs: {source_ids}"
        )
    else:
        st.info(
            "This recipe was generated without a usable Food.com source record "
            "and is not presented as dataset-grounded."
        )
    for warning in warnings:
        st.warning(warning)

    if recipe_id is not None:
        try:
            stored = storage.get_saved_recipe(recipe_id, db_path=DATABASE_PATH)
            if stored is None:
                st.warning("This recipe is no longer in saved history or favourites.")
                if st.session_state.get("opened_recipe_id") == recipe_id:
                    st.session_state.pop("opened_recipe_id", None)
                return
            is_favorite = stored.favorite_created_at is not None
            label = "Remove from Favorites" if is_favorite else "Save to Favorites"
            if st.button(label, key=f"favorite_{recipe_id}"):
                storage.set_favorite(
                    recipe_id,
                    not is_favorite,
                    db_path=DATABASE_PATH,
                )
                st.rerun()
        except (sqlite3.Error, OSError, ValueError):
            st.error("Could not update the saved status for this recipe.")


def _render_recipe_request() -> None:
    st.markdown(
        """
        <div class="recipe-hero">
            <div class="eyebrow">GOOD FOOD, LESS FUSS</div>
            <h1>Tell me what you have,<br>I'll cook the idea.</h1>
            <p>Simple recipes. Real ingredients. Perfect for bachelors.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label">TRY AN IDEA</div>', unsafe_allow_html=True)
    examples = (
        ("🥚  Eggs, bread", "Quick breakfast", "eggs, bread"),
        ("🍚  Rice, chicken", "Simple lunch", "rice, chicken"),
        ("🍜  Noodles, egg", "Under 15 minutes", "noodles, egg"),
        ("🥔  Potato, onion", "Budget friendly", "potato, onion"),
    )
    example_columns = st.columns(4)
    for column, (ingredients_label, description, ingredients_value) in zip(
        example_columns,
        examples,
    ):
        with column:
            if st.button(
                f"{ingredients_label}\n{description}",
                key=f"example_{ingredients_value.replace(', ', '_')}",
                use_container_width=True,
            ):
                st.session_state["ingredients_input"] = ingredients_value

    with st.form("recipe_request_form", border=False):
        st.text_area(
            "What ingredients do you have?",
            key="ingredients_input",
            placeholder="Type ingredients separated by commas — e.g. potato, onion, tomato",
            help="Add what you have on hand. Separate ingredients with commas.",
            height=150,
            label_visibility="visible",
        )
        preference_columns = st.columns(4)
        with preference_columns[0]:
            max_cooking_time = st.slider(
                "Cooking time",
                min_value=5,
                max_value=180,
                value=30,
                step=5,
                key="max_cooking_time",
                format="%d min",
            )
        with preference_columns[1]:
            servings = st.number_input(
                "Servings",
                min_value=1,
                max_value=10,
                value=2,
                step=1,
                key="servings",
            )
        with preference_columns[2]:
            dietary_preference = st.selectbox(
                "Dietary preference",
                ["Vegetarian", "Vegan", "Eggetarian", "Non-vegetarian"],
                key="dietary_preference",
            )
        with preference_columns[3]:
            spice_level = st.selectbox(
                "Spice level",
                ["Mild", "Medium", "Hot"],
                key="spice_level",
            )
        with st.expander("More preferences", expanded=False):
            advanced_columns = st.columns(2)
            with advanced_columns[0]:
                cuisine = st.selectbox(
                    "Cuisine",
                    [
                        "Any cuisine",
                        "South Indian",
                        "North Indian",
                        "Italian",
                        "Mexican",
                        "Chinese",
                        "Other",
                    ],
                    key="cuisine",
                )
            with advanced_columns[1]:
                equipment = st.selectbox(
                    "Available equipment",
                    [
                        "Any available equipment",
                        "Stovetop",
                        "Oven",
                        "Microwave",
                        "Pressure cooker",
                        "Air fryer",
                        "Single stove",
                    ],
                    key="equipment",
                )
            bachelor_mode = st.checkbox(
                "Bachelor-friendly mode",
                help="Prefer straightforward steps and minimal equipment.",
                key="bachelor_mode",
            )
        submitted = st.form_submit_button(
            "Generate My Recipe",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        ingredients = [
            item.strip()
            for item in st.session_state.get("ingredients_input", "").split(",")
            if item.strip()
        ]
        request = RecipeRequest(
            ingredients=ingredients,
            cuisine=cuisine,
            dietary_preference=dietary_preference,
            max_cooking_time=max_cooking_time,
            servings=int(servings),
            spice_level=spice_level,
            equipment=equipment,
            bachelor_mode=bachelor_mode,
        )
        st.session_state.pop("generation_result", None)
        st.session_state.pop("generation_error", None)
        st.session_state.pop("generation_save_error", None)
        st.session_state.pop("generation_record_id", None)
        try:
            result = generate_recipe(request)
            st.session_state["generation_result"] = result
            try:
                with timed_stage("sqlite_persistence"):
                    st.session_state["generation_record_id"] = storage.save_generation(
                        result,
                        request,
                        db_path=DATABASE_PATH,
                    )
            except Exception:
                st.session_state["generation_save_error"] = (
                    "The recipe was generated successfully but could not be saved to history."
                )
        except RecipeResponseValidationError as exc:
            for warning in exc.report.warnings:
                st.warning(warning)
            st.session_state["generation_error"] = (
                "The generated recipe did not pass validation: " + "; ".join(exc.report.errors)
            )
        except RecipeGenerationError as exc:
            st.session_state["generation_error"] = str(exc)
        except GeminiClientError as exc:
            _LOGGER.warning(
                "Gemini request failed exception_class=%s",
                type(exc).__name__,
            )
            st.session_state["generation_error"] = (
                "The recipe service could not complete the request. Please try again later."
            )
        except Exception as exc:
            _LOGGER.warning(
                "Recipe generation failed exception_class=%s",
                type(exc).__name__,
            )
            st.session_state["generation_error"] = (
                "An unexpected error occurred while generating the recipe. Please try again."
            )

    if st.session_state.get("generation_error"):
        st.error(st.session_state["generation_error"])
    if st.session_state.get("generation_save_error"):
        st.error(st.session_state["generation_save_error"])
    generation_result = st.session_state.get("generation_result")
    if isinstance(generation_result, GenerationResult):
        st.divider()
        record_id = st.session_state.get("generation_record_id")
        _display_recipe(generation_result, record_id if isinstance(record_id, int) else None)


def _render_saved_entry(entry: SavedRecipe, *, is_history: bool) -> None:
    created_at = entry.history_created_at if is_history else entry.favorite_created_at
    if created_at is None:
        created_at = entry.created_at
    with st.container(border=True):
        st.markdown(f"### {entry.recipe.recipe_name}")
        summary = [
            f"Total time: {entry.recipe.total_time_min} min"
            if entry.recipe.total_time_min is not None
            else "Total time: unknown",
            f"Servings: {entry.recipe.servings}"
            if entry.recipe.servings is not None
            else "Servings: unknown",
            _creation_label(created_at),
        ]
        st.caption(" · ".join(summary))
        controls = st.columns(2 if is_history else 1)
        with controls[0]:
            if st.button("Open Recipe", key=f"open_{'history' if is_history else 'favorite'}_{entry.id}"):
                st.session_state["opened_recipe_id"] = entry.id
        if is_history:
            with controls[1]:
                if st.button("Delete", key=f"delete_history_{entry.id}"):
                    try:
                        storage.delete_history_entry(entry.id, db_path=DATABASE_PATH)
                        if st.session_state.get("opened_recipe_id") == entry.id:
                            st.session_state.pop("opened_recipe_id", None)
                        st.rerun()
                    except (sqlite3.Error, OSError, ValueError):
                        st.error("Could not delete this history entry.")


def _render_opened_recipe() -> None:
    recipe_id = st.session_state.get("opened_recipe_id")
    if not isinstance(recipe_id, int):
        return
    st.divider()
    try:
        saved = storage.get_saved_recipe(recipe_id, db_path=DATABASE_PATH)
    except (sqlite3.Error, OSError, ValueError):
        st.error("Could not load this saved recipe.")
        return
    if saved is None:
        st.info("This saved recipe is no longer available.")
        st.session_state.pop("opened_recipe_id", None)
        return
    _display_recipe(saved)


def _render_history() -> None:
    st.title("History")
    try:
        history = storage.list_history(db_path=DATABASE_PATH)
    except (sqlite3.Error, OSError, ValueError):
        st.error("Could not load recipe history.")
        return

    if not history:
        st.info("No generated recipes in history yet.")
    else:
        for entry in history:
            _render_saved_entry(entry, is_history=True)
        if st.button("Clear History", key="clear_history"):
            st.session_state["confirm_clear_history"] = True
        if st.session_state.get("confirm_clear_history"):
            st.warning("Clear all history entries? Recipes still saved as favourites will remain.")
            confirmation_columns = st.columns(2)
            with confirmation_columns[0]:
                if st.button("Confirm Clear History", key="confirm_clear_history_button"):
                    try:
                        storage.clear_history(db_path=DATABASE_PATH)
                        st.session_state.pop("confirm_clear_history", None)
                        selected_id = st.session_state.get("opened_recipe_id")
                        if isinstance(selected_id, int):
                            still_saved = storage.get_saved_recipe(
                                selected_id,
                                db_path=DATABASE_PATH,
                            )
                            if still_saved is None:
                                st.session_state.pop("opened_recipe_id", None)
                        st.rerun()
                    except (sqlite3.Error, OSError, ValueError):
                        st.error("Could not clear recipe history.")
            with confirmation_columns[1]:
                if st.button("Cancel", key="cancel_clear_history"):
                    st.session_state.pop("confirm_clear_history", None)
                    st.rerun()
    _render_opened_recipe()


def _render_favorites() -> None:
    st.title("Saved Recipes")
    try:
        favorites = storage.list_favorites(db_path=DATABASE_PATH)
    except (sqlite3.Error, OSError, ValueError):
        st.error("Could not load saved recipes.")
        return
    if not favorites:
        st.info("No saved favourites yet.")
    else:
        for entry in favorites:
            _render_saved_entry(entry, is_history=False)
    _render_opened_recipe()


def main() -> None:
    st.set_page_config(page_title="RecipeGen AI", page_icon="🍲", layout="wide")
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp,
        [data-testid="stAppViewContainer"] {
            background: #090b09;
            color: #f4f5f0;
        }
        [data-testid="stSidebar"] {
            background: #0d100d;
            border-right: 1px solid #20251d;
            min-width: clamp(16rem, 22vw, 19rem);
            width: clamp(16rem, 22vw, 19rem);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #a8b09f;
        }
        .sidebar-brand {
            color: #f4f5f0;
            font-size: 1.45rem;
            font-weight: 800;
            letter-spacing: -.025em;
            line-height: 1.3;
            padding: .85rem .35rem .35rem;
        }
        .sidebar-brand strong { color: #c4f73b; }
        .sidebar-note {
            color: #aab39e;
            font-size: .92rem;
            line-height: 1.5;
            padding: 0 .35rem 1rem;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] > label {
            color: #9da694;
            font-size: .8rem;
            font-weight: 600;
            letter-spacing: .06em;
            margin: .35rem .25rem .7rem;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
            gap: .55rem;
            overflow: hidden;
            width: 100%;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            align-items: center;
            border: 1px solid transparent;
            border-radius: 10px;
            box-sizing: border-box;
            cursor: pointer;
            display: flex;
            margin: 0;
            min-height: 48px;
            padding: .65rem .75rem;
            transition: background-color .15s ease, border-color .15s ease;
            width: 100%;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: #141b12;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            background: #172115;
            border-color: #2d4321;
            box-shadow: inset 3px 0 #c4f73b;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label
        [data-testid="stMarkdownContainer"] p {
            color: #d9ded2;
            font-size: 17px;
            font-weight: 550;
            line-height: 1.35;
            margin: 0;
            overflow-wrap: anywhere;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked)
        [data-testid="stMarkdownContainer"] p {
            color: #c4f73b;
            font-weight: 700;
        }
        @media (max-width: 700px) {
            [data-testid="stSidebar"] {
                min-width: min(17rem, 85vw);
                width: min(17rem, 85vw);
            }
        }
        #MainMenu { visibility: hidden; }
        [data-testid="stRadio"] [role="radio"][aria-checked="true"] {
            border-color: #c4f73b;
        }
        [data-testid="stMain"] .block-container {
            max-width: 1000px;
            padding-top: 3.5rem;
        }
        .recipe-hero { margin: 1rem auto 2.25rem; max-width: 780px; }
        .recipe-hero .eyebrow,
        .section-label {
            color: #c4f73b;
            font-size: .72rem;
            font-weight: 700;
            letter-spacing: .16em;
        }
        .recipe-hero h1 {
            color: #f4f5f0;
            font-size: clamp(2.4rem, 6vw, 4.25rem);
            letter-spacing: -.055em;
            line-height: 1.07;
            margin: .8rem 0 1rem;
        }
        .recipe-hero p { color: #a8b09f; font-size: 1.1rem; }
        .section-label { margin: 0 0 .65rem; }
        [data-testid="stForm"] { border: 0; padding: 0; }
        [data-testid="stTextArea"] textarea,
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div {
            background: #111510;
            border-color: #30372c;
            border-radius: 12px;
        }
        [data-testid="stTextArea"] textarea {
            font-size: 1.05rem;
            min-height: 140px;
        }
        [data-testid="stSlider"] [role="slider"] {
            background: #c4f73b;
            border-color: #c4f73b;
        }
        [data-testid="stButton"] button,
        [data-testid="stFormSubmitButton"] button {
            border-radius: 12px;
            border-color: #30372c;
        }
        [data-testid="stButton"] button {
            background: #111510;
            color: #f4f5f0;
            min-height: 66px;
            text-align: left;
            white-space: pre-line;
        }
        [data-testid="stButton"] button:hover {
            border-color: #c4f73b;
            color: #c4f73b;
        }
        [data-testid="stFormSubmitButton"] button {
            background: #c4f73b;
            border-color: #c4f73b;
            color: #10130c;
            font-weight: 700;
            min-height: 3.25rem;
        }
        [data-testid="stFormSubmitButton"] button:hover {
            background: #d5ff68;
            border-color: #d5ff68;
            color: #10130c;
        }
        @media (max-width: 700px) {
            [data-testid="stMain"] .block-container { padding: 2rem 1rem; }
            .recipe-hero { margin-top: .5rem; }
            .recipe-hero h1 { font-size: 2.45rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">🍳 <strong>RecipeGen AI</strong></div>'
            '<div class="sidebar-note">Everyday cooking, made easy.</div>',
            unsafe_allow_html=True,
        )
        st.divider()
    selected_page = st.sidebar.radio("Navigate", PAGES, key="selected_page")
    if selected_page == PAGES[0]:
        _render_recipe_request()
    elif selected_page == PAGES[1]:
        _render_history()
    else:
        _render_favorites()


if __name__ == "__main__":
    main()
