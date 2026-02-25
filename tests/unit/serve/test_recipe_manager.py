"""Tests for recipe manager."""

from __future__ import annotations

import json
from pathlib import Path

from serve.recipe_manager import BUILTIN_RECIPES, RecipeManager


def test_list_builtin_recipes(tmp_path: Path) -> None:
    """Built-in recipes are listed."""
    manager = RecipeManager(tmp_path)
    recipes = manager.list_recipes()
    assert len(recipes) >= 4
    names = [r["name"] for r in recipes]
    assert "Coding Assistant" in names
    assert "Customer Support" in names


def test_get_builtin_recipe(tmp_path: Path) -> None:
    """Get a built-in recipe by name."""
    manager = RecipeManager(tmp_path)
    recipe = manager.get_recipe("coding_assistant")
    assert recipe["name"] == "Coding Assistant"
    assert recipe["method"] == "sft"
    assert "learning_rate" in recipe["hyperparameters"]


def test_import_recipe(tmp_path: Path) -> None:
    """Import a recipe file."""
    recipe_file = tmp_path / "test_recipe.json"
    recipe_file.write_text(json.dumps({
        "name": "test_recipe",
        "description": "Test",
        "method": "sft",
        "hyperparameters": {"epochs": 1},
    }))
    manager = RecipeManager(tmp_path)
    path = manager.import_recipe(str(recipe_file))
    assert Path(path).exists()
    recipes = manager.list_recipes()
    imported = [r for r in recipes if r["source"] == "imported"]
    assert len(imported) == 1


def test_get_unknown_recipe(tmp_path: Path) -> None:
    """Unknown recipe returns empty dict."""
    manager = RecipeManager(tmp_path)
    assert manager.get_recipe("nonexistent") == {}


def test_builtin_recipes_have_required_fields() -> None:
    """All built-in recipes have method and hyperparameters."""
    for key, recipe in BUILTIN_RECIPES.items():
        assert recipe.method
        assert recipe.hyperparameters
        assert recipe.description
