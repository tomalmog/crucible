"""Integration tests for training recipe management end-to-end workflow.

Covers RecipeManager SDK methods and CLI recipe subcommands using real
file I/O against a temporary directory.
"""

from __future__ import annotations

import json
from pathlib import Path

from cli.main import main
from serve.recipe_manager import RecipeManager


def test_list_includes_builtins(tmp_path: Path) -> None:
    """Listing recipes should include at least four built-in entries."""
    manager = RecipeManager(tmp_path)

    recipes = manager.list_recipes()

    assert len(recipes) >= 4
    for r in recipes:
        assert r["source"] == "builtin"


def test_get_builtin(tmp_path: Path) -> None:
    """Getting a built-in recipe should return method and hyperparameters."""
    manager = RecipeManager(tmp_path)

    recipe = manager.get_recipe("coding_assistant")

    assert "method" in recipe
    assert "hyperparameters" in recipe


def test_import_and_list(tmp_path: Path) -> None:
    """Importing a recipe JSON should make it appear in the recipe list."""
    manager = RecipeManager(tmp_path)
    recipe_file = tmp_path / "custom.json"
    recipe_file.write_text(json.dumps({
        "name": "my_recipe",
        "description": "Test",
        "method": "sft",
        "hyperparameters": {"lr": 0.01},
    }))

    manager.import_recipe(str(recipe_file))
    recipes = manager.list_recipes()

    imported = [r for r in recipes if r["source"] == "imported"]
    assert len(imported) >= 1
    names = [r["name"] for r in imported]
    assert "my_recipe" in names


def test_cli_list(tmp_path: Path, capsys) -> None:
    """CLI 'recipe list' should exit 0 and include built-in recipe names."""
    exit_code = main(["--data-root", str(tmp_path), "recipe", "list"])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "Coding Assistant" in captured


def test_cli_apply(tmp_path: Path) -> None:
    """CLI 'recipe apply coding_assistant' should exit 0."""
    exit_code = main([
        "--data-root", str(tmp_path),
        "recipe", "apply", "coding_assistant",
    ])

    assert exit_code == 0


def test_cli_import(tmp_path: Path) -> None:
    """CLI 'recipe import' should exit 0 for a valid recipe file."""
    recipe_path = tmp_path / "cli_recipe.json"
    recipe_path.write_text(json.dumps({
        "name": "my_recipe",
        "description": "Test",
        "method": "sft",
        "hyperparameters": {"lr": 0.01},
    }))

    exit_code = main([
        "--data-root", str(tmp_path),
        "recipe", "import", str(recipe_path),
    ])

    assert exit_code == 0
