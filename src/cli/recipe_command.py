"""Recipe command wiring for Forge CLI."""

from __future__ import annotations

import argparse
import json
from typing import Any

from serve.recipe_manager import RecipeManager
from store.dataset_sdk import ForgeClient


def run_recipe_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Handle recipe subcommand dispatch."""
    manager = RecipeManager(client._config.data_root)
    subcmd = args.recipe_subcommand
    if subcmd == "list":
        return _run_list(manager)
    if subcmd == "export":
        return _run_export(manager, args.run_id, args.output)
    if subcmd == "import":
        return _run_import(manager, args.path)
    if subcmd == "apply":
        return _run_apply(manager, args.name)
    return 1


def _run_list(manager: RecipeManager) -> int:
    recipes = manager.list_recipes()
    for r in recipes:
        print(f"{r['name']}  method={r['method']}  source={r['source']}")
    if not recipes:
        print("No recipes found.")
    return 0


def _run_export(manager: RecipeManager, run_id: str, output: str) -> int:
    path = manager.export_recipe(run_id, output)
    print(f"recipe_path={path}")
    return 0


def _run_import(manager: RecipeManager, path: str) -> int:
    dest = manager.import_recipe(path)
    print(f"imported={dest}")
    return 0


def _run_apply(manager: RecipeManager, name: str) -> int:
    recipe = manager.get_recipe(name)
    if not recipe:
        print(f"Recipe '{name}' not found.")
        return 1
    print(json.dumps(recipe, indent=2, default=str))
    return 0


def add_recipe_command(subparsers: Any) -> None:
    """Register recipe subcommand."""
    parser = subparsers.add_parser("recipe", help="Training recipe management")
    sub = parser.add_subparsers(dest="recipe_subcommand", required=True)
    sub.add_parser("list", help="List available recipes")
    export_p = sub.add_parser("export", help="Export run config as recipe")
    export_p.add_argument("--run-id", required=True, help="Training run ID")
    export_p.add_argument("--output", required=True, help="Output file path")
    import_p = sub.add_parser("import", help="Import a recipe file")
    import_p.add_argument("path", help="Path to recipe JSON file")
    apply_p = sub.add_parser("apply", help="Show recipe configuration")
    apply_p.add_argument("name", help="Recipe name")
