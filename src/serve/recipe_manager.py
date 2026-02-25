"""Training recipe management.

This module handles exporting, importing, listing, and applying
training recipes that encapsulate reproducible training configurations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingRecipe:
    """A training recipe that captures a complete training configuration.

    Attributes:
        name: Recipe name.
        description: What this recipe is for.
        method: Training method (sft, dpo-train, etc.).
        hyperparameters: Training hyperparameters.
        dataset_requirements: Description of required data format.
        eval_criteria: Evaluation criteria for the recipe.
    """

    name: str
    description: str
    method: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    dataset_requirements: str = ""
    eval_criteria: list[str] = field(default_factory=list)


BUILTIN_RECIPES: dict[str, TrainingRecipe] = {
    "coding_assistant": TrainingRecipe(
        name="Coding Assistant",
        description="Fine-tune a model for code generation and debugging",
        method="sft",
        hyperparameters={
            "epochs": 3, "learning_rate": 2e-5, "batch_size": 4,
            "max_token_length": 2048, "precision_mode": "bf16",
        },
        dataset_requirements="JSONL with prompt/response pairs containing code tasks",
        eval_criteria=["humaneval", "code_quality"],
    ),
    "customer_support": TrainingRecipe(
        name="Customer Support",
        description="Train a model for customer service interactions",
        method="sft",
        hyperparameters={
            "epochs": 5, "learning_rate": 1e-5, "batch_size": 8,
            "max_token_length": 1024, "precision_mode": "bf16",
        },
        dataset_requirements="JSONL with customer query/response pairs",
        eval_criteria=["helpfulness", "accuracy", "safety"],
    ),
    "reasoning_model": TrainingRecipe(
        name="Reasoning Model",
        description="Train a model for step-by-step reasoning and problem solving",
        method="grpo-train",
        hyperparameters={
            "epochs": 3, "learning_rate": 1e-5, "batch_size": 2,
            "max_token_length": 4096, "group_size": 8,
            "kl_coeff": 0.05, "precision_mode": "bf16",
        },
        dataset_requirements="JSONL with prompts requiring multi-step reasoning",
        eval_criteria=["gsm8k", "reasoning"],
    ),
    "domain_expert": TrainingRecipe(
        name="Domain Expert",
        description="Adapt a model to a specific knowledge domain",
        method="domain-adapt",
        hyperparameters={
            "epochs": 2, "learning_rate": 5e-6, "batch_size": 4,
            "max_token_length": 2048, "precision_mode": "bf16",
        },
        dataset_requirements="Domain-specific text corpus for continued pretraining",
        eval_criteria=["mmlu", "domain_accuracy"],
    ),
}


class RecipeManager:
    """Manage training recipes: export, import, list, and apply."""

    def __init__(self, data_root: Path) -> None:
        self._data_root = data_root
        self._recipes_dir = data_root / "recipes"

    def export_recipe(
        self,
        run_id: str,
        output_path: str,
    ) -> str:
        """Export a training run configuration as a recipe."""
        from serve.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(self._data_root)
        summary = tracker.get_run_summary(run_id)
        recipe = {
            "name": f"recipe_from_{run_id}",
            "description": f"Recipe exported from run {run_id}",
            "method": summary.get("method", "sft"),
            "hyperparameters": summary.get("hyperparameters", {}),
            "source_run_id": run_id,
        }
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(recipe, fh, indent=2, default=str)
        return str(path)

    def import_recipe(self, recipe_path: str) -> str:
        """Import a recipe file into the local recipes directory."""
        src = Path(recipe_path)
        if not src.exists():
            raise FileNotFoundError(f"Recipe file not found: {recipe_path}")
        with open(src, encoding="utf-8") as fh:
            recipe = json.load(fh)
        name = recipe.get("name", src.stem)
        self._recipes_dir.mkdir(parents=True, exist_ok=True)
        dest = self._recipes_dir / f"{name}.json"
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(recipe, fh, indent=2)
        return str(dest)

    def list_recipes(self) -> list[dict[str, Any]]:
        """List all available recipes (built-in + imported)."""
        recipes: list[dict[str, Any]] = []
        for key, recipe in BUILTIN_RECIPES.items():
            recipes.append({
                "name": recipe.name,
                "description": recipe.description,
                "method": recipe.method,
                "source": "builtin",
            })
        if self._recipes_dir.exists():
            for f in sorted(self._recipes_dir.glob("*.json")):
                with open(f, encoding="utf-8") as fh:
                    data = json.load(fh)
                recipes.append({
                    "name": data.get("name", f.stem),
                    "description": data.get("description", ""),
                    "method": data.get("method", ""),
                    "source": "imported",
                })
        return recipes

    def get_recipe(self, name: str) -> dict[str, Any]:
        """Get a recipe by name (checks built-in first, then imported)."""
        key = name.lower().replace(" ", "_")
        if key in BUILTIN_RECIPES:
            r = BUILTIN_RECIPES[key]
            return {
                "name": r.name, "description": r.description,
                "method": r.method, "hyperparameters": r.hyperparameters,
                "dataset_requirements": r.dataset_requirements,
                "eval_criteria": r.eval_criteria,
            }
        if self._recipes_dir.exists():
            path = self._recipes_dir / f"{key}.json"
            if path.exists():
                with open(path, encoding="utf-8") as fh:
                    return json.load(fh)
        return {}
