import { useState, useEffect } from "react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { useForge } from "../../context/ForgeContext";

interface Recipe {
  name: string;
  method: string;
  source: string;
}

export function RecipeLibrary() {
  const { dataRoot } = useForge();
  const command = useForgeCommand();
  const [recipes, setRecipes] = useState<Recipe[]>([]);
  const [recipeDetail, setRecipeDetail] = useState("");

  async function loadRecipes() {
    if (!dataRoot) return;
    const status = await command.run(dataRoot, ["recipe", "list"]);
    if (status.status === "completed" && command.output) {
      const lines = command.output.split("\n").filter((l) => l.trim());
      const parsed = lines.map((l) => {
        const parts = l.trim().split(/\s+/);
        return {
          name: parts[0] ?? "",
          method: parts[1]?.replace("method=", "") ?? "",
          source: parts[2]?.replace("source=", "") ?? "",
        };
      });
      setRecipes(parsed);
    }
  }

  async function applyRecipe(name: string) {
    if (!dataRoot) return;
    const status = await command.run(dataRoot, ["recipe", "apply", name]);
    if (status.status === "completed" && command.output) {
      setRecipeDetail(command.output);
    }
  }

  useEffect(() => {
    loadRecipes().catch(console.error);
  }, [dataRoot]);

  return (
    <div className="panel stack-md">
      <h3>Training Recipes</h3>
      {recipes.length === 0 ? (
        <p className="text-muted">Loading recipes...</p>
      ) : (
        <table className="data-table">
          <thead>
            <tr>
              <th>Recipe</th>
              <th>Method</th>
              <th>Source</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {recipes.map((r) => (
              <tr key={r.name}>
                <td>{r.name}</td>
                <td>{r.method}</td>
                <td>{r.source}</td>
                <td>
                  <button className="btn btn-sm" onClick={() => applyRecipe(r.name).catch(console.error)}>
                    Apply
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {recipeDetail && <pre className="console">{recipeDetail}</pre>}
    </div>
  );
}
