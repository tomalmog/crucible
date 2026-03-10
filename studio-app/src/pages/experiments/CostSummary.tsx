import { useState, useEffect } from "react";
import { useCrucibleCommand } from "../../hooks/useCrucibleCommand";
import { useCrucible } from "../../context/CrucibleContext";

export function CostSummary() {
  const { dataRoot } = useCrucible();
  const command = useCrucibleCommand();
  const [summary, setSummary] = useState("");
  const [loading, setLoading] = useState(false);

  async function loadSummary() {
    if (!dataRoot) return;
    setLoading(true);
    const status = await command.run(dataRoot, ["cost", "summary"]);
    if (status.status === "completed" && command.output) {
      setSummary(command.output);
    }
    setLoading(false);
  }

  // Fetch cost summary on mount and when dataRoot changes
  useEffect(() => {
    loadSummary().catch(console.error);
  }, [dataRoot]);

  return (
    <div className="panel stack">
      <div className="row">
        <h3>Cost Summary</h3>
        <div className="spacer" />
        <button className="btn btn-sm" onClick={() => loadSummary().catch(console.error)} disabled={loading}>
          Refresh
        </button>
      </div>
      {summary ? (
        <pre className="console">{summary}</pre>
      ) : (
        <p className="text-muted">No cost data available. Train a model to start tracking costs.</p>
      )}
    </div>
  );
}
