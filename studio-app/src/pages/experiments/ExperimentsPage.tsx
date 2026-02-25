import { useState, useEffect } from "react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { useForge } from "../../context/ForgeContext";
import { RunTable } from "./RunTable";
import { ExperimentDetail } from "./ExperimentDetail";
import { ExperimentCompare } from "./ExperimentCompare";

type View = "list" | "detail" | "compare";

export function ExperimentsPage() {
  const { dataRoot } = useForge();
  const command = useForgeCommand();
  const [view, setView] = useState<View>("list");
  const [runs, setRuns] = useState<any[]>([]);
  const [selectedRun, setSelectedRun] = useState<string>("");
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  async function loadRuns() {
    if (!dataRoot) return;
    setLoading(true);
    const status = await command.run(dataRoot, ["experiment", "list"]);
    if (status.status === "completed" && command.output) {
      const lines = command.output.split("\n").filter((l) => l.startsWith("  "));
      const parsed = lines.map((l) => {
        const parts = l.trim().split(/\s+/);
        return { run_id: parts[0], loss: parts[1]?.replace("loss=", "") ?? "-" };
      });
      setRuns(parsed);
    }
    setLoading(false);
  }

  useEffect(() => {
    loadRuns().catch(console.error);
  }, [dataRoot]);

  function showDetail(runId: string) {
    setSelectedRun(runId);
    setView("detail");
  }

  function startCompare(ids: string[]) {
    setCompareIds(ids);
    setView("compare");
  }

  return (
    <div>
      <div className="page-header">
        <h1>Experiments</h1>
        <div className="spacer" />
        {view !== "list" && (
          <button className="btn btn-ghost" onClick={() => setView("list")}>
            Back to List
          </button>
        )}
        <button className="btn" onClick={() => loadRuns().catch(console.error)} disabled={loading}>
          {loading ? "Loading..." : "Refresh"}
        </button>
      </div>

      {view === "list" && (
        <RunTable
          runs={runs}
          onSelect={showDetail}
          onCompare={startCompare}
        />
      )}

      {view === "detail" && (
        <ExperimentDetail runId={selectedRun} dataRoot={dataRoot ?? ""} />
      )}

      {view === "compare" && (
        <ExperimentCompare runIds={compareIds} dataRoot={dataRoot ?? ""} />
      )}
    </div>
  );
}
