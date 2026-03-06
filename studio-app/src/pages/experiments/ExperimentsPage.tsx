import { useState, useEffect } from "react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { useForge } from "../../context/ForgeContext";
import { PageHeader } from "../../components/shared/PageHeader";
import { RunTable } from "./RunTable";
import { ExperimentDetail } from "./ExperimentDetail";
import { ExperimentCompare } from "./ExperimentCompare";
import { EvalResultsView } from "./EvalResultsView";
import { LlmJudgeForm } from "./LlmJudgeForm";
import { CostSummary } from "./CostSummary";

type View = "list" | "detail" | "compare" | "eval" | "judge" | "cost";

export function ExperimentsPage() {
  const { dataRoot } = useForge();
  const command = useForgeCommand();
  const [view, setView] = useState<View>("list");
  const [runs, setRuns] = useState<{ run_id: string; loss: string }[]>([]);
  const [selectedRun, setSelectedRun] = useState<string>("");
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  async function loadRuns() {
    if (!dataRoot) return;
    setLoading(true);
    try {
      const status = await command.run(dataRoot, ["experiment", "list"]);
      if (status.status === "completed" && command.output) {
        const lines = command.output.split("\n").filter((l) => l.startsWith("  "));
        const parsed = lines.map((l) => {
          const parts = l.trim().split(/\s+/);
          return { run_id: parts[0], loss: parts[1]?.replace("loss=", "") ?? "-" };
        });
        setRuns(parsed);
      }
    } catch {
      // Command invoke failed (e.g. Tauri not available) — leave runs empty
    } finally {
      setLoading(false);
    }
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
    <>
      <PageHeader title="Experiments">
        {view !== "list" && (
          <button className="btn btn-ghost" onClick={() => setView("list")}>
            Back
          </button>
        )}
        <button className={`btn ${view === "list" ? "btn-primary" : ""}`} onClick={() => { setView("list"); loadRuns().catch(console.error); }} disabled={loading}>
          {loading ? "Loading..." : "Runs"}
        </button>
        <button className={`btn ${view === "eval" ? "btn-primary" : ""}`} onClick={() => setView("eval")}>
          Evaluate
        </button>
        <button className={`btn ${view === "judge" ? "btn-primary" : ""}`} onClick={() => setView("judge")}>
          LLM Judge
        </button>
        <button className={`btn ${view === "cost" ? "btn-primary" : ""}`} onClick={() => setView("cost")}>
          Cost
        </button>
      </PageHeader>

      {view === "list" && (
        <RunTable runs={runs} onSelect={showDetail} onCompare={startCompare} />
      )}
      {view === "detail" && (
        <ExperimentDetail runId={selectedRun} dataRoot={dataRoot ?? ""} />
      )}
      {view === "compare" && (
        <ExperimentCompare runIds={compareIds} dataRoot={dataRoot ?? ""} />
      )}
      {view === "eval" && <EvalResultsView />}
      {view === "judge" && <LlmJudgeForm />}
      {view === "cost" && <CostSummary />}
    </>
  );
}
