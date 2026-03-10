import { useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { SafetyEvalForm } from "./SafetyEvalForm";
import { SafetyGateForm } from "./SafetyGateForm";
import { SafetyResultsView } from "./SafetyResultsView";
import { useCrucible } from "../../context/CrucibleContext";

type Tab = "evaluate" | "gate";

export function SafetyPage() {
  const { dataRoot } = useCrucible();
  const [tab, setTab] = useState<Tab>("evaluate");
  const [lastOutput, setLastOutput] = useState("");

  return (
    <>
      <PageHeader title="Safety" />

      <div className="tab-list">
        <button className={`tab-item ${tab === "evaluate" ? "active" : ""}`} onClick={() => setTab("evaluate")}>
          Evaluate
        </button>
        <button className={`tab-item ${tab === "gate" ? "active" : ""}`} onClick={() => setTab("gate")}>
          Gate
        </button>
      </div>

      <div className="stack-lg">
        {tab === "evaluate" && <SafetyEvalForm dataRoot={dataRoot} onResult={setLastOutput} />}
        {tab === "gate" && <SafetyGateForm dataRoot={dataRoot} onResult={setLastOutput} />}
        {lastOutput && <SafetyResultsView output={lastOutput} />}
      </div>
    </>
  );
}
