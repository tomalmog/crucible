import { useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { TrainingMethodPicker } from "./TrainingMethodPicker";
import { TrainingWizard } from "./TrainingWizard";
import { TrainingRunHistory } from "./TrainingRunHistory";
import { TrainingMethod } from "../../types/training";
import { useForge } from "../../context/ForgeContext";

type View = "pick" | "wizard" | "history";

export function TrainingPage() {
  const [view, setView] = useState<View>("pick");
  const [selectedMethod, setSelectedMethod] = useState<TrainingMethod | null>(null);
  const { dataRoot } = useForge();

  function onPickMethod(method: TrainingMethod) {
    setSelectedMethod(method);
    setView("wizard");
  }

  function onBack() {
    setView("pick");
    setSelectedMethod(null);
  }

  return (
    <>
      <PageHeader title="Training">
        <button
          className={`btn ${view === "pick" || view === "wizard" ? "btn-primary" : ""}`}
          onClick={() => setView("pick")}
        >
          New Run
        </button>
        <button
          className={`btn ${view === "history" ? "btn-primary" : ""}`}
          onClick={() => setView("history")}
        >
          History
        </button>
      </PageHeader>

      {view === "pick" && <TrainingMethodPicker onSelect={onPickMethod} />}
      {view === "wizard" && selectedMethod && (
        <TrainingWizard method={selectedMethod} dataRoot={dataRoot} onBack={onBack} />
      )}
      {view === "history" && <TrainingRunHistory dataRoot={dataRoot} />}
    </>
  );
}
