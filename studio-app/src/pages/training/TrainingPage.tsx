import { useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { TabBar } from "../../components/shared/TabBar";
import { TrainingMethodPicker } from "./TrainingMethodPicker";
import { TrainingWizard } from "./TrainingWizard";
import { TrainingRunHistory } from "./TrainingRunHistory";
import { SweepConfigForm } from "./SweepConfigForm";
import { TrainingMethod } from "../../types/training";
import { useCrucible } from "../../context/CrucibleContext";

type View = "pick" | "wizard" | "history" | "sweep";
type Tab = "new-run" | "sweep" | "history";

const TRAINING_TABS = ["new-run", "sweep", "history"] as const;
const TAB_LABELS: Record<Tab, string> = {
  "new-run": "New Run",
  "sweep": "Sweep",
  "history": "History",
};

const VIEW_TO_TAB: Record<View, Tab> = {
  pick: "new-run",
  wizard: "new-run",
  sweep: "sweep",
  history: "history",
};

export function TrainingPage() {
  const [view, setView] = useState<View>("pick");
  const [selectedMethod, setSelectedMethod] = useState<TrainingMethod | null>(null);
  const { dataRoot } = useCrucible();

  function onPickMethod(method: TrainingMethod) {
    setSelectedMethod(method);
    setView("wizard");
  }

  function onBack() {
    setView("pick");
    setSelectedMethod(null);
  }

  function handleTabChange(t: Tab) {
    if (t === "new-run") { setView("pick"); setSelectedMethod(null); }
    else if (t === "sweep") setView("sweep");
    else setView("history");
  }

  return (
    <>
      <PageHeader title="Training" />
      <TabBar
        tabs={TRAINING_TABS}
        active={VIEW_TO_TAB[view]}
        onChange={handleTabChange}
        format={(t) => TAB_LABELS[t]}
      />

      {view === "pick" && <TrainingMethodPicker onSelect={onPickMethod} />}
      {view === "wizard" && selectedMethod && (
        <TrainingWizard method={selectedMethod} dataRoot={dataRoot} onBack={onBack} />
      )}
      {view === "sweep" && <SweepConfigForm />}
      {view === "history" && <TrainingRunHistory dataRoot={dataRoot} />}
    </>
  );
}
