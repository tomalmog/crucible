import { useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { TabBar } from "../../components/shared/TabBar";
import { TrainingMethodPicker } from "./TrainingMethodPicker";
import { TrainingWizard } from "./TrainingWizard";
import { SweepConfigForm } from "./SweepConfigForm";
import { TrainingMethod } from "../../types/training";
import { useCrucible } from "../../context/CrucibleContext";

type View = "pick" | "wizard" | "sweep";
type Tab = "new-run" | "sweep";

const TRAINING_TABS = ["new-run", "sweep"] as const;
const TAB_LABELS: Record<Tab, string> = {
  "new-run": "New Run",
  "sweep": "Sweep",
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
  }

  // Wizard takes over the full page — no header or tabs
  if (view === "wizard" && selectedMethod) {
    return <TrainingWizard method={selectedMethod} dataRoot={dataRoot} onBack={onBack} />;
  }

  const activeTab: Tab = view === "sweep" ? "sweep" : "new-run";

  return (
    <>
      <PageHeader title="Training" />
      <TabBar
        tabs={TRAINING_TABS}
        active={activeTab}
        onChange={handleTabChange}
        format={(t) => TAB_LABELS[t]}
      />

      {view === "pick" && <TrainingMethodPicker onSelect={onPickMethod} />}
      {view === "sweep" && <SweepConfigForm />}
    </>
  );
}
