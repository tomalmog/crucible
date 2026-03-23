import { useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { TabBar } from "../../components/shared/TabBar";
import { LogitLensForm } from "./LogitLensForm";
import { ActivationPcaForm } from "./ActivationPcaForm";
import { ActivationPatchingForm } from "./ActivationPatchingForm";

type InterpTab = "logit-lens" | "activation-pca" | "activation-patching";

const TABS: readonly InterpTab[] = ["logit-lens", "activation-pca", "activation-patching"];

const TAB_LABELS: Record<InterpTab, string> = {
  "logit-lens": "Logit Lens",
  "activation-pca": "Activation PCA",
  "activation-patching": "Activation Patching",
};

export function InterpPage() {
  const [tab, setTab] = useState<InterpTab>("logit-lens");

  return (
    <>
      <PageHeader title="Interpretability" />
      <TabBar tabs={TABS} active={tab} onChange={setTab} format={(t) => TAB_LABELS[t]} />
      <div className="page-body">
        {tab === "logit-lens" && <LogitLensForm />}
        {tab === "activation-pca" && <ActivationPcaForm />}
        {tab === "activation-patching" && <ActivationPatchingForm />}
      </div>
    </>
  );
}
