import { useState } from "react";
import { useLocation } from "react-router";
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

interface PrefillState {
  prefill?: Record<string, unknown>;
}

function resolveInitialTab(prefill?: Record<string, unknown>): InterpTab {
  if (prefill?.page !== "interpretability") return "logit-lens";
  const tab = prefill.tab as string;
  if (TABS.includes(tab as InterpTab)) return tab as InterpTab;
  return "logit-lens";
}

export function InterpPage() {
  const location = useLocation();
  const prefill = (location.state as PrefillState | null)?.prefill;
  const validPrefill = prefill?.page === "interpretability" ? prefill : undefined;
  const [tab, setTab] = useState<InterpTab>(resolveInitialTab(validPrefill));

  // Clear navigation state after reading to prevent re-apply on refresh
  if (prefill) {
    window.history.replaceState({}, "");
  }

  return (
    <>
      <PageHeader title="Interpretability" />
      <TabBar tabs={TABS} active={tab} onChange={setTab} format={(t) => TAB_LABELS[t]} />
      <div className="page-body">
        {tab === "logit-lens" && <LogitLensForm prefill={validPrefill?.tab === "logit-lens" ? validPrefill : undefined} />}
        {tab === "activation-pca" && <ActivationPcaForm prefill={validPrefill?.tab === "activation-pca" ? validPrefill : undefined} />}
        {tab === "activation-patching" && <ActivationPatchingForm prefill={validPrefill?.tab === "activation-patching" ? validPrefill : undefined} />}
      </div>
    </>
  );
}
