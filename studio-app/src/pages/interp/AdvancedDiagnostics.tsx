import { useState } from "react";
import { TabBar } from "../../components/shared/TabBar";
import { ActivationPatchingForm } from "./ActivationPatchingForm";
import { ActivationPcaForm } from "./ActivationPcaForm";
import { LinearProbeForm } from "./LinearProbeForm";
import { LogitLensForm } from "./LogitLensForm";
import { SaeForm } from "./SaeForm";
import { SteeringForm } from "./SteeringForm";

type DiagnosticTab = "logit-lens" | "activation-pca" | "activation-patching" | "linear-probe" | "sae" | "steering";

const DIAGNOSTIC_TABS: readonly DiagnosticTab[] = [
  "logit-lens",
  "activation-pca",
  "activation-patching",
  "linear-probe",
  "sae",
  "steering",
];

const DIAGNOSTIC_LABELS: Record<DiagnosticTab, string> = {
  "logit-lens": "Logit Lens",
  "activation-pca": "Activation PCA",
  "activation-patching": "Activation Patching",
  "linear-probe": "Linear Probe",
  sae: "SAE",
  steering: "Steering",
};

interface AdvancedDiagnosticsProps {
  prefill?: Record<string, unknown>;
}

export function AdvancedDiagnostics({ prefill }: AdvancedDiagnosticsProps): React.ReactNode {
  const [tab, setTab] = useState<DiagnosticTab>(resolveDiagnosticTab(prefill));

  return (
    <div className="stack-lg">
      <div className="info-banner">
        Use individual diagnostics when a health check points to a specific failure mode.
      </div>
      <TabBar tabs={DIAGNOSTIC_TABS} active={tab} onChange={setTab} format={(item) => DIAGNOSTIC_LABELS[item]} />
      {tab === "logit-lens" && <LogitLensForm prefill={prefill?.tab === "logit-lens" ? prefill : undefined} />}
      {tab === "activation-pca" && <ActivationPcaForm prefill={prefill?.tab === "activation-pca" ? prefill : undefined} />}
      {tab === "activation-patching" && <ActivationPatchingForm prefill={prefill?.tab === "activation-patching" ? prefill : undefined} />}
      {tab === "linear-probe" && <LinearProbeForm prefill={prefill?.tab === "linear-probe" ? prefill : undefined} />}
      {tab === "sae" && <SaeForm prefill={prefill && (prefill.tab === "sae-train" || prefill.tab === "sae-analyze") ? prefill : undefined} />}
      {tab === "steering" && <SteeringForm prefill={prefill && (prefill.tab === "steer-compute" || prefill.tab === "steer-apply") ? prefill : undefined} />}
    </div>
  );
}

function resolveDiagnosticTab(prefill?: Record<string, unknown>): DiagnosticTab {
  const tab = typeof prefill?.tab === "string" ? prefill.tab : "";
  if (tab === "activation-patch") return "activation-patching";
  if (tab === "sae-train" || tab === "sae-analyze") return "sae";
  if (tab === "steer-compute" || tab === "steer-apply") return "steering";
  if (tab === "logit-lens") return "logit-lens";
  if (tab === "activation-pca") return "activation-pca";
  if (tab === "activation-patching") return "activation-patching";
  if (tab === "linear-probe") return "linear-probe";
  if (tab === "sae") return "sae";
  if (tab === "steering") return "steering";
  return "logit-lens";
}
