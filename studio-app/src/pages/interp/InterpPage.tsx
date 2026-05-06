import { type ReactNode, useState } from "react";
import { useLocation } from "react-router";
import { PageHeader } from "../../components/shared/PageHeader";
import { TabBar } from "../../components/shared/TabBar";
import { LogitLensForm } from "./LogitLensForm";
import { ActivationPcaForm } from "./ActivationPcaForm";
import { ActivationPatchingForm } from "./ActivationPatchingForm";
import { LinearProbeForm } from "./LinearProbeForm";
import { SaeForm } from "./SaeForm";
import { SteeringForm } from "./SteeringForm";
import {
  INTERP_TAB_LABELS,
  INTERP_TABS,
  isInterpTab,
  type InterpTab,
} from "./interpTabs";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function readPrefill(state: unknown): Record<string, unknown> | undefined {
  if (!isRecord(state) || !isRecord(state.prefill)) return undefined;
  return state.prefill;
}

function resolveInitialTab(prefill?: Record<string, unknown>): InterpTab {
  if (prefill?.page !== "interpretability") return "logit-lens";
  const tab = prefill.tab;
  // Map sub-tabs (sae-train, sae-analyze, steer-compute, steer-apply) to parent tab
  if (tab === "sae-train" || tab === "sae-analyze") return "sae";
  if (tab === "steer-compute" || tab === "steer-apply") return "steering";
  if (isInterpTab(tab)) return tab;
  return "logit-lens";
}

export function InterpPage(): ReactNode {
  const location = useLocation();
  const prefill = readPrefill(location.state);
  const validPrefill = prefill?.page === "interpretability" ? prefill : undefined;
  const [selectedForm, setSelectedForm] = useState<InterpTab>(
    resolveInitialTab(validPrefill),
  );

  // Clear navigation state after reading to prevent re-apply on refresh
  if (prefill) {
    window.history.replaceState({}, "");
  }

  return (
    <>
      <PageHeader title="Interpretability" />
      <div className="interp-observatory">
        <section
          id="interp-form-suite"
          className="interp-form-suite"
          aria-label="Interpretability forms"
        >
          <TabBar
            tabs={INTERP_TABS}
            active={selectedForm}
            onChange={setSelectedForm}
            format={(nextForm) => INTERP_TAB_LABELS[nextForm]}
          />
          <div className="interp-form-active">
            {selectedForm === "logit-lens" && (
              <LogitLensForm
                prefill={validPrefill?.tab === "logit-lens" ? validPrefill : undefined}
              />
            )}
            {selectedForm === "activation-pca" && (
              <ActivationPcaForm
                prefill={validPrefill?.tab === "activation-pca" ? validPrefill : undefined}
              />
            )}
            {selectedForm === "activation-patching" && (
              <ActivationPatchingForm
                prefill={
                  validPrefill?.tab === "activation-patching" ? validPrefill : undefined
                }
              />
            )}
            {selectedForm === "linear-probe" && (
              <LinearProbeForm
                prefill={validPrefill?.tab === "linear-probe" ? validPrefill : undefined}
              />
            )}
            {selectedForm === "sae" && (
              <SaeForm
                prefill={
                  validPrefill &&
                  (validPrefill.tab === "sae-train" || validPrefill.tab === "sae-analyze")
                    ? validPrefill
                    : undefined
                }
              />
            )}
            {selectedForm === "steering" && (
              <SteeringForm
                prefill={
                  validPrefill &&
                  (validPrefill.tab === "steer-compute" ||
                    validPrefill.tab === "steer-apply")
                    ? validPrefill
                    : undefined
                }
              />
            )}
          </div>
        </section>
      </div>
    </>
  );
}
