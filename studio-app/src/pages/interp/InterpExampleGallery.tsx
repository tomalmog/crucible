import { type ReactNode, useState } from "react";
import { TabBar } from "../../components/shared/TabBar";
import { ActivationPatchingResults } from "./ActivationPatchingResults";
import { ActivationPcaResults } from "./ActivationPcaResults";
import { LinearProbeResults } from "./LinearProbeResults";
import { LogitLensResults } from "./LogitLensResults";
import { SaeAnalyzeResults, SaeTrainResults } from "./SaeResults";
import { SteerApplyResults, SteerComputeResults } from "./SteerResults";
import {
  SAMPLE_LOGIT_LENS,
  SAMPLE_PATCHING,
  SAMPLE_PCA,
  SAMPLE_PROBE,
  SAMPLE_SAE_ANALYZE,
  SAMPLE_SAE_TRAIN,
  SAMPLE_STEER_APPLY,
  SAMPLE_STEER_COMPUTE,
} from "./interpExampleData";
import { INTERP_TAB_LABELS, INTERP_TABS, type InterpTab } from "./interpTabs";

interface InterpExampleGalleryProps {
  activeTab?: InterpTab;
  onChange?: (tab: InterpTab) => void;
}

export function InterpExampleGallery({
  activeTab,
  onChange,
}: InterpExampleGalleryProps): ReactNode {
  const [localActiveTab, setLocalActiveTab] = useState<InterpTab>("logit-lens");
  const selectedTab = activeTab ?? localActiveTab;

  function handleChange(nextTab: InterpTab): void {
    if (onChange) {
      onChange(nextTab);
      return;
    }
    setLocalActiveTab(nextTab);
  }

  return (
    <section className="interp-example-gallery">
      <TabBar
        tabs={INTERP_TABS}
        active={selectedTab}
        onChange={handleChange}
        format={(tab) => INTERP_TAB_LABELS[tab]}
      />
      <div className="interp-example-stage">
        {selectedTab === "logit-lens" && <LogitLensResults result={SAMPLE_LOGIT_LENS} />}
        {selectedTab === "activation-pca" && <ActivationPcaResults result={SAMPLE_PCA} />}
        {selectedTab === "activation-patching" && <ActivationPatchingResults result={SAMPLE_PATCHING} />}
        {selectedTab === "linear-probe" && <LinearProbeResults result={SAMPLE_PROBE} />}
        {selectedTab === "sae" && (
          <div className="interp-example-duo">
            <SaeAnalyzeResults result={SAMPLE_SAE_ANALYZE} />
            <SaeTrainResults result={SAMPLE_SAE_TRAIN} />
          </div>
        )}
        {selectedTab === "steering" && (
          <div className="interp-example-duo">
            <SteerComputeResults result={SAMPLE_STEER_COMPUTE} />
            <SteerApplyResults result={SAMPLE_STEER_APPLY} />
          </div>
        )}
      </div>
    </section>
  );
}
