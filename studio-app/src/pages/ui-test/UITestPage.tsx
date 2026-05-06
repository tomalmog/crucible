import { useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { InterpExampleGallery } from "../interp/InterpExampleGallery";
import { InterpHero } from "../interp/InterpHero";
import type { InterpTab } from "../interp/interpTabs";
import { useModelAnatomyData } from "../interp/useModelAnatomyData";

export function UITestPage(): React.ReactNode {
  const [activeVisualization, setActiveVisualization] = useState<InterpTab>("logit-lens");
  const anatomy = useModelAnatomyData();

  return (
    <div className="form-lab-page">
      <PageHeader title="UI Library">
        <span className="form-lab-header-note">Mechanistic interp reference</span>
      </PageHeader>

      <InterpHero anatomy={anatomy} onSelect={setActiveVisualization} />
      <InterpExampleGallery
        activeTab={activeVisualization}
        onChange={setActiveVisualization}
      />
    </div>
  );
}
