import { useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { useForge } from "../../context/ForgeContext";
import { EmptyState } from "../../components/shared/EmptyState";
import { ModelListPanel } from "./ModelListPanel";
import { ModelOverview } from "./ModelOverview";
import { ModelDiffView } from "./ModelDiffView";
import { ModelActions } from "./ModelActions";
import { ModelMergeForm } from "./ModelMergeForm";

type Tab = "overview" | "diff" | "actions" | "merge";

export function ModelsPage() {
  const { dataRoot, modelVersions, selectedModel, refreshModels } = useForge();
  const [tab, setTab] = useState<Tab>("overview");
  const [isRefreshing, setIsRefreshing] = useState(false);

  async function handleRefresh() {
    setIsRefreshing(true);
    try {
      await refreshModels();
    } finally {
      setIsRefreshing(false);
    }
  }

  return (
    <>
      <PageHeader title="Model Registry">
        <button className="btn" onClick={() => handleRefresh().catch(console.error)} disabled={isRefreshing}>
          {isRefreshing ? "Loading..." : "Refresh"}
        </button>
      </PageHeader>

      <div className="two-column">
        <ModelListPanel />
        <div>
          <div className="tab-list">
            {(["overview", "diff", "actions", "merge"] as Tab[]).map((t) => (
              <button key={t} className={`tab-item ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>
                {t.charAt(0).toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>

          {tab === "overview" && (
            selectedModel
              ? <ModelOverview version={selectedModel} />
              : <EmptyState title="No model selected" description="Select a model version from the list." />
          )}
          {tab === "diff" && <ModelDiffView dataRoot={dataRoot} versions={modelVersions} />}
          {tab === "actions" && <ModelActions dataRoot={dataRoot} versions={modelVersions} />}
          {tab === "merge" && <ModelMergeForm />}
        </div>
      </div>
    </>
  );
}
