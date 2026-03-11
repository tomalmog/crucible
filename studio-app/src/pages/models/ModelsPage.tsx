import { useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { useCrucible } from "../../context/CrucibleContext";
import { EmptyState } from "../../components/shared/EmptyState";
import { ModelListPanel } from "./ModelListPanel";
import { ModelOverview } from "./ModelOverview";
import { ModelMergeForm } from "./ModelMergeForm";

type Tab = "overview" | "merge";

export function ModelsPage() {
  const { selectedModel, refreshModels } = useCrucible();
  const [tab, setTab] = useState<Tab>("overview");
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);

  async function handleRefresh(): Promise<void> {
    setIsRefreshing(true);
    setRefreshKey((k) => k + 1);
    await refreshModels();
  }

  return (
    <>
      <PageHeader title="Model Registry">
        <button className="btn" onClick={() => handleRefresh().catch(console.error)} disabled={isRefreshing}>
          {isRefreshing ? "Refreshing..." : "Refresh"}
        </button>
      </PageHeader>

      <div className="two-column">
        <ModelListPanel refreshKey={refreshKey} onRefreshingChange={setIsRefreshing} />
        <div>
          <div className="tab-list">
            {(["overview", "merge"] as Tab[]).map((t) => (
              <button key={t} className={`tab-item ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>
                {t.charAt(0).toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>

          {tab === "overview" && (
            selectedModel
              ? <ModelOverview entry={selectedModel} />
              : <EmptyState title="No model selected" description="Select a model from the list." />
          )}
          {tab === "merge" && <ModelMergeForm />}
        </div>
      </div>
    </>
  );
}
