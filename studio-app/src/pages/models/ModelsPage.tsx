import { useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { TabBar } from "../../components/shared/TabBar";
import { DetailPage } from "../../components/shared/DetailPage";
import { ListRow } from "../../components/shared/ListRow";
import { EmptyState } from "../../components/shared/EmptyState";
import { formatSize } from "../../components/shared/RegistryRow";
import { useCrucible } from "../../context/CrucibleContext";
import { ModelOverview } from "./ModelOverview";
import { ModelMergeForm } from "./ModelMergeForm";
import type { ModelEntry } from "../../types/models";

type DetailTab = "overview" | "merge";
const DETAIL_TABS = ["overview", "merge"] as const;

export function ModelsPage() {
  const { models, setSelectedModel, refreshModels } = useCrucible();
  const [detailEntry, setDetailEntry] = useState<ModelEntry | null>(null);
  const [tab, setTab] = useState<DetailTab>("overview");
  const [isRefreshing, setIsRefreshing] = useState(false);

  function handleSelect(entry: ModelEntry) {
    setSelectedModel(entry);
    setDetailEntry(entry);
    setTab("overview");
  }

  function handleBack() {
    setDetailEntry(null);
  }

  async function handleRefresh(): Promise<void> {
    setIsRefreshing(true);
    await refreshModels();
    setIsRefreshing(false);
  }

  if (detailEntry) {
    return (
      <DetailPage title={detailEntry.modelName} onBack={handleBack}>
        <TabBar tabs={DETAIL_TABS} active={tab} onChange={setTab} />
        {tab === "overview" && <ModelOverview entry={detailEntry} />}
        {tab === "merge" && <ModelMergeForm />}
      </DetailPage>
    );
  }

  return (
    <>
      <PageHeader title="Models">
        <button
          className="btn"
          onClick={() => handleRefresh().catch(console.error)}
          disabled={isRefreshing}
        >
          {isRefreshing ? "Refreshing..." : "Refresh"}
        </button>
      </PageHeader>

      {models.length === 0 ? (
        <EmptyState title="No models" description="Train a model or download one from the Hub." />
      ) : (
        <div className="panel panel-flush">
          {models.map((m) => (
            <ListRow
              key={m.modelName + m.modelPath}
              name={m.modelName}
              meta={
                <>
                  {m.hasRemote && <span className="badge">Remote</span>}
                  <span>{formatSize(m.sizeBytes)}</span>
                </>
              }
              onClick={() => handleSelect(m)}
            />
          ))}
        </div>
      )}
    </>
  );
}
