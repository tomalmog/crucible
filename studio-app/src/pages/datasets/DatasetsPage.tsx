import { useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { TabBar } from "../../components/shared/TabBar";
import { DetailPage } from "../../components/shared/DetailPage";
import { ListRow } from "../../components/shared/ListRow";
import { EmptyState } from "../../components/shared/EmptyState";
import { formatSize } from "../../components/shared/RegistryRow";
import { DatasetDashboard } from "./DatasetDashboard";
import { SampleInspector } from "./SampleInspector";
import { IngestForm } from "./IngestForm";
import { FilterForm } from "./FilterForm";
import { useCrucible } from "../../context/CrucibleContext";

type DetailTab = "overview" | "samples" | "ingest" | "filter";
const DETAIL_TABS = ["overview", "samples", "ingest", "filter"] as const;

export function DatasetsPage() {
  const { datasets, setSelectedDataset, refreshDatasets } = useCrucible();
  const [detailName, setDetailName] = useState<string | null>(null);
  const [tab, setTab] = useState<DetailTab>("overview");
  const [isRefreshing, setIsRefreshing] = useState(false);

  function handleSelect(name: string) {
    setSelectedDataset(name);
    setDetailName(name);
    setTab("overview");
  }

  function handleBack() {
    setDetailName(null);
  }

  async function handleRefresh(): Promise<void> {
    setIsRefreshing(true);
    await refreshDatasets();
    setIsRefreshing(false);
  }

  if (detailName) {
    return (
      <DetailPage title={detailName} onBack={handleBack}>
        <TabBar tabs={DETAIL_TABS} active={tab} onChange={setTab} />
        {tab === "overview" && <DatasetDashboard />}
        {tab === "samples" && <SampleInspector />}
        {tab === "ingest" && <IngestForm />}
        {tab === "filter" && <FilterForm />}
      </DetailPage>
    );
  }

  return (
    <>
      <PageHeader title="Datasets">
        <button
          className="btn"
          onClick={() => handleRefresh().catch(console.error)}
          disabled={isRefreshing}
        >
          {isRefreshing ? "Refreshing..." : "Refresh"}
        </button>
      </PageHeader>

      {datasets.length === 0 ? (
        <EmptyState title="No datasets" description="Ingest data from the Training page or CLI." />
      ) : (
        <div className="panel panel-flush">
          {datasets.map((ds) => (
            <ListRow
              key={ds.name}
              name={ds.name}
              meta={<span>{formatSize(ds.sizeBytes)}</span>}
              onClick={() => handleSelect(ds.name)}
            />
          ))}
        </div>
      )}
    </>
  );
}
