import { useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { DatasetListPanel } from "./DatasetListPanel";
import { DatasetDashboard } from "./DatasetDashboard";
import { SampleInspector } from "./SampleInspector";
import { IngestForm } from "./IngestForm";
import { FilterForm } from "./FilterForm";
import { useForge } from "../../context/ForgeContext";

type Tab = "overview" | "samples" | "ingest" | "filter";

export function DatasetsPage() {
  const [tab, setTab] = useState<Tab>("overview");
  const { selectedDataset, setSelectedDataset, refreshDatasets } = useForge();

  function handleSelectDataset(ds: string) {
    setSelectedDataset(ds);
    setTab("overview");
  }

  return (
    <>
      <PageHeader title="Datasets">
        <button className="btn" onClick={() => refreshDatasets().catch(console.error)}>
          Refresh
        </button>
      </PageHeader>

      <div className="two-column">
        <DatasetListPanel onSelect={handleSelectDataset} />
        <div>
          <div className="tab-list">
            {(["overview", "samples", "ingest", "filter"] as Tab[]).map((t) => (
              <button
                key={t}
                className={`tab-item ${tab === t ? "active" : ""}`}
                onClick={() => setTab(t)}
              >
                {t.charAt(0).toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>

          {tab === "overview" && <DatasetDashboard />}
          {tab === "samples" && <SampleInspector />}
          {tab === "ingest" && <IngestForm />}
          {tab === "filter" && selectedDataset && <FilterForm />}
          {tab === "filter" && !selectedDataset && (
            <p className="text-tertiary">Select a dataset first.</p>
          )}
        </div>
      </div>
    </>
  );
}
