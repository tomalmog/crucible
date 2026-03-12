import { useCallback, useEffect, useState } from "react";
import { Download, Loader2, Plus, Trash2, Upload } from "lucide-react";
import { PageHeader } from "../../components/shared/PageHeader";
import { TabBar } from "../../components/shared/TabBar";
import { DetailPage } from "../../components/shared/DetailPage";
import { ListRow } from "../../components/shared/ListRow";
import { EmptyState } from "../../components/shared/EmptyState";
import { IngestModal } from "../../components/shared/IngestModal";
import { ConfirmDeleteModal } from "../../components/shared/ConfirmDeleteModal";
import { formatSize } from "../../components/shared/RegistryRow";
import { DatasetDashboard } from "./DatasetDashboard";
import { SampleInspector } from "./SampleInspector";
import { FilterForm } from "./FilterForm";
import { useCrucible } from "../../context/CrucibleContext";
import { deleteDataset } from "../../api/studioApi";
import { listClusters, listRemoteDatasets, deleteRemoteDataset, pushDatasetToCluster, pullDatasetFromCluster } from "../../api/remoteApi";
import type { ClusterConfig, RemoteDatasetInfo } from "../../types/remote";

type DetailTab = "overview" | "samples" | "filter";
type LocationTab = "local" | "remote";
const DETAIL_TABS = ["overview", "samples", "filter"] as const;
const LOCATION_TABS = ["local", "remote"] as const;

export function DatasetsPage() {
  const { dataRoot, datasets, setSelectedDataset, refreshDatasets } = useCrucible();
  const [detailName, setDetailName] = useState<string | null>(null);
  const [tab, setTab] = useState<DetailTab>("overview");
  const [locationTab, setLocationTab] = useState<LocationTab>("local");
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [showIngest, setShowIngest] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [transferring, setTransferring] = useState<Set<string>>(new Set());

  const [clusters, setClusters] = useState<ClusterConfig[]>([]);
  const [selectedCluster, setSelectedCluster] = useState("");
  const [remoteDatasets, setRemoteDatasets] = useState<RemoteDatasetInfo[]>([]);
  const [loadingRemote, setLoadingRemote] = useState(false);

  useEffect(() => {
    listClusters(dataRoot)
      .then((c) => {
        setClusters(c);
        if (c.length > 0) setSelectedCluster(c[0].name);
      })
      .catch(console.error);
  }, [dataRoot]);

  const fetchRemote = useCallback(async () => {
    if (!selectedCluster) return;
    setLoadingRemote(true);
    try {
      setRemoteDatasets(await listRemoteDatasets(dataRoot, selectedCluster));
    } catch {
      setRemoteDatasets([]);
    } finally {
      setLoadingRemote(false);
    }
  }, [dataRoot, selectedCluster]);

  useEffect(() => {
    if (locationTab === "remote" && selectedCluster) {
      fetchRemote().catch(console.error);
    }
  }, [locationTab, selectedCluster, fetchRemote]);

  function handleSelect(name: string) {
    setSelectedDataset(name);
    setDetailName(name);
    setTab("overview");
  }

  function handleBack() {
    setDetailName(null);
  }

  async function handleDeleteDataset(): Promise<void> {
    if (!pendingDelete) return;
    setDeleting(true);
    try {
      if (isLocal) {
        await deleteDataset(dataRoot, pendingDelete);
        await refreshDatasets();
      } else {
        await deleteRemoteDataset(dataRoot, selectedCluster, pendingDelete);
        await fetchRemote();
      }
      if (detailName === pendingDelete) setDetailName(null);
    } finally {
      setPendingDelete(null);
      setDeleting(false);
    }
  }

  async function handleRefresh(): Promise<void> {
    setIsRefreshing(true);
    if (locationTab === "remote") {
      await fetchRemote();
    } else {
      await refreshDatasets();
    }
    setIsRefreshing(false);
  }

  async function handlePushDataset(name: string): Promise<void> {
    setTransferring((prev) => new Set(prev).add(name));
    try {
      await pushDatasetToCluster(dataRoot, selectedCluster, name);
      await Promise.all([refreshDatasets(), fetchRemote()]);
    } finally {
      setTransferring((prev) => {
        const next = new Set(prev);
        next.delete(name);
        return next;
      });
    }
  }

  async function handlePullDataset(name: string): Promise<void> {
    setTransferring((prev) => new Set(prev).add(name));
    try {
      await pullDatasetFromCluster(dataRoot, selectedCluster, name);
      await refreshDatasets();
    } finally {
      setTransferring((prev) => {
        const next = new Set(prev);
        next.delete(name);
        return next;
      });
    }
  }

  if (detailName) {
    return (
      <DetailPage title={detailName} onBack={handleBack}>
        <TabBar tabs={DETAIL_TABS} active={tab} onChange={setTab} />
        {tab === "overview" && <DatasetDashboard />}
        {tab === "samples" && <SampleInspector />}
        {tab === "filter" && <FilterForm />}
      </DetailPage>
    );
  }

  const isLocal = locationTab === "local";
  const rows = isLocal
    ? datasets.map((d) => ({ name: d.name, sizeBytes: d.sizeBytes }))
    : remoteDatasets.map((d) => ({ name: d.name, sizeBytes: d.sizeBytes }));

  const emptyMsg = isLocal
    ? "No local datasets. Ingest data from the Training page or CLI."
    : "No remote datasets on this cluster.";

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

      <TabBar
        tabs={LOCATION_TABS}
        active={locationTab}
        onChange={setLocationTab}
        format={(t) => t.charAt(0).toUpperCase() + t.slice(1)}
      />

      {!isLocal && clusters.length > 1 && (
        <select
          value={selectedCluster}
          onChange={(e) => setSelectedCluster(e.target.value)}
          style={{ marginBottom: 12, width: "100%" }}
        >
          {clusters.map((c) => (
            <option key={c.name} value={c.name}>{c.name}</option>
          ))}
        </select>
      )}

      {!isLocal && clusters.length === 0 ? (
        <EmptyState title="No clusters" description="Add a cluster in Settings to see remote datasets." />
      ) : loadingRemote && !isLocal ? (
        <EmptyState title="Loading..." description="Fetching remote datasets." />
      ) : rows.length === 0 ? (
        <EmptyState title="No datasets" description={emptyMsg} />
      ) : (
        <div className="panel panel-flush">
          {rows.map((d) => (
            <ListRow
              key={d.name}
              name={d.name}
              meta={<span>{formatSize(d.sizeBytes)}</span>}
              actions={
                <>
                  {isLocal ? (
                    <button
                      className="btn btn-ghost btn-sm btn-icon"
                      title="Push to remote cluster"
                      disabled={clusters.length === 0 || transferring.has(d.name)}
                      onClick={(e) => { e.stopPropagation(); handlePushDataset(d.name).catch(console.error); }}
                    >
                      {transferring.has(d.name) ? <Loader2 size={14} className="spin" /> : <Upload size={14} />}
                    </button>
                  ) : (
                    <button
                      className="btn btn-ghost btn-sm btn-icon"
                      title="Pull to local"
                      disabled={transferring.has(d.name)}
                      onClick={(e) => { e.stopPropagation(); handlePullDataset(d.name).catch(console.error); }}
                    >
                      {transferring.has(d.name) ? <Loader2 size={14} className="spin" /> : <Download size={14} />}
                    </button>
                  )}
                  <button
                    className="btn btn-ghost btn-sm btn-icon"
                    title="Delete dataset"
                    onClick={() => setPendingDelete(d.name)}
                  >
                    <Trash2 size={14} />
                  </button>
                </>
              }
              onClick={() => handleSelect(d.name)}
            />
          ))}
        </div>
      )}

      <button className="fab-add" onClick={() => setShowIngest(true)} title="Ingest dataset">
        <Plus size={22} />
      </button>

      {showIngest && (
        <IngestModal
          onComplete={() => setShowIngest(false)}
          onClose={() => setShowIngest(false)}
        />
      )}

      {pendingDelete && (
        <ConfirmDeleteModal
          title="Delete Dataset"
          itemName={pendingDelete}
          isDeleting={deleting}
          onConfirm={() => handleDeleteDataset().catch(console.error)}
          onCancel={() => setPendingDelete(null)}
        />
      )}
    </>
  );
}
