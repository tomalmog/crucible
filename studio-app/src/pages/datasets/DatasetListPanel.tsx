import { useState, useEffect, useCallback, useRef } from "react";
import { Loader2 } from "lucide-react";
import { useCrucible } from "../../context/CrucibleContext";
import { deleteDataset } from "../../api/studioApi";
import {
  listClusters,
  listRemoteDatasets,
  pushDatasetToCluster,
  pullDatasetFromCluster,
  deleteRemoteDataset,
} from "../../api/remoteApi";
import type { ClusterConfig, RemoteDatasetInfo } from "../../types/remote";
import { ConfirmDeleteModal } from "../../components/shared/ConfirmDeleteModal";
import { RegistryRow, type RowItem } from "../../components/shared/RegistryRow";

interface DatasetListPanelProps {
  onSelect?: (dataset: string) => void;
  refreshKey?: number;
  onRefreshingChange?: (busy: boolean) => void;
}

type ListTab = "local" | "remote";

export function DatasetListPanel({ onSelect, refreshKey, onRefreshingChange }: DatasetListPanelProps) {
  const { dataRoot, datasets, selectedDataset, setSelectedDataset, refreshDatasets } = useCrucible();

  const [listTab, setListTab] = useState<ListTab>("local");
  const [clusters, setClusters] = useState<ClusterConfig[]>([]);
  const [selectedCluster, setSelectedCluster] = useState("");
  const [remoteDatasets, setRemoteDatasets] = useState<RemoteDatasetInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [pushing, setPushing] = useState<string | null>(null);
  const [pulling, setPulling] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [pushTarget, setPushTarget] = useState<string | null>(null);
  const [pushResult, setPushResult] = useState<{ ds: string; cluster: string } | null>(null);

  useEffect(() => {
    listClusters(dataRoot).then((c) => {
      setClusters(c);
      if (c.length > 0) setSelectedCluster(c[0].name);
    }).catch(console.error);
  }, [dataRoot]);

  const fetchRemoteDatasets = useCallback(async (bypassCache?: boolean) => {
    if (!selectedCluster) return;
    setLoading(true);
    onRefreshingChange?.(true);
    try {
      const ds = await listRemoteDatasets(dataRoot, selectedCluster, bypassCache);
      setRemoteDatasets(ds);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
      onRefreshingChange?.(false);
    }
  }, [dataRoot, selectedCluster, onRefreshingChange]);

  // Track whether refreshKey changed (user clicked Refresh) to bypass cache
  const prevRefreshKey = useRef(refreshKey);
  useEffect(() => {
    const userTriggered = refreshKey !== prevRefreshKey.current;
    prevRefreshKey.current = refreshKey;
    if (listTab === "remote" && selectedCluster) {
      fetchRemoteDatasets(userTriggered).catch(console.error);
    } else {
      onRefreshingChange?.(false);
    }
  }, [listTab, selectedCluster, refreshKey, fetchRemoteDatasets]);

  async function confirmDelete(): Promise<void> {
    if (!pendingDelete) return;
    setDeleting(true);
    try {
      if (isLocal) {
        await deleteDataset(dataRoot, pendingDelete);
        await refreshDatasets();
      } else {
        await deleteRemoteDataset(dataRoot, selectedCluster, pendingDelete);
        await fetchRemoteDatasets();
      }
    } finally {
      setPendingDelete(null);
      setDeleting(false);
    }
  }

  function handlePushClick(ds: string) {
    if (clusters.length === 0) return;
    if (clusters.length === 1) {
      doPush(ds, clusters[0].name);
    } else {
      setPushTarget(pushTarget === ds ? null : ds);
    }
  }

  async function doPush(ds: string, cluster: string) {
    setPushTarget(null);
    setPushResult(null);
    setPushing(ds);
    try {
      await pushDatasetToCluster(dataRoot, cluster, ds);
      setPushResult({ ds, cluster });
      setTimeout(() => setPushResult(null), 4000);
    } catch (e) {
      console.error(e);
    } finally {
      setPushing(null);
    }
  }

  async function handlePull(ds: string) {
    setPulling(ds);
    try {
      await pullDatasetFromCluster(dataRoot, selectedCluster, ds);
      await refreshDatasets();
    } catch (e) {
      console.error(e);
    } finally {
      setPulling(null);
    }
  }

  const isLocal = listTab === "local";

  // Build unified row data for both tabs
  const localRows: RowItem[] = datasets.map((d) => ({
    name: d.name, sizeBytes: d.sizeBytes,
  }));
  const remoteRows: RowItem[] = remoteDatasets.map((d) => ({
    name: d.name, sizeBytes: d.sizeBytes,
  }));

  const rows = isLocal ? localRows : remoteRows;
  const emptyMsg = isLocal
    ? "No datasets found. Use the Ingest tab to add data."
    : "No datasets on this cluster.";

  return (
    <div className="panel overflow-auto">
      <h3 className="panel-title">Datasets</h3>

      <div className="tab-list">
        <button
          className={`tab-item ${listTab === "local" ? "active" : ""}`}
          onClick={() => setListTab("local")}
        >
          Local
        </button>
        <button
          className={`tab-item ${listTab === "remote" ? "active" : ""}`}
          onClick={() => setListTab("remote")}
        >
          Remote
        </button>
      </div>

      {!isLocal && clusters.length === 0 && (
        <p className="text-tertiary">No clusters registered. Add one in Settings.</p>
      )}

      {!isLocal && clusters.length > 0 && (
        <select
          className="input"
          value={selectedCluster}
          onChange={(e) => setSelectedCluster(e.target.value)}
          style={{ marginBottom: 8, width: "100%" }}
        >
          {clusters.map((c) => (
            <option key={c.name} value={c.name}>{c.name}</option>
          ))}
        </select>
      )}

      {!isLocal && loading ? (
        <div style={{ display: "flex", justifyContent: "center", padding: 16 }}>
          <Loader2 size={20} className="spin" />
        </div>
      ) : rows.length === 0 ? (
        <p className="text-tertiary">{emptyMsg}</p>
      ) : (
        <div>
          {rows.map((row) => (
            <div key={row.name}>
              <RegistryRow
                name={row.name}
                sizeBytes={row.sizeBytes}
                selected={isLocal && row.name === selectedDataset}
                transferBusy={isLocal ? pushing === row.name : pulling === row.name}
                transferIcon={isLocal ? "upload" : "download"}
                showTransfer={isLocal ? clusters.length > 0 : true}
                onSelect={isLocal ? () => {
                  if (onSelect) onSelect(row.name);
                  else setSelectedDataset(row.name);
                  setPushTarget(null);
                } : undefined}
                onTransfer={() => isLocal
                  ? handlePushClick(row.name)
                  : handlePull(row.name).catch(console.error)}
                onDelete={() => setPendingDelete(row.name)}
              />
              {isLocal && pushTarget === row.name && clusters.length > 1 && (
                <div style={{ padding: "4px 8px 8px", display: "flex", gap: 4, alignItems: "center" }}>
                  <span className="text-xs text-tertiary">Push to:</span>
                  {clusters.map((c) => (
                    <button
                      key={c.name}
                      className="btn btn-sm"
                      onClick={() => doPush(row.name, c.name).catch(console.error)}
                    >
                      {c.name}
                    </button>
                  ))}
                </div>
              )}
              {pushResult?.ds === row.name && (
                <div style={{ padding: "2px 8px 4px" }}>
                  <span className="text-xs" style={{ color: "var(--color-success)" }}>
                    Pushed to {pushResult.cluster}
                  </span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {pendingDelete && (
        <ConfirmDeleteModal
          title="Delete Dataset"
          itemName={pendingDelete}
          isDeleting={deleting}
          onConfirm={() => confirmDelete().catch(console.error)}
          onCancel={() => setPendingDelete(null)}
        />
      )}
    </div>
  );
}
