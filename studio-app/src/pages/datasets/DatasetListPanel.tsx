import { useState, useEffect, useCallback, useRef } from "react";
import { Trash2, Upload, Download, Loader2 } from "lucide-react";
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

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let size = bytes;
  for (const unit of units) {
    size /= 1024;
    if (size < 1024) return `${size.toFixed(1)} ${unit}`;
  }
  return `${size.toFixed(1)} PB`;
}

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
  const [confirmingDelete, setConfirmingDelete] = useState<string | null>(null);
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

  async function handleDeleteLocal(ds: string) {
    if (confirmingDelete !== ds) {
      setConfirmingDelete(ds);
      return;
    }
    await deleteDataset(dataRoot, ds);
    setConfirmingDelete(null);
    await refreshDatasets();
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

  async function handleDeleteRemote(ds: string) {
    if (confirmingDelete !== ds) {
      setConfirmingDelete(ds);
      return;
    }
    await deleteRemoteDataset(dataRoot, selectedCluster, ds);
    setConfirmingDelete(null);
    await fetchRemoteDatasets();
  }

  function switchTab(tab: ListTab) {
    setListTab(tab);
    setConfirmingDelete(null);
  }

  // Build unified row data for both tabs
  const localRows: RowItem[] = datasets.map((d) => ({
    name: d.name, sizeBytes: d.sizeBytes,
  }));
  const remoteRows: RowItem[] = remoteDatasets.map((d) => ({
    name: d.name, sizeBytes: d.sizeBytes,
  }));

  const isLocal = listTab === "local";
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
          onClick={() => switchTab("local")}
        >
          Local
        </button>
        <button
          className={`tab-item ${listTab === "remote" ? "active" : ""}`}
          onClick={() => switchTab("remote")}
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
              <DatasetRow
                name={row.name}
                sizeBytes={row.sizeBytes}
                selected={isLocal && row.name === selectedDataset}
                confirmingDelete={confirmingDelete === row.name}
                transferBusy={isLocal ? pushing === row.name : pulling === row.name}
                transferIcon={isLocal ? "upload" : "download"}
                showTransfer={isLocal ? clusters.length > 0 : true}
                onSelect={isLocal ? () => {
                  if (onSelect) onSelect(row.name);
                  else setSelectedDataset(row.name);
                  setConfirmingDelete(null);
                  setPushTarget(null);
                } : undefined}
                onTransfer={() => isLocal
                  ? handlePushClick(row.name)
                  : handlePull(row.name).catch(console.error)}
                onDelete={() => isLocal
                  ? handleDeleteLocal(row.name).catch(console.error)
                  : handleDeleteRemote(row.name).catch(console.error)}
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
    </div>
  );
}

/* ---- Shared dataset row ---- */

interface RowItem {
  name: string;
  sizeBytes: number;
}

function DatasetRow({ name, sizeBytes, selected, confirmingDelete, transferBusy, transferIcon, showTransfer, onSelect, onTransfer, onDelete }: {
  name: string;
  sizeBytes: number;
  selected?: boolean;
  confirmingDelete: boolean;
  transferBusy: boolean;
  transferIcon: "upload" | "download";
  showTransfer: boolean;
  onSelect?: () => void;
  onTransfer: () => void;
  onDelete: () => void;
}) {
  const TransferIcon = transferIcon === "upload" ? Upload : Download;
  const transferTitle = transferIcon === "upload" ? "Push to cluster" : "Pull to local";

  return (
    <div
      className={`flex-row${selected ? " active" : ""}`}
      style={{ alignItems: "center", padding: "4px 8px", gap: 8, cursor: onSelect ? "pointer" : undefined }}
      onClick={onSelect}
    >
      <span
        className="text-sm"
        style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
      >
        {name}
      </span>
      <span className="text-xs text-tertiary" style={{ flexShrink: 0 }}>
        {formatSize(sizeBytes)}
      </span>
      <div style={{ display: "flex", gap: 2, flexShrink: 0 }}>
        {showTransfer && (
          <button
            className="btn btn-ghost btn-sm btn-icon"
            onClick={(e) => { e.stopPropagation(); onTransfer(); }}
            title={transferTitle}
            disabled={transferBusy}
          >
            {transferBusy
              ? <Loader2 size={12} className="spin" />
              : <TransferIcon size={12} />}
          </button>
        )}
        {confirmingDelete ? (
          <button
            className="btn btn-sm"
            style={{ color: "var(--color-error)" }}
            onClick={(e) => { e.stopPropagation(); onDelete(); }}
          >
            Delete?
          </button>
        ) : (
          <button
            className="btn btn-ghost btn-sm btn-icon"
            onClick={(e) => { e.stopPropagation(); onDelete(); }}
            title="Delete dataset"
          >
            <Trash2 size={12} />
          </button>
        )}
      </div>
    </div>
  );
}
