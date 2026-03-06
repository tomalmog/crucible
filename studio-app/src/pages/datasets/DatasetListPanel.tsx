import { useState, useEffect, useCallback } from "react";
import { Trash2, Upload, Download, Loader2 } from "lucide-react";
import { useForge } from "../../context/ForgeContext";
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
}

type ListTab = "local" | "remote";

export function DatasetListPanel({ onSelect }: DatasetListPanelProps) {
  const { dataRoot, datasets, selectedDataset, setSelectedDataset, refreshDatasets } = useForge();

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

  const fetchRemoteDatasets = useCallback(async () => {
    if (!selectedCluster) return;
    setLoading(true);
    try {
      const ds = await listRemoteDatasets(dataRoot, selectedCluster);
      setRemoteDatasets(ds);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [dataRoot, selectedCluster]);

  useEffect(() => {
    if (listTab === "remote" && selectedCluster) {
      fetchRemoteDatasets().catch(console.error);
    }
  }, [listTab, selectedCluster, fetchRemoteDatasets]);

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
      // Toggle cluster picker for this dataset
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

      {listTab === "local" && <LocalList
        datasets={datasets}
        selectedDataset={selectedDataset}
        confirmingDelete={confirmingDelete}
        pushing={pushing}
        pushTarget={pushTarget}
        pushResult={pushResult}
        clusters={clusters}
        onSelect={(ds) => {
          if (onSelect) onSelect(ds);
          else setSelectedDataset(ds);
          setConfirmingDelete(null);
          setPushTarget(null);
        }}
        onDelete={(ds) => handleDeleteLocal(ds).catch(console.error)}
        onPushClick={handlePushClick}
        onPushConfirm={(ds, cluster) => doPush(ds, cluster).catch(console.error)}
      />}

      {listTab === "remote" && <RemoteList
        clusters={clusters}
        selectedCluster={selectedCluster}
        remoteDatasets={remoteDatasets}
        loading={loading}
        pulling={pulling}
        confirmingDelete={confirmingDelete}
        onClusterChange={setSelectedCluster}
        onPull={(ds) => handlePull(ds).catch(console.error)}
        onDelete={(ds) => handleDeleteRemote(ds).catch(console.error)}
      />}
    </div>
  );
}

/* ---- Local dataset list ---- */

function LocalList({ datasets, selectedDataset, confirmingDelete, pushing, pushTarget, pushResult, clusters, onSelect, onDelete, onPushClick, onPushConfirm }: {
  datasets: string[];
  selectedDataset: string | null;
  confirmingDelete: string | null;
  pushing: string | null;
  pushTarget: string | null;
  pushResult: { ds: string; cluster: string } | null;
  clusters: ClusterConfig[];
  onSelect: (ds: string) => void;
  onDelete: (ds: string) => void;
  onPushClick: (ds: string) => void;
  onPushConfirm: (ds: string, cluster: string) => void;
}) {
  if (datasets.length === 0) {
    return <p className="text-tertiary">No datasets found. Use the Ingest tab to add data.</p>;
  }
  return (
    <div>
      {datasets.map((ds) => (
        <div key={ds}>
          <div className="flex-row" style={{ alignItems: "center" }}>
            <button
              className={`nav-item ${ds === selectedDataset ? "active" : ""}`}
              style={{ flex: 1, minWidth: 0 }}
              onClick={() => onSelect(ds)}
            >
              {ds}
            </button>
            <div style={{ display: "flex", gap: 2, flexShrink: 0, marginLeft: "auto" }}>
              {clusters.length > 0 && (
                <button
                  className="btn btn-ghost btn-sm btn-icon"
                  onClick={() => onPushClick(ds)}
                  title="Push to cluster"
                  disabled={pushing === ds}
                >
                  {pushing === ds
                    ? <Loader2 size={12} className="spin" />
                    : <Upload size={12} />}
                </button>
              )}
              {confirmingDelete === ds ? (
                <button
                  className="btn btn-sm"
                  style={{ color: "var(--color-error)" }}
                  onClick={() => onDelete(ds)}
                >
                  Delete?
                </button>
              ) : (
                <button
                  className="btn btn-ghost btn-sm btn-icon"
                  onClick={() => onDelete(ds)}
                  title="Delete dataset"
                >
                  <Trash2 size={12} />
                </button>
              )}
            </div>
          </div>
          {pushTarget === ds && clusters.length > 1 && (
            <div style={{ padding: "4px 8px 8px", display: "flex", gap: 4, alignItems: "center" }}>
              <span className="text-xs text-tertiary">Push to:</span>
              {clusters.map((c) => (
                <button
                  key={c.name}
                  className="btn btn-sm"
                  onClick={() => onPushConfirm(ds, c.name)}
                >
                  {c.name}
                </button>
              ))}
            </div>
          )}
          {pushResult?.ds === ds && (
            <div style={{ padding: "2px 8px 4px" }}>
              <span className="text-xs" style={{ color: "var(--color-success)" }}>
                Pushed to {pushResult.cluster}
              </span>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

/* ---- Remote dataset list ---- */

function RemoteList({ clusters, selectedCluster, remoteDatasets, loading, pulling, confirmingDelete, onClusterChange, onPull, onDelete }: {
  clusters: ClusterConfig[];
  selectedCluster: string;
  remoteDatasets: RemoteDatasetInfo[];
  loading: boolean;
  pulling: string | null;
  confirmingDelete: string | null;
  onClusterChange: (name: string) => void;
  onPull: (ds: string) => void;
  onDelete: (ds: string) => void;
}) {
  if (clusters.length === 0) {
    return <p className="text-tertiary">No clusters registered. Add one in Settings.</p>;
  }
  return (
    <div>
      <select
        className="input"
        value={selectedCluster}
        onChange={(e) => onClusterChange(e.target.value)}
        style={{ marginBottom: 8, width: "100%" }}
      >
        {clusters.map((c) => (
          <option key={c.name} value={c.name}>{c.name}</option>
        ))}
      </select>

      {loading ? (
        <div style={{ display: "flex", justifyContent: "center", padding: 16 }}>
          <Loader2 size={20} className="spin" />
        </div>
      ) : remoteDatasets.length === 0 ? (
        <p className="text-tertiary">No datasets on this cluster.</p>
      ) : (
        <div>
          {remoteDatasets.map((rd) => (
            <div key={rd.name} className="flex-row" style={{ alignItems: "center", padding: "4px 8px", gap: 8 }}>
              <span className="text-sm" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", minWidth: 0 }}>{rd.name}</span>
              <span className="text-xs text-tertiary" style={{ flexShrink: 0 }}>
                {formatSize(rd.sizeBytes)}
              </span>
              <div style={{ display: "flex", gap: 2, flexShrink: 0, marginLeft: "auto" }}>
                <button
                  className="btn btn-ghost btn-sm btn-icon"
                  onClick={() => onPull(rd.name)}
                  title="Pull to local"
                  disabled={pulling === rd.name}
                >
                  {pulling === rd.name
                    ? <Loader2 size={12} className="spin" />
                    : <Download size={12} />}
                </button>
                {confirmingDelete === rd.name ? (
                  <button
                    className="btn btn-sm"
                    style={{ color: "var(--color-error)" }}
                    onClick={() => onDelete(rd.name)}
                  >
                    Delete?
                  </button>
                ) : (
                  <button
                    className="btn btn-ghost btn-sm btn-icon"
                    onClick={() => onDelete(rd.name)}
                    title="Delete remote dataset"
                  >
                    <Trash2 size={12} />
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
