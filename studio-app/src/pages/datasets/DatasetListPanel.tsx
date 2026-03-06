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

  async function handlePush(ds: string) {
    if (clusters.length === 0) return;
    setPushing(ds);
    try {
      await pushDatasetToCluster(dataRoot, clusters[0].name, ds);
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
        hasClusters={clusters.length > 0}
        onSelect={(ds) => {
          if (onSelect) onSelect(ds);
          else setSelectedDataset(ds);
          setConfirmingDelete(null);
        }}
        onDelete={(ds) => handleDeleteLocal(ds).catch(console.error)}
        onPush={(ds) => handlePush(ds).catch(console.error)}
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

function LocalList({ datasets, selectedDataset, confirmingDelete, pushing, hasClusters, onSelect, onDelete, onPush }: {
  datasets: string[];
  selectedDataset: string | null;
  confirmingDelete: string | null;
  pushing: string | null;
  hasClusters: boolean;
  onSelect: (ds: string) => void;
  onDelete: (ds: string) => void;
  onPush: (ds: string) => void;
}) {
  if (datasets.length === 0) {
    return <p className="text-tertiary">No datasets found. Use the Ingest tab to add data.</p>;
  }
  return (
    <div>
      {datasets.map((ds) => (
        <div key={ds} className="flex-row" style={{ alignItems: "center" }}>
          <button
            className={`nav-item ${ds === selectedDataset ? "active" : ""}`}
            style={{ flex: 1 }}
            onClick={() => onSelect(ds)}
          >
            {ds}
          </button>
          {hasClusters && (
            <button
              className="btn btn-ghost btn-sm btn-icon"
              style={{ flexShrink: 0 }}
              onClick={() => onPush(ds)}
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
              style={{ color: "var(--color-error)", flexShrink: 0 }}
              onClick={() => onDelete(ds)}
            >
              Delete?
            </button>
          ) : (
            <button
              className="btn btn-ghost btn-sm btn-icon"
              style={{ flexShrink: 0 }}
              onClick={() => onDelete(ds)}
              title="Delete dataset"
            >
              <Trash2 size={12} />
            </button>
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
            <div key={rd.name} className="flex-row" style={{ alignItems: "center" }}>
              <div style={{ flex: 1, padding: "4px 8px" }}>
                <span className="text-sm">{rd.name}</span>
                <span className="text-xs text-tertiary" style={{ marginLeft: 8 }}>
                  {rd.recordCount} records
                </span>
              </div>
              <button
                className="btn btn-ghost btn-sm btn-icon"
                style={{ flexShrink: 0 }}
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
                  style={{ color: "var(--color-error)", flexShrink: 0 }}
                  onClick={() => onDelete(rd.name)}
                >
                  Delete?
                </button>
              ) : (
                <button
                  className="btn btn-ghost btn-sm btn-icon"
                  style={{ flexShrink: 0 }}
                  onClick={() => onDelete(rd.name)}
                  title="Delete remote dataset"
                >
                  <Trash2 size={12} />
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
