import { useState, useEffect, useCallback, useRef } from "react";
import { Loader2 } from "lucide-react";
import { useCrucible } from "../../context/CrucibleContext";
import { useCrucibleCommand } from "../../hooks/useCrucibleCommand";
import {
  listClusters,
  getRemoteModelSizes,
} from "../../api/remoteApi";
import type { ClusterConfig } from "../../types/remote";
import type { ModelEntry } from "../../types/models";
import { ConfirmDeleteModal } from "../../components/shared/ConfirmDeleteModal";
import { RegistryRow, type RowItem } from "../../components/shared/RegistryRow";
import { startCrucibleCommand, getCrucibleCommandStatus } from "../../api/studioApi";

type ListTab = "local" | "remote";

interface ModelListPanelProps {
  refreshKey?: number;
  onRefreshingChange?: (busy: boolean) => void;
}

export function ModelListPanel({ refreshKey, onRefreshingChange }: ModelListPanelProps) {
  const {
    dataRoot,
    models,
    selectedModel,
    setSelectedModel,
    refreshModels,
  } = useCrucible();
  const command = useCrucibleCommand();

  const [listTab, setListTab] = useState<ListTab>("local");
  const [clusters, setClusters] = useState<ClusterConfig[]>([]);
  const [selectedCluster, setSelectedCluster] = useState("");
  const [remoteSizes, setRemoteSizes] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(false);
  const [pulling, setPulling] = useState<string | null>(null);
  const [pushing, setPushing] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<ModelEntry | null>(null);
  const [pushTarget, setPushTarget] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  useEffect(() => {
    listClusters(dataRoot).then((c) => {
      setClusters(c);
      if (c.length > 0) setSelectedCluster(c[0].name);
    }).catch(console.error);
  }, [dataRoot]);

  const fetchRemoteModels = useCallback(async (bypassCache?: boolean) => {
    if (!selectedCluster) return;
    setLoading(true);
    onRefreshingChange?.(true);
    try {
      const sizes = await getRemoteModelSizes(dataRoot, selectedCluster, bypassCache);
      setRemoteSizes(sizes);
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
      fetchRemoteModels(userTriggered).catch(console.error);
    } else {
      onRefreshingChange?.(false);
    }
  }, [listTab, selectedCluster, refreshKey, fetchRemoteModels]);

  async function confirmDelete(): Promise<void> {
    if (!pendingDelete) return;
    const args = ["model", "delete", "--name", pendingDelete.modelName, "--yes"];
    if (pendingDelete.hasLocal) args.push("--delete-local");
    if (pendingDelete.hasRemote) args.push("--include-remote");
    await command.run(dataRoot, args);
    setPendingDelete(null);
    await refreshModels();
  }

  function handlePushClick(entry: ModelEntry) {
    if (clusters.length === 0) return;
    if (clusters.length === 1) {
      doPush(entry, clusters[0].name);
    } else {
      setPushTarget(pushTarget === entry.modelName ? null : entry.modelName);
    }
  }

  async function doPush(entry: ModelEntry, cluster: string): Promise<void> {
    setPushTarget(null);
    setPushing(entry.modelName);
    try {
      const task = await startCrucibleCommand(dataRoot, [
        "remote", "push-model",
        "--cluster", cluster,
        "--name", entry.modelName,
      ]);
      await pollUntilDone(task.task_id);
      await refreshModels();
    } catch (e) {
      console.error(e);
    } finally {
      setPushing(null);
    }
  }

  async function handlePull(entry: ModelEntry): Promise<void> {
    setPulling(entry.modelName);
    try {
      const task = await startCrucibleCommand(dataRoot, [
        "model", "pull", "--name", entry.modelName,
      ]);
      await pollUntilDone(task.task_id);
      await refreshModels();
    } catch (e) {
      console.error(e);
    } finally {
      setPulling(null);
    }
  }

  function pollUntilDone(taskId: string): Promise<void> {
    return new Promise((resolve, reject) => {
      pollRef.current = setInterval(async () => {
        try {
          const status = await getCrucibleCommandStatus(taskId);
          if (status.status !== "running") {
            if (pollRef.current) clearInterval(pollRef.current);
            pollRef.current = null;
            if (status.status === "completed") resolve();
            else reject(new Error(status.stderr || "Command failed"));
          }
        } catch (err) {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
          reject(err);
        }
      }, 2000);
    });
  }

  const isLocal = listTab === "local";

  // Build unified row data — same pattern as DatasetListPanel
  const clusterHost = clusters.find((c) => c.name === selectedCluster)?.host ?? selectedCluster;

  const localRows: RowItem[] = models
    .filter((m) => m.hasLocal)
    .map((m) => ({ name: m.modelName, sizeBytes: m.sizeBytes }));

  const remoteRows: RowItem[] = models
    .filter((m) => m.hasRemote && (!selectedCluster || m.remoteHost === clusterHost))
    .map((m) => ({ name: m.modelName, sizeBytes: remoteSizes[m.modelName] ?? 0 }));

  const rows = isLocal ? localRows : remoteRows;
  const emptyMsg = isLocal
    ? "No local models registered yet."
    : "No remote models found.";

  // Lookup ModelEntry by name for actions
  const entryByName = new Map(models.map((m) => [m.modelName, m]));

  return (
    <div className="panel overflow-auto">
      <h3 className="panel-title">Models</h3>

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
        <p className="text-tertiary">No clusters registered.</p>
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
          {rows.map((row) => {
            const entry = entryByName.get(row.name);
            return (
              <div key={row.name}>
                <RegistryRow
                  name={row.name}
                  sizeBytes={row.sizeBytes}
                  selected={row.name === selectedModel?.modelName}
                  transferBusy={isLocal ? pushing === row.name : pulling === row.name}
                  transferIcon={isLocal ? "upload" : "download"}
                  showTransfer={isLocal ? clusters.length > 0 : true}
                  onSelect={() => {
                    if (entry) setSelectedModel(entry);
                    setPushTarget(null);
                  }}
                  onTransfer={() => {
                    if (!entry) return;
                    if (isLocal) handlePushClick(entry);
                    else handlePull(entry).catch(console.error);
                  }}
                  onDelete={() => { if (entry) setPendingDelete(entry); }}
                />
                {isLocal && pushTarget === row.name && clusters.length > 1 && (
                  <div style={{ padding: "4px 8px 8px", display: "flex", gap: 4, alignItems: "center" }}>
                    <span className="text-xs text-tertiary">Push to:</span>
                    {clusters.map((c) => (
                      <button
                        key={c.name}
                        className="btn btn-sm"
                        onClick={() => { if (entry) doPush(entry, c.name).catch(console.error); }}
                      >
                        {c.name}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {pendingDelete && (
        <ConfirmDeleteModal
          title="Delete Model"
          itemName={pendingDelete.modelName}
          description="This will remove the model and associated files."
          isDeleting={command.isRunning}
          onConfirm={() => confirmDelete().catch(console.error)}
          onCancel={() => setPendingDelete(null)}
        />
      )}
    </div>
  );
}
