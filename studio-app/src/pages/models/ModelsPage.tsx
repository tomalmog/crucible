import { useCallback, useEffect, useMemo, useState } from "react";
import { Download, Loader2, Plus, Trash2, Upload } from "lucide-react";
import { PageHeader } from "../../components/shared/PageHeader";
import { TabBar } from "../../components/shared/TabBar";
import { DetailPage } from "../../components/shared/DetailPage";
import { ListRow } from "../../components/shared/ListRow";
import { EmptyState } from "../../components/shared/EmptyState";
import { RegisterModelModal } from "../../components/shared/RegisterModelModal";
import { ConfirmDeleteModal } from "../../components/shared/ConfirmDeleteModal";
import { formatSize } from "../../components/shared/RegistryRow";
import { ClusterSelect } from "../../components/shared/ClusterSelect";
import { useCrucible } from "../../context/CrucibleContext";
import { useCrucibleCommand } from "../../hooks/useCrucibleCommand";
import { startCrucibleCommand, getCrucibleCommandStatus } from "../../api/studioApi";
import { listClusters, getRemoteModelSizes } from "../../api/remoteApi";
import { ModelOverview } from "./ModelOverview";
import { ModelMergeForm } from "./ModelMergeForm";
import type { ModelEntry } from "../../types/models";
import type { ClusterConfig } from "../../types/remote";

type DetailTab = "overview" | "merge";
type LocationTab = "local" | "remote";
const DETAIL_TABS = ["overview", "merge"] as const;
const LOCATION_TABS = ["local", "remote"] as const;

const POLL_MS = 400;

export function ModelsPage() {
  const { dataRoot, models, setSelectedModel, refreshModels } = useCrucible();
  const command = useCrucibleCommand();
  const [detailEntry, setDetailEntry] = useState<ModelEntry | null>(null);
  const [tab, setTab] = useState<DetailTab>("overview");
  const [locationTab, setLocationTab] = useState<LocationTab>("local");
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [showRegister, setShowRegister] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<ModelEntry | null>(null);
  const [deleteLocalFiles, setDeleteLocalFiles] = useState(true);

  const [clusters, setClusters] = useState<ClusterConfig[]>([]);
  const [selectedCluster, setSelectedCluster] = useState("");
  const [remoteSizes, setRemoteSizes] = useState<Record<string, number>>({});
  const [transferring, setTransferring] = useState<Set<string>>(new Set());
  const [transferError, setTransferError] = useState<string | null>(null);
  const hostToCluster = useMemo(() => {
    const map = new Map<string, string>();
    for (const c of clusters) map.set(c.host, c.name);
    return map;
  }, [clusters]);

  useEffect(() => {
    listClusters(dataRoot)
      .then((c) => {
        setClusters(c);
        if (c.length > 0) setSelectedCluster(c[0].name);
      })
      .catch(console.error);
  }, [dataRoot]);

  const fetchRemoteSizes = useCallback(async (bypassCache?: boolean) => {
    if (!selectedCluster) return;
    try {
      const sizes = await getRemoteModelSizes(dataRoot, selectedCluster, bypassCache);
      setRemoteSizes(sizes);
    } catch {
      setRemoteSizes({});
    }
  }, [dataRoot, selectedCluster]);

  useEffect(() => {
    if (locationTab === "remote" && selectedCluster) {
      fetchRemoteSizes().catch(console.error);
    }
  }, [locationTab, selectedCluster, fetchRemoteSizes]);

  const pollUntilDone = useCallback(async (taskId: string) => {
    while (true) {
      const s = await getCrucibleCommandStatus(taskId);
      if (s.status !== "running") return s;
      await new Promise((r) => setTimeout(r, POLL_MS));
    }
  }, []);

  async function handlePushModel(entry: ModelEntry): Promise<void> {
    const name = entry.modelName;
    setTransferError(null);
    setTransferring((prev) => new Set(prev).add(name));
    try {
      const { task_id } = await startCrucibleCommand(dataRoot, [
        "remote", "push-model", "--cluster", selectedCluster, "--name", name,
      ]);
      const status = await pollUntilDone(task_id);
      if (status.status === "failed") throw new Error(status.stderr || "Push failed");
      await refreshModels();
    } catch (err) {
      setTransferError(`Push "${name}" failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setTransferring((prev) => { const n = new Set(prev); n.delete(name); return n; });
    }
  }

  async function handlePullModel(entry: ModelEntry): Promise<void> {
    const name = entry.modelName;
    setTransferError(null);
    setTransferring((prev) => new Set(prev).add(name));
    try {
      const { task_id } = await startCrucibleCommand(dataRoot, [
        "model", "pull", "--name", name,
      ]);
      const status = await pollUntilDone(task_id);
      if (status.status === "failed") throw new Error(status.stderr || "Pull failed");
      await refreshModels();
    } catch (err) {
      setTransferError(`Pull "${name}" failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setTransferring((prev) => { const n = new Set(prev); n.delete(name); return n; });
    }
  }

  function handleSelect(entry: ModelEntry) {
    setSelectedModel(entry);
    setDetailEntry(entry);
    setTab("overview");
  }

  function handleBack() {
    setDetailEntry(null);
  }

  async function handleDelete(): Promise<void> {
    if (!pendingDelete) return;
    const args = ["model", "delete", "--name", pendingDelete.modelName, "--yes"];
    if (deleteLocalFiles && pendingDelete.hasLocal) args.push("--delete-local");
    if (pendingDelete.hasRemote) args.push("--include-remote");
    await command.run(dataRoot, args);
    setPendingDelete(null);
    if (detailEntry?.modelName === pendingDelete.modelName) setDetailEntry(null);
    await refreshModels();
  }

  async function handleRefresh(): Promise<void> {
    setIsRefreshing(true);
    await refreshModels();
    if (locationTab === "remote") {
      await fetchRemoteSizes(true);
    }
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

  const clusterHost = clusters.find((c) => c.name === selectedCluster)?.host ?? selectedCluster;
  const filtered = models.filter((m) =>
    locationTab === "local"
      ? m.hasLocal
      : m.hasRemote && (!selectedCluster || m.remoteHost === clusterHost),
  );

  const emptyMsg = locationTab === "local"
    ? "No local models. Train a model or download one from the Hub."
    : "No remote models.";

  return (
    <>
      <PageHeader title="Models">
        {clusters.length > 0 && (
          <ClusterSelect clusters={clusters} value={selectedCluster} onChange={setSelectedCluster} />
        )}
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

      {transferError && (
        <p className="error-text" style={{ marginBottom: 8 }}>{transferError}</p>
      )}

      {filtered.length === 0 ? (
        <EmptyState title="No models" description={emptyMsg} />
      ) : (
        <div className="panel panel-flush">
          {filtered.map((m) => (
            <ListRow
              key={m.modelName + m.modelPath}
              name={m.modelName}
              meta={
                <>
                  {locationTab === "local" && m.hasRemote && <span className="badge">Also Remote</span>}
                  {locationTab === "remote" && m.remoteHost && (
                    <span className="badge">{hostToCluster.get(m.remoteHost) || m.remoteHost}</span>
                  )}
                  <span>{formatSize(locationTab === "remote" ? (remoteSizes[m.modelName] ?? m.sizeBytes) : m.sizeBytes)}</span>
                </>
              }
              actions={
                <>
                  {locationTab === "local" ? (
                    <button
                      className="btn btn-ghost btn-sm btn-icon"
                      title="Push to remote cluster"
                      disabled={clusters.length === 0 || transferring.has(m.modelName)}
                      onClick={(e) => { e.stopPropagation(); handlePushModel(m).catch(console.error); }}
                    >
                      {transferring.has(m.modelName) ? <Loader2 size={14} className="spin" /> : <Upload size={14} />}
                    </button>
                  ) : (
                    <button
                      className="btn btn-ghost btn-sm btn-icon"
                      title="Pull to local"
                      disabled={transferring.has(m.modelName)}
                      onClick={(e) => { e.stopPropagation(); handlePullModel(m).catch(console.error); }}
                    >
                      {transferring.has(m.modelName) ? <Loader2 size={14} className="spin" /> : <Download size={14} />}
                    </button>
                  )}
                  <button
                    className="btn btn-ghost btn-sm btn-icon"
                    title="Delete model"
                    onClick={() => { setDeleteLocalFiles(true); setPendingDelete(m); }}
                  >
                    <Trash2 size={14} />
                  </button>
                </>
              }
              onClick={() => handleSelect(m)}
            />
          ))}
        </div>
      )}

      <button className="fab-add" onClick={() => setShowRegister(true)} title="Register model">
        <Plus size={22} />
      </button>

      {showRegister && (
        <RegisterModelModal
          onComplete={() => setShowRegister(false)}
          onClose={() => setShowRegister(false)}
        />
      )}

      {pendingDelete && (
        <ConfirmDeleteModal
          title="Delete Model"
          itemName={pendingDelete.modelName}
          isDeleting={command.isRunning}
          checkbox={pendingDelete.hasLocal ? {
            label: "Delete files from disk",
            checked: deleteLocalFiles,
            onChange: setDeleteLocalFiles,
          } : undefined}
          onConfirm={() => handleDelete().catch(console.error)}
          onCancel={() => setPendingDelete(null)}
        />
      )}
    </>
  );
}
