import { useCallback, useEffect, useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { useCrucible } from "../../context/CrucibleContext";
import { listClusters } from "../../api/remoteApi";
import { startCrucibleCommand, getCrucibleCommandStatus } from "../../api/studioApi";
import type { ClusterConfig } from "../../types/remote";
import { ClusterCard } from "./ClusterCard";
import { ClusterRegisterForm } from "./ClusterRegisterForm";
import { Loader2, Plus, Server } from "lucide-react";

type View = "list" | "register" | "edit";

export function ClustersPage() {
  const { dataRoot } = useCrucible();
  const [view, setView] = useState<View>("list");
  const [clusters, setClusters] = useState<ClusterConfig[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [editingCluster, setEditingCluster] = useState<ClusterConfig | null>(null);

  const refresh = useCallback(() => {
    if (!dataRoot) return;
    listClusters(dataRoot)
      .then(setClusters)
      .catch(() => setClusters([]))
      .finally(() => setIsLoading(false));
  }, [dataRoot]);

  // Fetch clusters on mount
  useEffect(() => {
    refresh();
  }, [refresh]);

  async function waitForTask(taskId: string, onProgress?: (output: string) => void) {
    let status = await getCrucibleCommandStatus(taskId);
    while (status.status === "running") {
      onProgress?.([status.stdout, status.stderr].filter(Boolean).join("\n"));
      await new Promise((r) => setTimeout(r, 500));
      status = await getCrucibleCommandStatus(taskId);
    }
    onProgress?.([status.stdout, status.stderr].filter(Boolean).join("\n"));
    return status;
  }

  function handleRemove(name: string) {
    startCrucibleCommand(dataRoot, ["remote", "remove-cluster", "--cluster", name])
      .then(({ task_id }) => waitForTask(task_id))
      .then(() => refresh())
      .catch(() => {});
  }

  async function handleValidate(name: string, onProgress?: (output: string) => void) {
    const { task_id } = await startCrucibleCommand(dataRoot, ["remote", "validate-cluster", "--cluster", name]);
    const status = await waitForTask(task_id, onProgress);
    if (status.status === "failed") throw new Error(status.stderr || "Validation failed");
    refresh();
  }

  async function handleResetEnv(name: string) {
    const { task_id } = await startCrucibleCommand(dataRoot, ["remote", "reset-env", "--cluster", name]);
    await waitForTask(task_id);
  }

  if (view === "register") {
    return (
      <>
        <PageHeader title="Register Cluster">
          <button className="btn btn-sm" onClick={() => setView("list")}>
            Back to List
          </button>
        </PageHeader>
        <ClusterRegisterForm onRegistered={() => { refresh(); setView("list"); }} />
      </>
    );
  }

  if (view === "edit" && editingCluster) {
    return (
      <>
        <PageHeader title="Edit Cluster">
          <button className="btn btn-sm" onClick={() => { setEditingCluster(null); setView("list"); }}>
            Back to List
          </button>
        </PageHeader>
        <ClusterRegisterForm
          editCluster={editingCluster}
          onRegistered={() => { refresh(); setEditingCluster(null); setView("list"); }}
        />
      </>
    );
  }

  return (
    <>
      <PageHeader title="Clusters">
        <button className="btn btn-primary btn-sm" onClick={() => setView("register")}>
          <Plus size={14} /> Register Cluster
        </button>
      </PageHeader>

      {isLoading ? (
        <div style={{ display: "flex", justifyContent: "center", padding: 32 }}>
          <Loader2 size={24} className="spin" />
        </div>
      ) : clusters.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">
            <Server />
          </div>
          <h3>No clusters registered</h3>
          <p>Register a Slurm cluster to submit remote training jobs.</p>
        </div>
      ) : (
        <div className="stack-md">
          {clusters.map((c) => (
            <ClusterCard
              key={c.name}
              cluster={c}
              onRemove={() => handleRemove(c.name)}
              onValidate={(onProgress) => handleValidate(c.name, onProgress)}
              onResetEnv={() => handleResetEnv(c.name)}
              onEdit={() => { setEditingCluster(c); setView("edit"); }}
            />
          ))}
        </div>
      )}
    </>
  );
}
