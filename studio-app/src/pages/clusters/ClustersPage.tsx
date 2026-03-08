import { useCallback, useEffect, useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { useForge } from "../../context/ForgeContext";
import { listClusters } from "../../api/remoteApi";
import { startForgeCommand, getForgeCommandStatus } from "../../api/studioApi";
import type { ClusterConfig } from "../../types/remote";
import { ClusterCard } from "./ClusterCard";
import { ClusterRegisterForm } from "./ClusterRegisterForm";
import { Loader2, Plus, Server } from "lucide-react";

type View = "list" | "register";

export function ClustersPage() {
  const { dataRoot } = useForge();
  const [view, setView] = useState<View>("list");
  const [clusters, setClusters] = useState<ClusterConfig[]>([]);
  const [isLoading, setIsLoading] = useState(true);

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

  async function waitForTask(taskId: string) {
    let status = await getForgeCommandStatus(taskId);
    while (status.status === "running") {
      await new Promise((r) => setTimeout(r, 500));
      status = await getForgeCommandStatus(taskId);
    }
  }

  function handleRemove(name: string) {
    startForgeCommand(dataRoot, ["remote", "remove-cluster", "--cluster", name])
      .then(({ task_id }) => waitForTask(task_id))
      .then(() => refresh())
      .catch(() => {});
  }

  async function handleValidate(name: string) {
    const { task_id } = await startForgeCommand(dataRoot, ["remote", "validate-cluster", "--cluster", name]);
    await waitForTask(task_id);
    refresh();
  }

  async function handleResetEnv(name: string) {
    const { task_id } = await startForgeCommand(dataRoot, ["remote", "reset-env", "--cluster", name]);
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
              onValidate={() => handleValidate(c.name)}
              onResetEnv={() => handleResetEnv(c.name)}
            />
          ))}
        </div>
      )}
    </>
  );
}
