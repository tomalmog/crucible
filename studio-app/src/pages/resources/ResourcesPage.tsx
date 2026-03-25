import { RefreshCw, Loader2 } from "lucide-react";
import { PageHeader } from "../../components/shared/PageHeader";
import { useResourceData } from "../../hooks/useResourceData";
import { StoragePanel } from "./StoragePanel";
import { HardwarePanel } from "./HardwarePanel";
import { ClusterInfoPanel } from "./ClusterInfoPanel";
import { ActivityPanel } from "./ActivityPanel";
import { CleanupPanel } from "./CleanupPanel";

export function ResourcesPage() {
  const {
    storage,
    orphans,
    hardware,
    clusters,
    remoteStorage,
    localJobs,
    loading,
    refresh,
  } = useResourceData();

  return (
    <>
      <PageHeader title="Resources">
        <button className="btn btn-sm" onClick={refresh} disabled={loading}>
          {loading ? <Loader2 size={14} className="spin" /> : <RefreshCw size={14} />}
          Refresh
        </button>
      </PageHeader>

      <div className="resource-card-list">
        <StoragePanel storage={storage} remoteStorage={remoteStorage} />
        <HardwarePanel hardware={hardware} clusters={clusters} />
        <ClusterInfoPanel remoteStorage={remoteStorage} clusters={clusters} loading={loading} />
        <ActivityPanel localJobs={localJobs} />
        <CleanupPanel orphans={orphans} storage={storage} onRefresh={refresh} />
      </div>
    </>
  );
}
