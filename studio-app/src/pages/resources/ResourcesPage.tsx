import { RefreshCw, Loader2 } from "lucide-react";
import { PageHeader } from "../../components/shared/PageHeader";
import { useResourceData } from "../../hooks/useResourceData";
import { StoragePanel } from "./StoragePanel";
import { HardwarePanel } from "./HardwarePanel";
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
    remoteJobs,
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

      <div className="grid-2">
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <StoragePanel storage={storage} remoteStorage={remoteStorage} />
          <CleanupPanel orphans={orphans} storage={storage} onRefresh={refresh} />
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <HardwarePanel hardware={hardware} clusters={clusters} />
          <ActivityPanel localJobs={localJobs} remoteJobs={remoteJobs} />
        </div>
      </div>
    </>
  );
}
