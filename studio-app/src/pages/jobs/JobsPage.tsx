import { useEffect, useMemo, useRef, useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { useJobs } from "../../hooks/useJobs";
import { useRemoteJobs } from "../../hooks/useRemoteJobs";
import { useUnifiedJobs } from "../../hooks/useUnifiedJobs";
import { useCrucible } from "../../context/CrucibleContext";
import type { CommandTaskStatus } from "../../types";
import type { RemoteJobRecord } from "../../types/remote";
import type { JobRecord } from "../../types/jobs";
import { syncRemoteJobStatus } from "../../api/remoteApi";
import { Activity, Loader2 } from "lucide-react";
import { TabBar } from "../../components/shared/TabBar";
import { RemoteJobResultDetail } from "./RemoteJobResultDetail";
import { UnifiedJobResultDetail } from "./UnifiedJobResultDetail";
import { RemoteJobRow } from "./RemoteJobRow";
import { UnifiedJobRow } from "./UnifiedJobRow";

type StatusFilter = "all" | "running" | "completed" | "failed";
type LocationFilter = "all" | "local" | "remote";
type TaskTypeFilter = "all" | "training" | "eval" | "sweep" | "interp";

const INTERP_COMMANDS = new Set([
  "logit-lens", "activation-pca", "activation-patch",
  "linear-probe", "sae-train", "sae-analyze",
  "steer-compute", "steer-apply",
]);

function classifyRemoteJob(job: RemoteJobRecord): TaskTypeFilter {
  if (job.isSweep) return "sweep";
  if (job.trainingMethod === "eval") return "eval";
  if (job.trainingMethod && INTERP_COMMANDS.has(job.trainingMethod)) return "interp";
  return "training";
}

function classifyUnifiedJob(job: JobRecord): TaskTypeFilter {
  if (job.isSweep) return "sweep";
  if (job.jobType === "eval") return "eval";
  if (job.jobType && INTERP_COMMANDS.has(job.jobType)) return "interp";
  return "training";
}

const TYPE_TABS: readonly TaskTypeFilter[] = ["all", "training", "eval", "sweep", "interp"] as const;
const STATUS_OPTIONS: readonly StatusFilter[] = ["all", "running", "completed", "failed"] as const;
const LOCATION_OPTIONS: readonly LocationFilter[] = ["all", "local", "remote"] as const;

export function statusBadgeClass(status: string): string {
  switch (status) {
    case "running":
    case "submitting":
    case "pending":
      return "badge badge-accent";
    case "completed":
      return "badge badge-success";
    case "failed":
      return "badge badge-error";
    default:
      return "badge";
  }
}

export function jobAccentColor(status: string): string {
  switch (status) {
    case "running": case "submitting": case "pending": return "var(--accent)";
    case "completed": return "var(--success)";
    case "failed": return "var(--error)";
    default: return "var(--border)";
  }
}

function normalizeStatus(status: string): StatusFilter {
  if (status === "running" || status === "pending" || status === "submitting") return "running";
  if (status === "completed") return "completed";
  return "failed";
}

type MergedJob =
  | { kind: "remote"; job: RemoteJobRecord; sortKey: number }
  | { kind: "unified"; job: JobRecord; localTask?: CommandTaskStatus; sortKey: number };

export function JobsPage() {
  const { jobs, kill, rename } = useJobs();
  const { dataRoot, refreshModels } = useCrucible();
  const { jobs: remoteJobs, isLoading: isRemoteLoading, refresh: refreshRemote, removeJob: removeRemoteJob, cancelJob: cancelRemoteJob } = useRemoteJobs(dataRoot);
  const { jobs: unifiedJobs, isLoading: isUnifiedLoading, refresh: refreshUnified, removeJob: removeUnifiedJob, cancel: cancelUnifiedJob } = useUnifiedJobs(dataRoot);
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [locationFilter, setLocationFilter] = useState<LocationFilter>("all");
  const [typeFilter, setTypeFilter] = useState<TaskTypeFilter>("all");
  const [viewingRemoteJob, setViewingRemoteJob] = useState<RemoteJobRecord | null>(null);
  const [viewingUnifiedJob, setViewingUnifiedJob] = useState<{ job: JobRecord; localTask?: CommandTaskStatus } | null>(null);
  const syncedRef = useRef<Set<string>>(new Set());
  const remoteJobsRef = useRef(remoteJobs);
  remoteJobsRef.current = remoteJobs;

  // Auto-sync running remote job states via the CLI.
  useEffect(() => {
    if (!dataRoot) return;
    const syncRunning = () => {
      const now = Date.now();
      const running = remoteJobsRef.current.filter(
        (j) =>
          ((j.state === "running" || j.state === "pending") ||
           (j.state === "completed" && !j.modelPathRemote &&
            now - new Date(j.updatedAt).getTime() < 120_000)) &&
          !syncedRef.current.has(j.jobId),
      );
      for (const rj of running) {
        syncedRef.current.add(rj.jobId);
        syncRemoteJobStatus(dataRoot, rj.jobId)
          .then(() => {
            refreshRemote();
            refreshModels().catch(console.error);
          })
          .catch(console.error)
          .finally(() => syncedRef.current.delete(rj.jobId));
      }
    };
    syncRunning();
    const interval = setInterval(syncRunning, 5_000);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataRoot]);

  // Build a lookup of local task_id → CommandTaskStatus for unified local jobs
  const localTaskMap = useMemo(() => {
    const map = new Map<string, CommandTaskStatus>();
    for (const j of jobs) map.set(j.task_id, j);
    return map;
  }, [jobs]);

  // Collect IDs of unified jobs so we can de-dup legacy remote entries
  const unifiedBackendIds = useMemo(() => new Set(unifiedJobs.map((j) => j.backendJobId).filter(Boolean)), [unifiedJobs]);

  // Build merged sorted list
  const merged = useMemo(() => {
    const items: MergedJob[] = [];

    // Unified jobs (from .crucible/jobs/)
    for (const j of unifiedJobs) {
      if (statusFilter !== "all" && normalizeStatus(j.state) !== statusFilter) continue;
      const isRemote = j.backend !== "local";
      if (locationFilter === "local" && isRemote) continue;
      if (locationFilter === "remote" && !isRemote) continue;
      if (typeFilter !== "all" && classifyUnifiedJob(j) !== typeFilter) continue;
      const ts = new Date(j.createdAt).getTime() || 0;
      // For local unified jobs, find matching task store entry
      const localTask = j.backend === "local" ? localTaskMap.get(j.jobId) : undefined;
      items.push({ kind: "unified", job: j, localTask, sortKey: ts });
    }

    // Legacy remote jobs (from .crucible/remote-jobs/) — skip if already in unified store
    if (locationFilter !== "local") {
      for (const j of remoteJobs) {
        if (unifiedBackendIds.has(j.jobId)) continue; // already shown as unified
        if (statusFilter !== "all" && normalizeStatus(j.state) !== statusFilter) continue;
        if (typeFilter !== "all" && classifyRemoteJob(j) !== typeFilter) continue;
        const ts = new Date(j.submittedAt).getTime() || 0;
        items.push({ kind: "remote", job: j, sortKey: ts });
      }
    }

    items.sort((a, b) => b.sortKey - a.sortKey);
    return items;
  }, [remoteJobs, unifiedJobs, statusFilter, locationFilter, typeFilter, localTaskMap, unifiedBackendIds]);

  const runningCount = jobs.filter((j) => j.status === "running").length
    + remoteJobs.filter((j) => j.state === "running" || j.state === "pending" || j.state === "submitting").length
    + unifiedJobs.filter((j) => j.state === "running" || j.state === "pending" || j.state === "submitting").length;

  if (viewingRemoteJob) {
    return <RemoteJobResultDetail job={viewingRemoteJob} onBack={() => setViewingRemoteJob(null)} />;
  }

  if (viewingUnifiedJob) {
    return (
      <UnifiedJobResultDetail
        job={viewingUnifiedJob.job}
        localTask={viewingUnifiedJob.localTask}
        onBack={() => setViewingUnifiedJob(null)}
      />
    );
  }

  const isLoading = isRemoteLoading || isUnifiedLoading;

  return (
    <>
      <PageHeader title="Jobs">
        {runningCount > 0 && (
          <span className="badge badge-accent">{runningCount} running</span>
        )}
      </PageHeader>

      <TabBar tabs={TYPE_TABS} active={typeFilter} onChange={setTypeFilter} />

      <div className="filter-pills">
        {STATUS_OPTIONS.map((s) => (
          <button
            key={s}
            className={`filter-pill${statusFilter === s ? " active" : ""}`}
            onClick={() => setStatusFilter(s)}
          >
            {s === "all" ? "All Status" : s.charAt(0).toUpperCase() + s.slice(1)}
          </button>
        ))}
        <span className="filter-pill-divider" />
        {LOCATION_OPTIONS.map((l) => (
          <button
            key={l}
            className={`filter-pill${locationFilter === l ? " active" : ""}`}
            onClick={() => setLocationFilter(l)}
          >
            {l === "all" ? "All Locations" : l.charAt(0).toUpperCase() + l.slice(1)}
          </button>
        ))}
      </div>

      {merged.length === 0 && !isLoading ? (
        <div className="empty-state">
          <div className="empty-state-icon">
            <Activity />
          </div>
          <h3>No jobs</h3>
          <p>Launch a training run from the Training page to see it here.</p>
        </div>
      ) : (
        <div className="job-card-list">
          {isLoading && merged.length === 0 && (
            <div style={{ display: "flex", justifyContent: "center", padding: 16 }}>
              <Loader2 size={20} className="spin" />
            </div>
          )}
          {merged.map((item) => {
            if (item.kind === "unified") {
              return (
                <UnifiedJobRow
                  key={item.job.jobId}
                  job={item.job}
                  localTask={item.localTask}
                  onDelete={() => removeUnifiedJob(item.job.jobId)}
                  onCancel={() => cancelUnifiedJob(item.job.jobId).catch(console.error)}
                  onView={() => setViewingUnifiedJob({ job: item.job, localTask: item.localTask })}
                  onRefresh={refreshUnified}
                  onKill={item.localTask ? () => kill(item.localTask!.task_id) : undefined}
                  onRename={item.localTask ? (label) => rename(item.localTask!.task_id, label) : undefined}
                />
              );
            }
            return (
              <RemoteJobRow
                key={item.job.jobId}
                job={item.job}
                onDelete={() => removeRemoteJob(item.job.jobId)}
                onCancel={() => cancelRemoteJob(item.job.jobId).catch(console.error)}
                onView={() => setViewingRemoteJob(item.job)}
                onRefresh={refreshRemote}
              />
            );
          })}
        </div>
      )}
    </>
  );
}
