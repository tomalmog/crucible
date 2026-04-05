import { useMemo, useState } from "react";
import { useLocation } from "react-router";
import { PageHeader } from "../../components/shared/PageHeader";
import { useJobs } from "../../hooks/useJobs";
import { useUnifiedJobs } from "../../hooks/useUnifiedJobs";
import { useCrucible } from "../../context/CrucibleContext";
import type { CommandTaskStatus } from "../../types";
import type { JobRecord } from "../../types/jobs";
import { Activity, Loader2 } from "lucide-react";
import { TabBar } from "../../components/shared/TabBar";
import { UnifiedJobResultDetail } from "./UnifiedJobResultDetail";
import { UnifiedJobRow } from "./UnifiedJobRow";

type StatusFilter = "all" | "running" | "completed" | "failed";
type LocationFilter = "all" | "local" | "remote";
type TaskTypeFilter = "all" | "training" | "eval" | "sweep" | "interp" | "hub" | "ingest";

const INTERP_COMMANDS = new Set([
  "logit-lens", "activation-pca", "activation-patch",
  "linear-probe", "sae-train", "sae-analyze",
  "steer-compute", "steer-apply",
]);

function classifyUnifiedJob(job: JobRecord): TaskTypeFilter {
  if (job.isSweep) return "sweep";
  if (job.jobType === "eval") return "eval";
  if (job.jobType && INTERP_COMMANDS.has(job.jobType)) return "interp";
  if (job.jobType === "hub-download") return "hub";
  if (job.jobType === "ingest") return "ingest";
  return "training";
}

const TYPE_TABS: readonly TaskTypeFilter[] = ["all", "training", "eval", "sweep", "interp", "ingest", "hub"] as const;
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
  | { kind: "unified"; job: JobRecord; localTask?: CommandTaskStatus; sortKey: number };

export function JobsPage() {
  const location = useLocation();
  const navState = location.state as { statusFilter?: StatusFilter } | null;
  if (navState) window.history.replaceState({}, "");
  const { jobs, kill, rename } = useJobs();
  const { dataRoot } = useCrucible();
  const { jobs: unifiedJobs, isLoading: isUnifiedLoading, refresh: refreshUnified, removeJob: removeUnifiedJob, cancel: cancelUnifiedJob } = useUnifiedJobs(dataRoot);
  const [statusFilter, setStatusFilter] = useState<StatusFilter>(navState?.statusFilter ?? "all");
  const [locationFilter, setLocationFilter] = useState<LocationFilter>("all");
  const [typeFilter, setTypeFilter] = useState<TaskTypeFilter>("all");
  const [viewingUnifiedJob, setViewingUnifiedJob] = useState<{ job: JobRecord; localTask?: CommandTaskStatus } | null>(null);

  // Build a lookup of local task_id → CommandTaskStatus for unified local jobs
  const localTaskMap = useMemo(() => {
    const map = new Map<string, CommandTaskStatus>();
    for (const j of jobs) map.set(j.task_id, j);
    return map;
  }, [jobs]);

  // Build merged sorted list
  const merged = useMemo(() => {
    const items: MergedJob[] = [];

    // Unified jobs (from .crucible/jobs/)
    for (const j of unifiedJobs) {
      const isLocal = j.backend === "local";
      const localTask = isLocal ? localTaskMap.get(j.jobId) : undefined;
      // Orphaned local job: state is "running" but no in-memory task (app was restarted)
      const job = (isLocal && j.state === "running" && !localTask)
        ? { ...j, state: "failed" as const, errorMessage: j.errorMessage || "Process lost — app was restarted while job was running." }
        : j;
      if (statusFilter !== "all" && normalizeStatus(job.state) !== statusFilter) continue;
      const isRemote = !isLocal;
      if (locationFilter === "local" && isRemote) continue;
      if (locationFilter === "remote" && !isRemote) continue;
      if (typeFilter !== "all" && classifyUnifiedJob(job) !== typeFilter) continue;
      const ts = new Date(job.createdAt).getTime() || 0;
      items.push({ kind: "unified", job, localTask, sortKey: ts });
    }

    items.sort((a, b) => b.sortKey - a.sortKey);
    return items;
  }, [unifiedJobs, statusFilter, locationFilter, typeFilter, localTaskMap]);

  const runningCount = jobs.filter((j) => j.status === "running").length
    + unifiedJobs.filter((j) => j.state === "running" || j.state === "pending" || j.state === "submitting").length;

  if (viewingUnifiedJob) {
    return (
      <UnifiedJobResultDetail
        job={viewingUnifiedJob.job}
        localTask={viewingUnifiedJob.localTask}
        onBack={() => setViewingUnifiedJob(null)}
      />
    );
  }

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

      {merged.length === 0 && !isUnifiedLoading ? (
        <div className="empty-state">
          <div className="empty-state-icon">
            <Activity />
          </div>
          <h3>No jobs</h3>
          <p>Launch a training run from the Training page to see it here.</p>
        </div>
      ) : (
        <div className="job-card-list">
          {isUnifiedLoading && merged.length === 0 && (
            <div style={{ display: "flex", justifyContent: "center", padding: 16 }}>
              <Loader2 size={20} className="spin" />
            </div>
          )}
          {merged.map((item) => (
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
          ))}
        </div>
      )}
    </>
  );
}
