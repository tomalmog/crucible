import { useEffect, useMemo, useRef, useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { useJobs } from "../../hooks/useJobs";
import { useRemoteJobs } from "../../hooks/useRemoteJobs";
import { useCrucible } from "../../context/CrucibleContext";
import { CommandTaskStatus } from "../../types";
import type { RemoteJobRecord } from "../../types/remote";
import { syncRemoteJobStatus } from "../../api/remoteApi";
import { Activity, Loader2 } from "lucide-react";
import { JobResultDetail } from "./JobResultDetail";
import { RemoteJobResultDetail } from "./RemoteJobResultDetail";
import { JobRow } from "./JobRow";
import { RemoteJobRow } from "./RemoteJobRow";

type StatusFilter = "all" | "running" | "completed" | "failed";
type LocationFilter = "all" | "local" | "remote";
type TaskTypeFilter = "all" | "training" | "eval" | "sweep";

const TRAINING_COMMANDS = new Set([
  "train", "sft", "dpo-train", "rlhf-train", "lora-train", "lora-merge",
  "distill", "domain-adapt", "distributed-train", "grpo-train",
  "qlora-train", "kto-train", "orpo-train", "multimodal-train", "rlvr-train",
]);

function classifyLocalJob(command: string): TaskTypeFilter {
  if (command === "sweep") return "sweep";
  if (command === "eval") return "eval";
  if (TRAINING_COMMANDS.has(command)) return "training";
  return "training"; // default for unknown commands shown on jobs page
}

function classifyRemoteJob(job: RemoteJobRecord): TaskTypeFilter {
  if (job.isSweep) return "sweep";
  if (job.trainingMethod === "eval") return "eval";
  return "training";
}

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

export function extractCrucibleError(stderr: string): string | null {
  const lines = stderr.split("\n");
  for (let i = lines.length - 1; i >= 0; i--) {
    const match = lines[i].match(/Crucible\w+Error:\s*(.+)/);
    if (match) return match[1].trim();
  }
  return null;
}

/** Normalize a job's status to one of our filter values. */
function normalizeStatus(status: string): StatusFilter {
  if (status === "running" || status === "pending" || status === "submitting") return "running";
  if (status === "completed") return "completed";
  return "failed";
}

type UnifiedJob =
  | { kind: "local"; job: CommandTaskStatus; sortKey: number }
  | { kind: "remote"; job: RemoteJobRecord; sortKey: number };

export function JobsPage() {
  const { jobs, kill, rename, remove } = useJobs();
  const { dataRoot, refreshModels } = useCrucible();
  const { jobs: remoteJobs, isLoading: isRemoteLoading, refresh: refreshRemote, removeJob: removeRemoteJob, cancelJob: cancelRemoteJob } = useRemoteJobs(dataRoot);
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [locationFilter, setLocationFilter] = useState<LocationFilter>("all");
  const [typeFilter, setTypeFilter] = useState<TaskTypeFilter>("all");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [viewingJob, setViewingJob] = useState<CommandTaskStatus | null>(null);
  const [viewingRemoteJob, setViewingRemoteJob] = useState<RemoteJobRecord | null>(null);
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

  // Build unified sorted list: most recent first
  const unified = useMemo(() => {
    const items: UnifiedJob[] = [];

    if (locationFilter !== "remote") {
      const now = Date.now();
      for (let i = 0; i < jobs.length; i++) {
        const j = jobs[i];
        if (statusFilter !== "all" && normalizeStatus(j.status) !== statusFilter) continue;
        if (typeFilter !== "all" && classifyLocalJob(j.command) !== typeFilter) continue;
        // Local jobs lack timestamps. The array is already newest-first,
        // so assign descending keys comparable to remote timestamps.
        items.push({ kind: "local", job: j, sortKey: now - i });
      }
    }

    if (locationFilter !== "local") {
      for (const j of remoteJobs) {
        if (statusFilter !== "all" && normalizeStatus(j.state) !== statusFilter) continue;
        if (typeFilter !== "all" && classifyRemoteJob(j) !== typeFilter) continue;
        const ts = new Date(j.submittedAt).getTime() || 0;
        items.push({ kind: "remote", job: j, sortKey: ts });
      }
    }

    // Sort descending — most recent first
    items.sort((a, b) => b.sortKey - a.sortKey);
    return items;
  }, [jobs, remoteJobs, statusFilter, locationFilter, typeFilter]);

  const runningCount = jobs.filter((j) => j.status === "running").length
    + remoteJobs.filter((j) => j.state === "running" || j.state === "pending" || j.state === "submitting").length;

  function toggleExpand(taskId: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(taskId)) next.delete(taskId);
      else next.add(taskId);
      return next;
    });
  }

  if (viewingJob) {
    return <JobResultDetail job={viewingJob} onBack={() => setViewingJob(null)} />;
  }

  if (viewingRemoteJob) {
    return <RemoteJobResultDetail job={viewingRemoteJob} onBack={() => setViewingRemoteJob(null)} />;
  }

  return (
    <>
      <PageHeader title="Jobs">
        {runningCount > 0 && (
          <span className="badge badge-accent">{runningCount} running</span>
        )}
      </PageHeader>

      <div className="filter-bar">
        <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value as StatusFilter)}>
          <option value="all">All Status</option>
          <option value="running">Running</option>
          <option value="completed">Completed</option>
          <option value="failed">Failed</option>
        </select>
        <select value={locationFilter} onChange={(e) => setLocationFilter(e.target.value as LocationFilter)}>
          <option value="all">All Locations</option>
          <option value="local">Local</option>
          <option value="remote">Remote</option>
        </select>
        <select value={typeFilter} onChange={(e) => setTypeFilter(e.target.value as TaskTypeFilter)}>
          <option value="all">All Types</option>
          <option value="training">Training</option>
          <option value="eval">Eval</option>
          <option value="sweep">Sweep</option>
        </select>
      </div>

      {unified.length === 0 && !isRemoteLoading ? (
        <div className="empty-state">
          <div className="empty-state-icon">
            <Activity />
          </div>
          <h3>No jobs</h3>
          <p>Launch a training run from the Training page to see it here.</p>
        </div>
      ) : (
        <div className="panel panel-flush">
          {isRemoteLoading && unified.length === 0 && (
            <div style={{ display: "flex", justifyContent: "center", padding: 16 }}>
              <Loader2 size={20} className="spin" />
            </div>
          )}
          {unified.map((item) =>
            item.kind === "local" ? (
              <JobRow
                key={item.job.task_id}
                job={item.job}
                isExpanded={expanded.has(item.job.task_id)}
                onToggle={() => toggleExpand(item.job.task_id)}
                onKill={() => kill(item.job.task_id)}
                onRename={(label) => rename(item.job.task_id, label)}
                onDelete={() => remove(item.job.task_id)}
                onView={() => setViewingJob(item.job)}
              />
            ) : (
              <RemoteJobRow
                key={item.job.jobId}
                job={item.job}
                onDelete={() => removeRemoteJob(item.job.jobId)}
                onCancel={() => cancelRemoteJob(item.job.jobId).catch(console.error)}
                onView={() => setViewingRemoteJob(item.job)}
                onRefresh={refreshRemote}
              />
            ),
          )}
        </div>
      )}
    </>
  );
}
