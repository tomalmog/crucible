import { useEffect, useRef, useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { useJobs } from "../../hooks/useJobs";
import { useRemoteJobs } from "../../hooks/useRemoteJobs";
import { useForge } from "../../context/ForgeContext";
import { CommandTaskStatus } from "../../types";
import { syncRemoteJobStatus } from "../../api/remoteApi";
import { Activity } from "lucide-react";
import { JobResultDetail } from "./JobResultDetail";
import { JobRow } from "./JobRow";
import { RemoteJobRow } from "./RemoteJobRow";

type Filter = "all" | "running" | "completed" | "failed" | "remote";

const FILTERS: { key: Filter; label: string }[] = [
  { key: "all", label: "All" },
  { key: "running", label: "Running" },
  { key: "completed", label: "Completed" },
  { key: "failed", label: "Failed" },
  { key: "remote", label: "Remote" },
];

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

export function extractForgeError(stderr: string): string | null {
  const lines = stderr.split("\n");
  for (let i = lines.length - 1; i >= 0; i--) {
    const match = lines[i].match(/Forge\w+Error:\s*(.+)/);
    if (match) return match[1].trim();
  }
  return null;
}

export function JobsPage() {
  const { jobs, kill, rename, remove } = useJobs();
  const { dataRoot, refreshModels } = useForge();
  const { jobs: remoteJobs, refresh: refreshRemote, removeJob: removeRemoteJob, cancelJob: cancelRemoteJob } = useRemoteJobs(dataRoot);
  const [filter, setFilter] = useState<Filter>("all");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [viewingJob, setViewingJob] = useState<CommandTaskStatus | null>(null);
  const syncedRef = useRef<Set<string>>(new Set());
  const remoteJobsRef = useRef(remoteJobs);
  remoteJobsRef.current = remoteJobs;

  // Auto-sync running remote job states via the CLI (bypasses task store).
  // Runs once on mount + every 30s. Uses a ref so the interval always
  // reads the latest remoteJobs list instead of a stale closure.
  useEffect(() => {
    if (!dataRoot) return;
    const syncRunning = () => {
      const now = Date.now();
      const running = remoteJobsRef.current.filter(
        (j) =>
          ((j.state === "running" || j.state === "pending") ||
           // Keep syncing recently-completed jobs without a model path
           // so model discovery retries on NFS propagation delay.
           // Stop after 2 minutes to avoid infinite polling.
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

  const filtered = filter === "remote"
    ? []
    : filter === "all" ? jobs : jobs.filter((j) => j.status === filter);
  const filteredRemote = filter === "all" || filter === "remote"
    ? remoteJobs
    : remoteJobs.filter((j) => j.state === filter);

  function toggleExpand(taskId: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(taskId)) next.delete(taskId);
      else next.add(taskId);
      return next;
    });
  }

  const runningCount = jobs.filter((j) => j.status === "running").length;

  if (viewingJob) {
    return <JobResultDetail job={viewingJob} onBack={() => setViewingJob(null)} />;
  }

  return (
    <>
      <PageHeader title="Jobs">
        {runningCount > 0 && (
          <span className="badge badge-accent">{runningCount} running</span>
        )}
      </PageHeader>

      <div className="tab-list">
        {FILTERS.map((f) => (
          <button
            key={f.key}
            className={`tab-item ${filter === f.key ? "active" : ""}`}
            onClick={() => setFilter(f.key)}
          >
            {f.label}
          </button>
        ))}
      </div>

      {filtered.length === 0 && filteredRemote.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">
            <Activity />
          </div>
          <h3>No {filter === "all" ? "" : filter} jobs</h3>
          <p>Launch a training run from the Training page to see it here.</p>
        </div>
      ) : (
        <div className="panel panel-flush">
          {filtered.map((job) => (
            <JobRow
              key={job.task_id}
              job={job}
              isExpanded={expanded.has(job.task_id)}
              onToggle={() => toggleExpand(job.task_id)}
              onKill={() => kill(job.task_id)}
              onRename={(label) => rename(job.task_id, label)}
              onDelete={() => remove(job.task_id)}
              onView={() => setViewingJob(job)}
            />
          ))}
          {filteredRemote.map((rj) => (
            <RemoteJobRow
              key={rj.jobId}
              job={rj}
              onDelete={() => removeRemoteJob(rj.jobId)}
              onCancel={() => cancelRemoteJob(rj.jobId).catch(console.error)}
            />
          ))}
        </div>
      )}
    </>
  );
}
