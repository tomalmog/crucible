import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { useJobs } from "../../hooks/useJobs";
import { useRemoteJobs } from "../../hooks/useRemoteJobs";
import { useForge } from "../../context/ForgeContext";
import { CommandTaskStatus } from "../../types";
import type { RemoteJobRecord } from "../../types/remote";
import { parseTrainingProgress } from "../training/TrainingRunMonitor";
import { startForgeCommand, getForgeCommandStatus } from "../../api/studioApi";
import { getRemoteJobLogs } from "../../api/remoteApi";
import {
  Activity,
  Square,
  ChevronDown,
  ChevronRight,
  Eye,
  Pencil,
  Trash2,
  Check,
  X,
  Server,
  Terminal,
  RefreshCw,
  Download,
} from "lucide-react";
import { JobResultDetail } from "./JobResultDetail";

type Filter = "all" | "running" | "completed" | "failed" | "remote";

const FILTERS: { key: Filter; label: string }[] = [
  { key: "all", label: "All" },
  { key: "running", label: "Running" },
  { key: "completed", label: "Completed" },
  { key: "failed", label: "Failed" },
  { key: "remote", label: "Remote" },
];

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

function statusBadgeClass(status: string): string {
  switch (status) {
    case "running":
      return "badge badge-accent";
    case "completed":
      return "badge badge-success";
    case "failed":
      return "badge badge-error";
    default:
      return "badge";
  }
}

export function JobsPage() {
  const { jobs, kill, rename, remove } = useJobs();
  const { dataRoot, refreshModels } = useForge();
  const { jobs: remoteJobs, refresh: refreshRemote, removeJob: removeRemoteJob } = useRemoteJobs(dataRoot);
  const [filter, setFilter] = useState<Filter>("all");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [viewingJob, setViewingJob] = useState<CommandTaskStatus | null>(null);
  const syncedRef = useRef<Set<string>>(new Set());

  // Auto-sync running remote job states via the CLI
  useEffect(() => {
    if (!dataRoot) return;
    const running = remoteJobs.filter(
      (j) => (j.state === "running" || j.state === "pending") && !syncedRef.current.has(j.jobId),
    );
    for (const rj of running) {
      syncedRef.current.add(rj.jobId);
      startForgeCommand(dataRoot, ["remote", "status", "--job-id", rj.jobId])
        .then((task) => {
          // Poll until complete, then refresh the list
          const check = async () => {
            for (let i = 0; i < 30; i++) {
              await new Promise((r) => setTimeout(r, 1000));
              const s = await getForgeCommandStatus(task.task_id);
              if (s.status !== "running") {
                syncedRef.current.delete(rj.jobId);
                refreshRemote();
                // Model may have been auto-registered on completion
                refreshModels().catch(console.error);
                return;
              }
            }
            syncedRef.current.delete(rj.jobId);
          };
          check();
        })
        .catch(() => syncedRef.current.delete(rj.jobId));
    }
  }, [dataRoot, remoteJobs, refreshRemote, refreshModels]);

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
            <RemoteJobRow key={rj.jobId} job={rj} onDelete={() => removeRemoteJob(rj.jobId)} />
          ))}
        </div>
      )}
    </>
  );
}

function extractForgeError(stderr: string): string | null {
  const lines = stderr.split("\n");
  for (let i = lines.length - 1; i >= 0; i--) {
    const match = lines[i].match(/Forge\w+Error:\s*(.+)/);
    if (match) return match[1].trim();
  }
  return null;
}

function JobRow({
  job,
  isExpanded,
  onToggle,
  onKill,
  onRename,
  onDelete,
  onView,
}: {
  job: CommandTaskStatus;
  isExpanded: boolean;
  onToggle: () => void;
  onKill: () => void;
  onRename: (label: string) => void;
  onDelete: () => void;
  onView: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const commandLabel = [job.command, ...job.args.slice(1)].join(" ");
  const displayName = job.label || job.task_id;
  const isFinished = job.status !== "running";
  const progress = useMemo(
    () => (job.stdout ? parseTrainingProgress(job.stdout) : null),
    [job.stdout],
  );

  function startEditing() {
    setDraft(job.label || "");
    setEditing(true);
  }

  function confirmRename() {
    onRename(draft.trim());
    setEditing(false);
  }

  function cancelRename() {
    setEditing(false);
  }

  return (
    <div className="run-row section-divider">
      <div className="run-row-header">
        <div className="flex-row">
          <button className="btn btn-ghost btn-sm btn-icon" onClick={onToggle}>
            {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </button>
          {editing ? (
            <div className="flex-row-tight">
              <input
                autoFocus
                className="job-inline-input"
                value={draft}
                onChange={(e) => setDraft(e.currentTarget.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") confirmRename();
                  if (e.key === "Escape") cancelRename();
                }}
                placeholder={job.task_id}
              />
              <button className="btn btn-ghost btn-sm btn-icon" onClick={confirmRename}>
                <Check size={12} />
              </button>
              <button className="btn btn-ghost btn-sm btn-icon" onClick={cancelRename}>
                <X size={12} />
              </button>
            </div>
          ) : (
            <>
              <span className="run-row-id">{displayName}</span>
              <button
                className="btn btn-ghost btn-sm btn-icon"
                onClick={startEditing}
                title="Rename"
              >
                <Pencil size={11} />
              </button>
            </>
          )}
          <span className={statusBadgeClass(job.status)}>{job.status}</span>
        </div>
        <div className="flex-row">
          <span className="run-row-meta">
            {isFinished ? `took ${formatElapsed(job.elapsed_seconds)}` : formatElapsed(job.elapsed_seconds)}
          </span>
          {job.status === "running" && (
            <button className="btn btn-sm" onClick={onKill} title="Kill process">
              <Square size={12} /> Kill
            </button>
          )}
          {isFinished && (
            <>
              <button className="btn btn-sm" onClick={onView} title="View result">
                <Eye size={12} /> Result
              </button>
              <button className="btn btn-ghost btn-sm" onClick={onDelete} title="Delete job">
                <Trash2 size={12} />
              </button>
            </>
          )}
        </div>
      </div>

      <div className="run-row-path">{commandLabel}</div>

      {job.status === "running" && progress && (
        <div className="job-progress-meta">
          <span>Epoch {progress.epoch}/{progress.totalEpochs}</span>
          {progress.loss != null && <span>Loss: {progress.loss.toFixed(4)}</span>}
          {progress.meanReward != null && <span>Reward: {progress.meanReward.toFixed(4)}</span>}
        </div>
      )}

      {job.status === "running" && (
        <div className="progress-bar gap-top-sm">
          <div className="progress-bar-header">
            <span className="progress-label">Progress</span>
            <span className="progress-value">{job.progress_percent.toFixed(0)}%</span>
          </div>
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{ width: `${job.progress_percent}%` }}
            />
          </div>
          <div className="progress-bar-footer">
            <span>Elapsed {formatElapsed(job.elapsed_seconds)}</span>
            <span>~{formatElapsed(job.remaining_seconds)} remaining</span>
          </div>
        </div>
      )}

      {isExpanded && (
        <div className="job-expanded">
          {job.stdout && (
            <div>
              <div className="job-output-label">
                stdout
              </div>
              <pre className="console console-short">{job.stdout}</pre>
            </div>
          )}
          {job.stderr && (
            <div>
              {job.status === "failed" && (() => {
                const friendly = extractForgeError(job.stderr);
                return friendly ? (
                  <div className="error-alert-prominent">
                    {friendly}
                  </div>
                ) : null;
              })()}
              <details open={job.status !== "failed"}>
                <summary className={`job-traceback-toggle ${job.status === "failed" ? "error-text" : ""}`}>
                  {job.status === "failed" ? "full traceback" : "logs"}
                </summary>
                <pre className="console console-short">{job.stderr}</pre>
              </details>
            </div>
          )}
          {!job.stdout && !job.stderr && (
            <div className="job-no-output">
              No output yet.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function RemoteJobRow({ job, onDelete }: { job: RemoteJobRecord; onDelete: () => void }) {
  const { dataRoot } = useForge();
  const sweepTag = job.isSweep ? ` (sweep, ${job.sweepArraySize} trials)` : "";
  const [showLogs, setShowLogs] = useState(false);
  const [logs, setLogs] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const logEndRef = useRef<HTMLDivElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchLogs = useCallback(async () => {
    if (!dataRoot) return;
    setLoading(true);
    try {
      const content = await getRemoteJobLogs(dataRoot, job.jobId);
      setLogs(content || "No logs available yet.");
    } catch (err) {
      setLogs(`Error fetching logs: ${err}`);
    } finally {
      setLoading(false);
    }
  }, [dataRoot, job.jobId]);

  const toggleLogs = useCallback(() => {
    const next = !showLogs;
    setShowLogs(next);
    if (next && !logs) {
      fetchLogs();
    }
  }, [showLogs, logs, fetchLogs]);

  // Auto-refresh logs for running jobs
  useEffect(() => {
    if (showLogs && job.state === "running") {
      pollRef.current = setInterval(fetchLogs, 10_000);
      return () => {
        if (pollRef.current) clearInterval(pollRef.current);
      };
    }
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [showLogs, job.state, fetchLogs]);

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const [pullProgress, setPullProgress] = useState<string[]>([]);
  const [pullDone, setPullDone] = useState(false);
  const [pullError, setPullError] = useState<string | null>(null);
  const [pulling, setPulling] = useState(false);

  const handlePull = useCallback(async () => {
    if (!dataRoot) return;
    setPulling(true);
    setPullProgress([]);
    setPullError(null);
    setPullDone(false);
    try {
      const pullArgs = [
        "remote", "pull-model", "--job-id", job.jobId,
      ];
      if (job.modelName) {
        pullArgs.push("--model-name", job.modelName);
      }
      const task = await startForgeCommand(dataRoot, pullArgs);
      const poll = setInterval(async () => {
        try {
          const status = await getForgeCommandStatus(task.task_id);
          const lines = (status.stdout || "")
            .split("\n")
            .filter((l: string) => l.startsWith("FORGE_PULL_PROGRESS: "))
            .map((l: string) => l.replace("FORGE_PULL_PROGRESS: ", ""));
          if (lines.length > 0) setPullProgress(lines);
          if (status.status !== "running") {
            clearInterval(poll);
            if (status.status === "completed") {
              setPullDone(true);
            } else {
              setPullError(status.stderr || "Pull failed");
            }
            setPulling(false);
          }
        } catch {
          clearInterval(poll);
          setPulling(false);
          setPullError("Lost connection to pull task");
        }
      }, 2000);
    } catch (err) {
      setPulling(false);
      setPullError(`Failed to start pull: ${err}`);
    }
  }, [dataRoot, job.jobId, job.trainingMethod]);

  const isRunning = job.state === "running" || job.state === "pending";
  const isCompleted = job.state === "completed";
  const hasLocalModel = !!job.modelPathLocal;

  return (
    <div className="run-row section-divider">
      <div className="run-row-header">
        <div className="flex-row">
          <button className="btn btn-ghost btn-sm btn-icon" onClick={toggleLogs}>
            {showLogs ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </button>
          <span className="run-row-id">{job.modelName || job.jobId}</span>
          {job.modelName && (
            <span className="run-row-meta" style={{ opacity: 0.6 }}>{job.jobId}</span>
          )}
          <span className="badge"><Server size={10} /> {job.clusterName}</span>
          <span className={statusBadgeClass(job.state)}>{job.state}</span>
          {isRunning && (
            <span className="badge badge-accent" style={{ fontSize: "0.7rem" }}>
              <Activity size={10} /> live
            </span>
          )}
        </div>
        <div className="flex-row">
          <span className="run-row-meta">Slurm {job.slurmJobId}</span>
          <button
            className="btn btn-sm"
            onClick={(e) => { e.stopPropagation(); toggleLogs(); }}
            title="View logs"
          >
            <Terminal size={12} /> Logs
          </button>
          {isCompleted && !hasLocalModel && !pulling && !pullDone && (
            <button
              className="btn btn-sm"
              onClick={(e) => { e.stopPropagation(); handlePull(); }}
              title="Download model to local machine"
            >
              <Download size={12} /> Pull Model
            </button>
          )}
          {showLogs && (
            <button
              className="btn btn-ghost btn-sm"
              onClick={(e) => { e.stopPropagation(); fetchLogs(); }}
              title="Refresh logs"
            >
              <RefreshCw size={12} />
            </button>
          )}
          {!isRunning && (
            <button
              className="btn btn-ghost btn-sm"
              onClick={(e) => { e.stopPropagation(); onDelete(); }}
              title="Delete job"
            >
              <Trash2 size={12} />
            </button>
          )}
        </div>
      </div>
      <div className="run-row-path">
        {job.trainingMethod}{sweepTag}
      </div>
      {job.submittedAt && (
        <div className="run-row-path">Submitted: {job.submittedAt}</div>
      )}
      {job.modelPathRemote && (
        <div className="run-row-path" style={{ opacity: 0.7, fontSize: "0.8rem" }}>
          Remote model: {job.modelPathRemote}
        </div>
      )}
      {(pulling || pullDone || pullError) && (
        <div className="gap-top-sm" style={{ padding: "0 var(--space-md)" }}>
          {pulling && (
            <div className="pull-steps">
              {pullProgress.map((step, i) => (
                <div key={i} className="pull-step">{step}</div>
              ))}
              {pullProgress.length === 0 && <div className="pull-step">Starting pull...</div>}
            </div>
          )}
          {pullDone && (
            <div className="pull-success flex-row" style={{ gap: "var(--space-xs)", color: "var(--clr-success)" }}>
              <Check size={14} /> Model pulled successfully!
            </div>
          )}
          {pullError && (
            <div className="error-alert-prominent">{pullError}</div>
          )}
        </div>
      )}
      {showLogs && (
        <div className="job-expanded">
          {loading && !logs && (
            <div className="job-no-output">Fetching logs from cluster...</div>
          )}
          {logs ? (
            <div>
              <div className="job-output-label">
                Remote Logs {loading && "(refreshing...)"}
              </div>
              <pre className="console console-short" style={{ maxHeight: "400px", overflow: "auto" }}>
                {logs}
                <div ref={logEndRef} />
              </pre>
            </div>
          ) : (
            !loading && <div className="job-no-output">No logs available yet.</div>
          )}
        </div>
      )}
    </div>
  );
}
