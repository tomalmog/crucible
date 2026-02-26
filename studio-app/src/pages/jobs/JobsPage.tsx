import { useMemo, useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { useJobs } from "../../hooks/useJobs";
import { CommandTaskStatus } from "../../types";
import { parseTrainingProgress } from "../training/TrainingRunMonitor";
import {
  Activity,
  Square,
  ChevronDown,
  ChevronRight,
  Pencil,
  Trash2,
  Check,
  X,
} from "lucide-react";

type Filter = "all" | "running" | "completed" | "failed";

const FILTERS: { key: Filter; label: string }[] = [
  { key: "all", label: "All" },
  { key: "running", label: "Running" },
  { key: "completed", label: "Completed" },
  { key: "failed", label: "Failed" },
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
  const [filter, setFilter] = useState<Filter>("all");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const filtered = filter === "all" ? jobs : jobs.filter((j) => j.status === filter);

  function toggleExpand(taskId: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(taskId)) next.delete(taskId);
      else next.add(taskId);
      return next;
    });
  }

  const runningCount = jobs.filter((j) => j.status === "running").length;

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

      {filtered.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">
            <Activity />
          </div>
          <h3>No {filter === "all" ? "" : filter} jobs</h3>
          <p>Launch a training run from the Training page to see it here.</p>
        </div>
      ) : (
        <div className="panel" style={{ padding: 0, overflow: "hidden" }}>
          {filtered.map((job) => (
            <JobRow
              key={job.task_id}
              job={job}
              isExpanded={expanded.has(job.task_id)}
              onToggle={() => toggleExpand(job.task_id)}
              onKill={() => kill(job.task_id)}
              onRename={(label) => rename(job.task_id, label)}
              onDelete={() => remove(job.task_id)}
            />
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
}: {
  job: CommandTaskStatus;
  isExpanded: boolean;
  onToggle: () => void;
  onKill: () => void;
  onRename: (label: string) => void;
  onDelete: () => void;
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
    <div className="run-row" style={{ borderBottom: "1px solid var(--border)" }}>
      <div className="run-row-header">
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <button className="btn btn-ghost btn-sm" onClick={onToggle} style={{ padding: 2 }}>
            {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </button>
          {editing ? (
            <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <input
                autoFocus
                value={draft}
                onChange={(e) => setDraft(e.currentTarget.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") confirmRename();
                  if (e.key === "Escape") cancelRename();
                }}
                placeholder={job.task_id}
                style={{ width: 180, padding: "2px 6px", fontSize: "0.75rem" }}
              />
              <button className="btn btn-ghost btn-sm" onClick={confirmRename} style={{ padding: 2 }}>
                <Check size={12} />
              </button>
              <button className="btn btn-ghost btn-sm" onClick={cancelRename} style={{ padding: 2 }}>
                <X size={12} />
              </button>
            </div>
          ) : (
            <>
              <span className="run-row-id">{displayName}</span>
              <button
                className="btn btn-ghost btn-sm"
                onClick={startEditing}
                title="Rename"
                style={{ padding: 2 }}
              >
                <Pencil size={11} />
              </button>
            </>
          )}
          <span className={statusBadgeClass(job.status)}>{job.status}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span className="run-row-meta">
            {isFinished ? `took ${formatElapsed(job.elapsed_seconds)}` : formatElapsed(job.elapsed_seconds)}
          </span>
          {job.status === "running" && (
            <button className="btn btn-sm" onClick={onKill} title="Kill process">
              <Square size={12} /> Kill
            </button>
          )}
          {isFinished && (
            <button className="btn btn-ghost btn-sm" onClick={onDelete} title="Delete job">
              <Trash2 size={12} />
            </button>
          )}
        </div>
      </div>

      <div className="run-row-path">{commandLabel}</div>

      {job.status === "running" && progress && (
        <div style={{
          display: "flex",
          gap: 16,
          fontSize: "0.75rem",
          color: "var(--text-secondary)",
          marginTop: 4,
        }}>
          <span>Epoch {progress.epoch}/{progress.totalEpochs}</span>
          {progress.loss != null && <span>Loss: {progress.loss.toFixed(4)}</span>}
          {progress.meanReward != null && <span>Reward: {progress.meanReward.toFixed(4)}</span>}
        </div>
      )}

      {job.status === "running" && (
        <div className="progress-bar" style={{ marginTop: 6 }}>
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
        <div style={{ marginTop: 8, display: "grid", gap: 8 }}>
          {job.stdout && (
            <div>
              <div style={{ fontSize: "0.6875rem", color: "var(--text-tertiary)", marginBottom: 4 }}>
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
                  <div style={{
                    padding: "8px 12px",
                    marginBottom: 8,
                    borderRadius: 6,
                    background: "color-mix(in srgb, var(--error) 12%, transparent)",
                    border: "1px solid color-mix(in srgb, var(--error) 30%, transparent)",
                    color: "var(--error)",
                    fontSize: "0.8125rem",
                    fontWeight: 500,
                  }}>
                    {friendly}
                  </div>
                ) : null;
              })()}
              <details open={job.status !== "failed"}>
                <summary style={{
                  fontSize: "0.6875rem",
                  color: job.status === "failed" ? "var(--error)" : "var(--text-tertiary)",
                  marginBottom: 4,
                  cursor: "pointer",
                  userSelect: "none",
                }}>
                  {job.status === "failed" ? "full traceback" : "logs"}
                </summary>
                <pre className="console console-short">{job.stderr}</pre>
              </details>
            </div>
          )}
          {!job.stdout && !job.stderr && (
            <div style={{ fontSize: "0.75rem", color: "var(--text-tertiary)", padding: "8px 0" }}>
              No output yet.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
