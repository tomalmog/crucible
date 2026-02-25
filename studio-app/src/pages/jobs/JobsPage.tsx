import { useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { useJobs } from "../../hooks/useJobs";
import { CommandTaskStatus } from "../../types";
import { Activity, Square, ChevronDown, ChevronRight } from "lucide-react";

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
  const { jobs, kill } = useJobs();
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
          <p>Launch a training run or command from the Training page to see it here.</p>
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
            />
          ))}
        </div>
      )}
    </>
  );
}

function JobRow({
  job,
  isExpanded,
  onToggle,
  onKill,
}: {
  job: CommandTaskStatus;
  isExpanded: boolean;
  onToggle: () => void;
  onKill: () => void;
}) {
  const commandLabel = [job.command, ...job.args.slice(1)].join(" ");

  return (
    <div className="run-row" style={{ borderBottom: "1px solid var(--border)" }}>
      <div className="run-row-header">
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <button className="btn btn-ghost btn-sm" onClick={onToggle} style={{ padding: 2 }}>
            {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </button>
          <span className="run-row-id">{job.task_id}</span>
          <span className={statusBadgeClass(job.status)}>{job.status}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span className="run-row-meta">{formatElapsed(job.elapsed_seconds)}</span>
          {job.status === "running" && (
            <button className="btn btn-sm" onClick={onKill} title="Kill process">
              <Square size={12} /> Kill
            </button>
          )}
        </div>
      </div>

      <div className="run-row-path">{commandLabel}</div>

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
              <div style={{ fontSize: "0.6875rem", color: "var(--error)", marginBottom: 4 }}>
                stderr
              </div>
              <pre className="console console-short">{job.stderr}</pre>
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
