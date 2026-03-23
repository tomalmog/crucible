import { useMemo, useState } from "react";
import { CommandTaskStatus } from "../../types";
import { parseTrainingProgress } from "../training/TrainingRunMonitor";
import { formatDuration } from "../../components/shared/formatDuration";
import { jobAccentColor, extractCrucibleError } from "./JobsPage";
import { formatTimeAgo } from "../../utils/formatTime";
import {
  Square,
  ChevronRight,
  Pencil,
  Trash2,
  Check,
  X,
} from "lucide-react";

export function JobRow({
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
    <div
      className="job-card"
      style={{ "--job-accent": jobAccentColor(job.status) } as React.CSSProperties}
      onClick={() => {
        if (editing) return;
        if (isFinished) onView();
        else onToggle();
      }}
    >
      {/* Line 1: status dot + name | secondary actions + status */}
      <div className="run-row-header">
        <div className="flex-row">
          <span className={"job-status-dot" + (job.status === "running" ? " pulsing" : "")} />
          {editing ? (
            <div className="flex-row-tight" onClick={(e) => e.stopPropagation()}>
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
                onClick={(e) => { e.stopPropagation(); startEditing(); }}
                title="Rename"
              >
                <Pencil size={11} />
              </button>
            </>
          )}
        </div>
        <div className="flex-row">
          {job.status === "running" && (
            <button className="btn btn-ghost btn-sm btn-icon" onClick={(e) => { e.stopPropagation(); onKill(); }} title="Kill process">
              <Square size={12} />
            </button>
          )}
          {isFinished && (
            <button className="btn btn-ghost btn-sm btn-icon" onClick={(e) => { e.stopPropagation(); onDelete(); }} title="Delete job">
              <Trash2 size={12} />
            </button>
          )}
          <span className="run-row-meta">{job.status}</span>
          <ChevronRight size={14} className="job-card-chevron" />
        </div>
      </div>

      {/* Line 2: meta — command label + elapsed time + timestamp */}
      <div className="job-card-meta">
        <span>{commandLabel}</span>
        <span>
          {isFinished ? `took ${formatDuration(job.elapsed_seconds)} · ` : `${formatDuration(job.elapsed_seconds)} · `}
          {formatTimeAgo(Date.now() - job.elapsed_seconds * 1000)}
        </span>
      </div>

      {/* Progress strip (running jobs) */}
      {job.status === "running" && (
        <>
          <div className="job-progress-strip">
            <div className="job-progress-strip-fill" style={{ width: `${job.progress_percent}%` }} />
          </div>
          <div className="job-card-meta">
            <span>{job.progress_percent.toFixed(0)}% · Elapsed {formatDuration(job.elapsed_seconds)}</span>
            <span>~{formatDuration(job.remaining_seconds)} remaining</span>
          </div>
        </>
      )}

      {/* Inline metrics */}
      {job.status === "running" && progress && (
        <div className="job-progress-meta">
          <span>Epoch {progress.epoch}/{progress.totalEpochs}</span>
          {progress.loss != null && <span>Loss: {progress.loss.toFixed(4)}</span>}
          {progress.meanReward != null && <span>Reward: {progress.meanReward.toFixed(4)}</span>}
        </div>
      )}

      {/* Expanded output */}
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
                const friendly = extractCrucibleError(job.stderr);
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
