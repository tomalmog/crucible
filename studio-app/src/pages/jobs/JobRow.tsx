import { useMemo, useState } from "react";
import { CommandTaskStatus } from "../../types";
import { parseTrainingProgress } from "../training/TrainingRunMonitor";
import { formatDuration } from "../../components/shared/formatDuration";
import { statusBadgeClass, extractForgeError } from "./JobsPage";
import {
  Square,
  ChevronDown,
  ChevronRight,
  Eye,
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
            {isFinished ? `took ${formatDuration(job.elapsed_seconds)}` : formatDuration(job.elapsed_seconds)}
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
            <span>Elapsed {formatDuration(job.elapsed_seconds)}</span>
            <span>~{formatDuration(job.remaining_seconds)} remaining</span>
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
