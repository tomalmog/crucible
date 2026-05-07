import { useMemo, useState } from "react";
import type { CSSProperties } from "react";
import { useCrucible } from "../../context/CrucibleContext";
import type { CommandTaskStatus } from "../../types";
import type { JobRecord } from "../../types/jobs";
import { TERMINAL_JOB_STATES } from "../../types/jobs";
import { jobLabel } from "../../utils/jobLabels";
import { formatTimeAgo } from "../../utils/formatTime";
import { jobAccentColor } from "./JobsPage";
import { parseTrainingProgress } from "../training/TrainingRunMonitor";
import {
  ACTIVE_STATES,
  configString,
  NON_TRAINING_TYPES,
  runTypeLabel,
} from "./jobRowDisplay";
import {
  LocalJobOutputPanel,
  LocalProgressPanel,
  PullProgressPanel,
  RemoteJobLogPanel,
} from "./UnifiedJobOutputPanels";
import { useRemoteModelPull } from "./useRemoteModelPull";
import { useUnifiedJobLogs } from "./useUnifiedJobLogs";
import {
  ChevronRight,
  Square,
  Trash2,
  Check,
  RefreshCw,
  Download,
  XCircle,
  Pencil,
  X,
  Loader2,
} from "lucide-react";

export function UnifiedJobRow({
  job,
  localTask,
  onDelete,
  onCancel,
  onView,
  onRefresh,
  onKill,
  onRename,
}: {
  job: JobRecord;
  /** For local jobs, the CommandTaskStatus for stdout/stderr streaming. */
  localTask?: CommandTaskStatus;
  onDelete: () => void;
  onCancel: () => void;
  onView: () => void;
  onRefresh?: () => void;
  onKill?: () => void;
  onRename?: (label: string) => void;
}) {
  const { dataRoot, refreshModels } = useCrucible();
  const isLocal = job.backend === "local";
  const isRemote = !isLocal;
  const sweepTag = job.isSweep ? ` (sweep, ${job.sweepTrialCount} trials)` : "";

  // Inline rename state (for local jobs)
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");

  const displayName = job.label || jobLabel(job.jobType, job.modelName) || job.jobId;
  const projectName = configString(job.config, "projectName");
  const evalObjective = configString(job.config, "evalObjective");
  const promotionStage = configString(job.config, "promotionStage") || "dev";
  const isFinished = TERMINAL_JOB_STATES.has(job.state);
  const isRunning = ACTIVE_STATES.has(job.state) || job.state === "submitting";
  const isSubmitting = job.state === "submitting";
  const isFailed = job.state === "failed";
  const isCompleted = job.state === "completed";
  const hasLocalModel = !!job.modelPathLocal;
  const failedOnCluster = isFailed && isRemote && !!job.backendJobId;
  const jobCardStyle: CSSProperties & { "--job-accent": string } = {
    "--job-accent": jobAccentColor(job.state),
  };

  const {
    fetchLogs,
    handleLogScroll,
    loading,
    localLogRef,
    logContainerRef,
    logs,
    showLogs,
    toggleLogs,
  } = useUnifiedJobLogs({
    dataRoot,
    failedOnCluster,
    isLocal,
    isRemote,
    isRunning,
    isSubmitting,
    job,
    localTask,
    onRefresh,
  });

  const {
    handlePull,
    pullDone,
    pullError,
    pulling,
    pullProgress,
  } = useRemoteModelPull({ dataRoot, job, refreshModels });

  // --- Inline rename (local only) ---
  function startEditing() {
    setDraft(job.label || "");
    setEditing(true);
  }

  function confirmRename() {
    onRename?.(draft.trim());
    setEditing(false);
  }

  function cancelRename() {
    setEditing(false);
  }

  // --- Expand/collapse for local jobs ---
  const [localExpanded, setLocalExpanded] = useState(false);

  // Parse training metrics from local stdout
  const progress = useMemo(
    () => (isLocal && localTask?.stdout ? parseTrainingProgress(localTask.stdout) : null),
    [isLocal, localTask?.stdout],
  );

  return (
    <div
      className="job-card"
      style={jobCardStyle}
      onClick={() => {
        if (editing) return;
        if (isFinished) onView();
        else if (isLocal && localTask) setLocalExpanded((p) => !p);
        else if (isLocal && !localTask) onView(); // orphaned local job — view persisted data
        else toggleLogs();
      }}
    >
      {/* Line 1: status dot + name | secondary actions + status */}
      <div className="run-row-header">
        <div className="flex-row">
          {job.state === "pending" ? (
            <Loader2 size={10} className="spin" style={{ color: "var(--text-tertiary)", marginRight: 6 }} />
          ) : (
            <span className={"job-status-dot" + (isRunning ? " pulsing" : "")} />
          )}
          {isLocal && editing ? (
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
                placeholder={job.jobId}
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
              {isLocal && onRename && (
                <button className="btn btn-ghost btn-sm btn-icon" onClick={(e) => { e.stopPropagation(); startEditing(); }} title="Rename">
                  <Pencil size={11} />
                </button>
              )}
            </>
          )}
        </div>
        <div className="flex-row">
          {isLocal && job.state === "running" && onKill && (
            <button className="btn btn-ghost btn-sm btn-icon" onClick={(e) => { e.stopPropagation(); onKill(); }} title="Kill process">
              <Square size={12} />
            </button>
          )}
          {isRemote && isCompleted && !hasLocalModel && !pulling && !pullDone && !NON_TRAINING_TYPES.has(job.jobType) && (
            <button className="btn btn-ghost btn-sm btn-icon" onClick={(e) => { e.stopPropagation(); handlePull().catch(console.error); }} title="Pull model">
              <Download size={12} />
            </button>
          )}
          {isRemote && ACTIVE_STATES.has(job.state) && (
            <button className="btn btn-ghost btn-sm btn-icon" onClick={(e) => { e.stopPropagation(); onCancel(); }} title="Cancel job">
              <XCircle size={12} />
            </button>
          )}
          {isRemote && showLogs && (
            <button className="btn btn-ghost btn-sm btn-icon" onClick={(e) => { e.stopPropagation(); fetchLogs(true).catch(console.error); }} title="Refresh logs">
              <RefreshCw size={12} />
            </button>
          )}
          {isFinished && (
            <button className="btn btn-ghost btn-sm btn-icon" onClick={(e) => { e.stopPropagation(); onDelete(); }} title="Delete job">
              <Trash2 size={12} />
            </button>
          )}
          <span className="run-row-meta">{job.state}</span>
          <ChevronRight size={14} className="job-card-chevron" />
        </div>
      </div>

      {/* Line 2: meta — job type + cluster on left, timestamp on right */}
      <div className="job-card-meta">
        <span>
          {runTypeLabel(job.jobType)}{sweepTag}
          {isRemote && job.backendCluster && ` · ${job.backendCluster}`}
        </span>
        <span>{formatTimeAgo(job.createdAt)}</span>
      </div>

      {(projectName || evalObjective) && (
        <div className="run-row-business">
          {projectName && <span>Project: {projectName}</span>}
          {evalObjective && <span>Eval gate: {evalObjective}</span>}
          <span>Stage: {promotionStage}</span>
        </div>
      )}

      {/* Submit / pending phase messages */}
      {isSubmitting && job.submitPhase && (
        <div className="run-row-path" style={{ color: "var(--accent)", animation: "pulse 1.5s ease-in-out infinite" }}>
          {job.submitPhase}
        </div>
      )}
      {job.state === "pending" && (
        <div className="run-row-path" style={{ color: "var(--accent)", animation: "pulse 1.5s ease-in-out infinite" }}>
          {job.submitPhase || (isRemote ? "Queued — waiting for resources..." : "Starting...")}
        </div>
      )}
      {isFailed && isRemote && job.submitPhase && (
        <div className="error-alert-prominent">
          {job.submitPhase}
          {job.errorMessage && job.errorMessage !== job.submitPhase && (
            <span style={{ display: "block", marginTop: 4, fontWeight: 400 }}>{job.errorMessage}</span>
          )}
        </div>
      )}
      {isFailed && job.errorMessage && !(isRemote && job.submitPhase) && (
        <div className="error-alert-prominent">{job.errorMessage}</div>
      )}
      {job.modelPath && isRemote && (
        <div className="run-row-path" style={{ opacity: 0.7, fontSize: "0.8rem" }}>
          Remote model: {job.modelPath}
        </div>
      )}

      <LocalProgressPanel job={job} localTask={localTask} progress={progress} />
      <PullProgressPanel
        pullDone={pullDone}
        pullError={pullError}
        pulling={pulling}
        pullProgress={pullProgress}
      />
      {isLocal && (
        <LocalJobOutputPanel
          isExpanded={localExpanded}
          job={job}
          localLogRef={localLogRef}
          localTask={localTask}
        />
      )}
      {isRemote && (
        <RemoteJobLogPanel
          handleLogScroll={handleLogScroll}
          loading={loading}
          logContainerRef={logContainerRef}
          logs={logs}
          showLogs={showLogs}
        />
      )}
    </div>
  );
}
