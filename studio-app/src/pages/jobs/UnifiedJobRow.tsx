import { useCallback, useEffect, useRef, useState } from "react";
import { useCrucible } from "../../context/CrucibleContext";
import type { JobRecord } from "../../types/jobs";
import { TERMINAL_JOB_STATES } from "../../types/jobs";
import { getJobLogs, syncJobState } from "../../api/jobsApi";
import { startCrucibleCommand, getCrucibleCommandStatus } from "../../api/studioApi";
import { statusBadgeClass } from "./JobsPage";
import {
  Activity,
  ChevronDown,
  ChevronRight,
  Eye,
  Square,
  Trash2,
  Check,
  Server,
  Terminal,
  RefreshCw,
  Download,
  XCircle,
  Pencil,
  X,
} from "lucide-react";

const ACTIVE_STATES = new Set(["running", "pending"]);
const NON_TRAINING_TYPES = new Set(["eval", "logit-lens", "activation-pca", "activation-patch"]);

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
  localTask?: { stdout: string; stderr: string; status: string; elapsed_seconds: number; remaining_seconds: number; progress_percent: number; label: string | null; task_id: string };
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

  // Log streaming state (for remote jobs)
  const [showLogs, setShowLogs] = useState(false);
  const [logs, setLogs] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const logContainerRef = useRef<HTMLPreElement>(null);
  const localLogRef = useRef<HTMLPreElement>(null);
  const streamTaskRef = useRef<string | null>(null);
  const streamPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isStreamingRef = useRef(false);
  const userScrolledRef = useRef(false);
  const isAutoScrollingRef = useRef(false);

  // Inline rename state (for local jobs)
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");

  // Pull model state (for remote jobs)
  const [pullProgress, setPullProgress] = useState<string[]>([]);
  const [pullDone, setPullDone] = useState(false);
  const [pullError, setPullError] = useState<string | null>(null);
  const [pulling, setPulling] = useState(false);

  const displayName = job.label || job.modelName || job.jobId;
  const isFinished = TERMINAL_JOB_STATES.has(job.state);
  const isRunning = ACTIVE_STATES.has(job.state) || job.state === "submitting";
  const isSubmitting = job.state === "submitting";
  const isFailed = job.state === "failed";
  const isCompleted = job.state === "completed";
  const hasLocalModel = !!job.modelPathLocal;
  const failedOnCluster = isFailed && isRemote && !!job.backendJobId;

  // --- Remote log fetching ---
  const fetchLogs = useCallback(async (_bypassCache?: boolean) => {
    if (!dataRoot || !isRemote) return;
    setLoading(true);
    try {
      const content = await getJobLogs(dataRoot, job.jobId);
      setLogs(content?.trim() || "No logs available yet.");
    } catch (err) {
      setLogs(`Error fetching logs: ${err}`);
    } finally {
      setLoading(false);
    }
  }, [dataRoot, job.jobId, isRemote]);

  const startLogStream = useCallback(async () => {
    if (!dataRoot || streamTaskRef.current || !isRemote) return;
    // Stream via legacy remote logs for now (the backend_job_id is the rj- ID)
    const legacyId = job.backendJobId;
    if (!legacyId) { fetchLogs(); return; }
    setLoading(true);
    isStreamingRef.current = true;
    userScrolledRef.current = false;
    try {
      const { task_id } = await startCrucibleCommand(dataRoot, [
        "remote", "logs", "--job-id", legacyId, "--follow", "--tail", "200",
      ]);
      streamTaskRef.current = task_id;
      const completionDetectedRef = { current: false };
      streamPollRef.current = setInterval(async () => {
        try {
          const status = await getCrucibleCommandStatus(task_id);
          const stdout = status.stdout || "";
          if (stdout) {
            const trimmed = stdout.trim();
            setLogs(prev => trimmed.length >= prev.length ? trimmed : prev);
            setLoading(false);
            if (!completionDetectedRef.current &&
                (stdout.includes("CRUCIBLE_AGENT_COMPLETE") || stdout.includes("CRUCIBLE_AGENT_ERROR"))) {
              completionDetectedRef.current = true;
              syncJobState(dataRoot!, job.jobId, true)
                .then(() => onRefresh?.())
                .catch(console.error);
            }
          }
          if (status.status !== "running") {
            if (streamPollRef.current) clearInterval(streamPollRef.current);
            streamPollRef.current = null;
            streamTaskRef.current = null;
            isStreamingRef.current = false;
            setLoading(false);
          }
        } catch {
          if (streamPollRef.current) clearInterval(streamPollRef.current);
          streamPollRef.current = null;
          streamTaskRef.current = null;
          isStreamingRef.current = false;
          setLoading(false);
        }
      }, 2_000);
    } catch (err) {
      setLogs(`Error starting log stream: ${err}`);
      isStreamingRef.current = false;
      setLoading(false);
    }
  }, [dataRoot, job.jobId, job.backendJobId, fetchLogs, onRefresh, isRemote]);

  const stopLogStream = useCallback(() => {
    if (streamPollRef.current) {
      clearInterval(streamPollRef.current);
      streamPollRef.current = null;
    }
    streamTaskRef.current = null;
  }, []);

  const toggleLogs = useCallback(() => {
    if (isLocal) return; // local jobs show logs inline via localTask
    const next = !showLogs;
    setShowLogs(next);
    if (next && !logs) {
      if (ACTIVE_STATES.has(job.state)) startLogStream();
      else fetchLogs();
    }
    if (!next) stopLogStream();
  }, [showLogs, logs, fetchLogs, startLogStream, stopLogStream, job.state, isLocal]);

  useEffect(() => {
    if (!ACTIVE_STATES.has(job.state) && streamTaskRef.current) {
      stopLogStream();
    }
  }, [job.state, stopLogStream]);

  useEffect(() => () => stopLogStream(), [stopLogStream]);

  useEffect(() => {
    if (!isStreamingRef.current || userScrolledRef.current) return;
    const el = logContainerRef.current;
    if (el) {
      isAutoScrollingRef.current = true;
      el.scrollTop = el.scrollHeight;
      requestAnimationFrame(() => { isAutoScrollingRef.current = false; });
    }
  }, [logs]);

  // Autoscroll local job logs when running
  useEffect(() => {
    if (!isLocal || !localTask) return;
    const el = localLogRef.current;
    if (el && localTask.status === "running") {
      isAutoScrollingRef.current = true;
      el.scrollTop = el.scrollHeight;
      requestAnimationFrame(() => { isAutoScrollingRef.current = false; });
    }
  }, [isLocal, localTask?.stdout, localTask?.status]);

  const handleLogScroll = useCallback((e: React.UIEvent<HTMLPreElement>) => {
    if (!isStreamingRef.current || isAutoScrollingRef.current) return;
    const el = e.currentTarget;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    userScrolledRef.current = !atBottom;
  }, []);

  // Auto-expand/stream for remote active jobs
  useEffect(() => {
    if (isRemote && failedOnCluster && !showLogs && !logs) setShowLogs(true);
  }, [failedOnCluster]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (isRemote && isRunning && !isSubmitting && !showLogs) {
      setShowLogs(true);
      startLogStream();
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // --- Model pull (remote only) ---
  const handlePull = useCallback(async () => {
    if (!dataRoot || !job.backendJobId) return;
    setPulling(true);
    setPullProgress([]);
    setPullError(null);
    setPullDone(false);
    try {
      const pullArgs = ["remote", "pull-model", "--job-id", job.backendJobId];
      if (job.modelName) pullArgs.push("--model-name", job.modelName);
      const task = await startCrucibleCommand(dataRoot, pullArgs);
      const poll = setInterval(async () => {
        try {
          const status = await getCrucibleCommandStatus(task.task_id);
          const lines = (status.stdout || "")
            .split("\n")
            .filter((l: string) => l.startsWith("CRUCIBLE_PULL_PROGRESS: "))
            .map((l: string) => l.replace("CRUCIBLE_PULL_PROGRESS: ", ""));
          if (lines.length > 0) setPullProgress(lines);
          if (status.status !== "running") {
            clearInterval(poll);
            if (status.status === "completed") {
              setPullDone(true);
              refreshModels().catch(console.error);
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
  }, [dataRoot, job.backendJobId, job.modelName, refreshModels]);

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

  return (
    <div className="run-row section-divider">
      <div className="run-row-header">
        <div className="flex-row">
          <button
            className="btn btn-ghost btn-sm btn-icon"
            onClick={() => isLocal ? setLocalExpanded((p) => !p) : toggleLogs()}
          >
            {(isLocal ? localExpanded : showLogs)
              ? <ChevronDown size={14} />
              : <ChevronRight size={14} />}
          </button>
          {isLocal && editing ? (
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
                <button className="btn btn-ghost btn-sm btn-icon" onClick={startEditing} title="Rename">
                  <Pencil size={11} />
                </button>
              )}
            </>
          )}
          {isRemote && job.backendCluster && (
            <span className="badge"><Server size={10} /> {job.backendCluster}</span>
          )}
          <span className={statusBadgeClass(job.state)}>{job.state}</span>
          {isRunning && (
            <span className="badge badge-accent" style={{ fontSize: "0.7rem" }}>
              <Activity size={10} /> live
            </span>
          )}
        </div>
        <div className="flex-row">
          {isRemote && !isSubmitting && job.backendJobId && (
            <span className="run-row-meta">
              {job.backend === "slurm" ? `Slurm ${job.backendJobId.startsWith("rj-") ? "" : job.backendJobId}` : job.backendJobId}
            </span>
          )}
          {isLocal && localTask && (
            <span className="run-row-meta">
              {isFinished
                ? `took ${formatDur(localTask.elapsed_seconds)}`
                : formatDur(localTask.elapsed_seconds)}
            </span>
          )}
          {isLocal && job.state === "running" && onKill && (
            <button className="btn btn-sm" onClick={onKill} title="Kill process">
              <Square size={12} /> Kill
            </button>
          )}
          {(isCompleted || isFailed) && (
            <button className="btn btn-sm" onClick={(e) => { e.stopPropagation(); onView(); }} title="View result">
              <Eye size={12} /> Result
            </button>
          )}
          {isRemote && !isSubmitting && (
            <button className="btn btn-sm" onClick={(e) => { e.stopPropagation(); toggleLogs(); }} title="View logs">
              <Terminal size={12} /> Logs
            </button>
          )}
          {isRemote && isCompleted && !hasLocalModel && !pulling && !pullDone && !NON_TRAINING_TYPES.has(job.jobType) && (
            <button className="btn btn-sm" onClick={(e) => { e.stopPropagation(); handlePull(); }} title="Download model">
              <Download size={12} /> Pull Model
            </button>
          )}
          {isRemote && ACTIVE_STATES.has(job.state) && (
            <button className="btn btn-sm" onClick={(e) => { e.stopPropagation(); onCancel(); }} title="Cancel job">
              <XCircle size={12} /> Cancel
            </button>
          )}
          {isRemote && showLogs && (
            <button className="btn btn-ghost btn-sm" onClick={(e) => { e.stopPropagation(); fetchLogs(true); }} title="Refresh logs">
              <RefreshCw size={12} />
            </button>
          )}
          {isFinished && (
            <button className="btn btn-ghost btn-sm" onClick={(e) => { e.stopPropagation(); onDelete(); }} title="Delete job">
              <Trash2 size={12} />
            </button>
          )}
        </div>
      </div>

      <div className="run-row-path">
        {job.jobType}{sweepTag}
      </div>
      {isSubmitting && job.submitPhase && (
        <div className="run-row-path" style={{ color: "var(--clr-accent)", animation: "pulse 1.5s ease-in-out infinite" }}>
          {job.submitPhase}
        </div>
      )}
      {job.state === "pending" && isRemote && (
        <div className="run-row-path" style={{ color: "var(--clr-accent)", animation: "pulse 1.5s ease-in-out infinite" }}>
          Queued — waiting for resources...
        </div>
      )}
      {isFailed && isRemote && job.submitPhase && (
        <div className="error-alert-prominent" style={{ margin: "var(--space-xs) var(--space-md)" }}>
          {job.submitPhase}
        </div>
      )}
      {job.modelPath && isRemote && (
        <div className="run-row-path" style={{ opacity: 0.7, fontSize: "0.8rem" }}>
          Remote model: {job.modelPath}
        </div>
      )}

      {/* Local job progress bar */}
      {isLocal && localTask && localTask.status === "running" && (
        <div className="progress-bar gap-top-sm">
          <div className="progress-bar-header">
            <span className="progress-label">Progress</span>
            <span className="progress-value">{localTask.progress_percent.toFixed(0)}%</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${localTask.progress_percent}%` }} />
          </div>
          <div className="progress-bar-footer">
            <span>Elapsed {formatDur(localTask.elapsed_seconds)}</span>
            <span>~{formatDur(localTask.remaining_seconds)} remaining</span>
          </div>
        </div>
      )}

      {/* Pull progress (remote) */}
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
          {pullError && <div className="error-alert-prominent">{pullError}</div>}
        </div>
      )}

      {/* Local job expanded output */}
      {isLocal && localExpanded && localTask && (
        <div className="job-expanded">
          {localTask.stdout && (
            <div>
              <div className="job-output-label">stdout</div>
              <pre ref={localLogRef} className="console console-tall">{localTask.stdout}</pre>
            </div>
          )}
          {localTask.stderr && (
            <div>
              <details open={localTask.status !== "failed"}>
                <summary className={`job-traceback-toggle ${localTask.status === "failed" ? "error-text" : ""}`}>
                  {localTask.status === "failed" ? "full traceback" : "logs"}
                </summary>
                <pre className="console console-short">{localTask.stderr}</pre>
              </details>
            </div>
          )}
          {!localTask.stdout && !localTask.stderr && (
            <div className="job-no-output">No output yet.</div>
          )}
        </div>
      )}

      {/* Remote job logs */}
      {isRemote && showLogs && (
        <div className="job-expanded">
          {loading && !logs && (
            <div className="job-no-output">Fetching logs from cluster...</div>
          )}
          {logs ? (
            <div>
              <div className="job-output-label">Remote Logs {loading && "(refreshing...)"}</div>
              <pre ref={logContainerRef} className="console console-tall" onScroll={handleLogScroll}>
                {logs}
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

function formatDur(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}
