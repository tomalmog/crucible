import { useCallback, useEffect, useRef, useState } from "react";
import { useCrucible } from "../../context/CrucibleContext";
import type { RemoteJobRecord } from "../../types/remote";
import { startCrucibleCommand, getCrucibleCommandStatus } from "../../api/studioApi";
import { getRemoteJobLogs, syncRemoteJobStatus } from "../../api/remoteApi";
import { jobLabel } from "../../utils/jobLabels";
import { jobAccentColor } from "./JobsPage";
import { formatTimeAgo } from "../../utils/formatTime";
import {
  ChevronRight,
  Trash2,
  Check,
  RefreshCw,
  Download,
  XCircle,
} from "lucide-react";

const ACTIVE_STATES = new Set(["running", "pending"]);

export function RemoteJobRow({ job, onDelete, onCancel, onView, onRefresh }: { job: RemoteJobRecord; onDelete: () => void; onCancel: () => void; onView: () => void; onRefresh?: () => void }) {
  const { dataRoot, refreshModels } = useCrucible();
  const sweepTag = job.isSweep ? ` (sweep, ${job.sweepArraySize} trials)` : "";
  const [showLogs, setShowLogs] = useState(false);
  const [logs, setLogs] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const logContainerRef = useRef<HTMLPreElement>(null);
  const streamTaskRef = useRef<string | null>(null);
  const streamPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isStreamingRef = useRef(false);
  const userScrolledRef = useRef(false);
  const isAutoScrollingRef = useRef(false);

  // One-shot fetch for completed/failed jobs
  const fetchLogs = useCallback(async (bypassCache?: boolean) => {
    if (!dataRoot) return;
    setLoading(true);
    try {
      const content = await getRemoteJobLogs(dataRoot, job.jobId, job.state, bypassCache);
      setLogs(content?.trim() || "No logs available yet.");
    } catch (err) {
      setLogs(`Error fetching logs: ${err}`);
    } finally {
      setLoading(false);
    }
  }, [dataRoot, job.jobId, job.state]);

  // Start streaming logs via background task for active jobs
  const startLogStream = useCallback(async () => {
    if (!dataRoot || streamTaskRef.current) return;
    setLoading(true);
    isStreamingRef.current = true;
    userScrolledRef.current = false;
    try {
      const { task_id } = await startCrucibleCommand(dataRoot, [
        "remote", "logs", "--job-id", job.jobId, "--follow", "--tail", "200",
      ]);
      streamTaskRef.current = task_id;
      // Poll the task's stdout for incremental output
      const completionDetectedRef = { current: false };
      streamPollRef.current = setInterval(async () => {
        try {
          const status = await getCrucibleCommandStatus(task_id);
          const stdout = status.stdout || "";
          if (stdout) {
            const trimmed = stdout.trim();
            setLogs(prev => trimmed.length >= prev.length ? trimmed : prev);
            setLoading(false);
            // Detect agent completion in log stream → trigger immediate status sync
            if (!completionDetectedRef.current &&
                (stdout.includes("CRUCIBLE_AGENT_COMPLETE") || stdout.includes("CRUCIBLE_AGENT_ERROR"))) {
              completionDetectedRef.current = true;
              syncRemoteJobStatus(dataRoot!, job.jobId, true)
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
  }, [dataRoot, job.jobId, fetchLogs, onRefresh]);

  const stopLogStream = useCallback(() => {
    if (streamPollRef.current) {
      clearInterval(streamPollRef.current);
      streamPollRef.current = null;
    }
    streamTaskRef.current = null;
  }, []);

  const toggleLogs = useCallback(() => {
    const next = !showLogs;
    setShowLogs(next);
    if (next && !logs) {
      if (ACTIVE_STATES.has(job.state)) {
        startLogStream();
      } else {
        fetchLogs();
      }
    }
    if (!next) stopLogStream();
  }, [showLogs, logs, fetchLogs, startLogStream, stopLogStream, job.state]);

  // Clean up stream on unmount or when job completes
  useEffect(() => {
    if (!ACTIVE_STATES.has(job.state) && streamTaskRef.current) {
      stopLogStream();
    }
  }, [job.state, stopLogStream]);

  useEffect(() => () => stopLogStream(), [stopLogStream]);

  // Auto-scroll log container to bottom only while actively streaming
  // and user hasn't manually scrolled up
  useEffect(() => {
    if (!isStreamingRef.current || userScrolledRef.current) return;
    const el = logContainerRef.current;
    if (el) {
      isAutoScrollingRef.current = true;
      el.scrollTop = el.scrollHeight;
      requestAnimationFrame(() => { isAutoScrollingRef.current = false; });
    }
  }, [logs]);

  // Detect when user scrolls away from bottom → stop auto-scroll
  const handleLogScroll = useCallback((e: React.UIEvent<HTMLPreElement>) => {
    if (!isStreamingRef.current || isAutoScrollingRef.current) return;
    const el = e.currentTarget;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    userScrolledRef.current = !atBottom;
  }, []);

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
  }, [dataRoot, job.jobId, job.trainingMethod, refreshModels]);

  const isSubmitting = job.state === "submitting";
  const isRunning = job.state === "running" || job.state === "pending" || isSubmitting;
  const isCompleted = job.state === "completed";
  const isFailed = job.state === "failed";
  const isFinished = isCompleted || isFailed;
  const hasLocalModel = !!job.modelPathLocal;
  const failedOnCluster = isFailed && !!job.slurmJobId;

  // Auto-expand (but don't fetch) logs for jobs that failed on the cluster
  useEffect(() => {
    if (failedOnCluster && !showLogs && !logs) {
      setShowLogs(true);
    }
  }, [failedOnCluster]); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-start log streaming for running jobs
  useEffect(() => {
    if (isRunning && !isSubmitting && !showLogs) {
      setShowLogs(true);
      startLogStream();
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div
      className="job-card"
      style={{ "--job-accent": jobAccentColor(job.state) } as React.CSSProperties}
      onClick={() => {
        if (isFinished) onView();
        else toggleLogs();
      }}
    >
      {/* Line 1: status dot + name | secondary actions + status */}
      <div className="run-row-header">
        <div className="flex-row">
          <span className={"job-status-dot" + (isRunning ? " pulsing" : "")} />
          <span className="run-row-id">{jobLabel(job.trainingMethod, job.modelName) || job.jobId}</span>
        </div>
        <div className="flex-row">
          {isCompleted && !hasLocalModel && !pulling && !pullDone && !["eval", "logit-lens", "activation-pca", "activation-patch"].includes(job.trainingMethod) && (
            <button className="btn btn-ghost btn-sm btn-icon" onClick={(e) => { e.stopPropagation(); handlePull(); }} title="Pull model">
              <Download size={12} />
            </button>
          )}
          {(job.state === "pending" || job.state === "running") && (
            <button className="btn btn-ghost btn-sm btn-icon" onClick={(e) => { e.stopPropagation(); onCancel(); }} title="Cancel job">
              <XCircle size={12} />
            </button>
          )}
          {showLogs && (
            <button className="btn btn-ghost btn-sm btn-icon" onClick={(e) => { e.stopPropagation(); fetchLogs(true); }} title="Refresh logs">
              <RefreshCw size={12} />
            </button>
          )}
          {!isRunning && (
            <button className="btn btn-ghost btn-sm btn-icon" onClick={(e) => { e.stopPropagation(); onDelete(); }} title="Delete job">
              <Trash2 size={12} />
            </button>
          )}
          <span className="run-row-meta">{job.state}</span>
          <ChevronRight size={14} className="job-card-chevron" />
        </div>
      </div>

      {/* Line 2: meta — method + cluster + timestamp */}
      <div className="job-card-meta">
        <span>{job.trainingMethod}{sweepTag} · {job.clusterName}</span>
        <span>{formatTimeAgo(job.submittedAt)}</span>
      </div>

      {/* Submit / pending phase messages */}
      {isSubmitting && job.submitPhase && (
        <div className="run-row-path" style={{ color: "var(--accent)", animation: "pulse 1.5s ease-in-out infinite" }}>
          {job.submitPhase}
        </div>
      )}
      {job.state === "pending" && (
        <div className="run-row-path" style={{ color: "var(--accent)", animation: "pulse 1.5s ease-in-out infinite" }}>
          Queued in Slurm — waiting for resources...
        </div>
      )}
      {isFailed && job.submitPhase && (
        <div className="error-alert-prominent">{job.submitPhase}</div>
      )}
      {job.modelPathRemote && (
        <div className="run-row-path" style={{ opacity: 0.7, fontSize: "0.8rem" }}>
          Remote model: {job.modelPathRemote}
        </div>
      )}

      {/* Pull progress */}
      {(pulling || pullDone || pullError) && (
        <div>
          {pulling && (
            <div className="pull-steps">
              {pullProgress.map((step, i) => (
                <div key={i} className="pull-step">{step}</div>
              ))}
              {pullProgress.length === 0 && <div className="pull-step">Starting pull...</div>}
            </div>
          )}
          {pullDone && (
            <div className="pull-success flex-row" style={{ gap: "var(--space-xs)" }}>
              <Check size={14} /> Model pulled successfully!
            </div>
          )}
          {pullError && (
            <div className="error-alert-prominent">{pullError}</div>
          )}
        </div>
      )}

      {/* Logs */}
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
