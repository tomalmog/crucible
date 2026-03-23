import { useCallback, useEffect, useRef, useState } from "react";
import { useCrucible } from "../../context/CrucibleContext";
import type { RemoteJobRecord } from "../../types/remote";
import { startCrucibleCommand, getCrucibleCommandStatus } from "../../api/studioApi";
import { getRemoteJobLogs, syncRemoteJobStatus } from "../../api/remoteApi";
import { statusBadgeClass } from "./JobsPage";
import {
  Activity,
  ChevronDown,
  ChevronRight,
  Eye,
  Trash2,
  Check,
  Server,
  Terminal,
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
  const logEndRef = useRef<HTMLDivElement>(null);
  const streamTaskRef = useRef<string | null>(null);
  const streamPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

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
            setLogs(stdout.trim());
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
            setLoading(false);
            // Stream ended — fetch full logs to capture anything missed
            fetchLogs(true);
          }
        } catch {
          if (streamPollRef.current) clearInterval(streamPollRef.current);
          streamPollRef.current = null;
          streamTaskRef.current = null;
          setLoading(false);
        }
      }, 2_000);
    } catch (err) {
      setLogs(`Error starting log stream: ${err}`);
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
      // Job finished — do a final one-shot fetch for complete logs
      fetchLogs(true);
    }
  }, [job.state, stopLogStream, fetchLogs]);

  useEffect(() => () => stopLogStream(), [stopLogStream]);

  // Auto-scroll log container to bottom (without moving the page)
  useEffect(() => {
    const el = logEndRef.current?.parentElement;
    if (el) el.scrollTop = el.scrollHeight;
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
          {!isSubmitting && job.slurmJobId && (
            <span className="run-row-meta">Slurm {job.slurmJobId}</span>
          )}
          {(isCompleted || isFailed) && (
            <button
              className="btn btn-sm"
              onClick={(e) => { e.stopPropagation(); onView(); }}
              title="View result"
            >
              <Eye size={12} /> Result
            </button>
          )}
          {!isSubmitting && (
            <button
              className="btn btn-sm"
              onClick={(e) => { e.stopPropagation(); toggleLogs(); }}
              title="View logs"
            >
              <Terminal size={12} /> Logs
            </button>
          )}
          {isCompleted && !hasLocalModel && !pulling && !pullDone && !["eval", "logit-lens", "activation-pca", "activation-patch"].includes(job.trainingMethod) && (
            <button
              className="btn btn-sm"
              onClick={(e) => { e.stopPropagation(); handlePull(); }}
              title="Download model to local machine"
            >
              <Download size={12} /> Pull Model
            </button>
          )}
          {(job.state === "pending" || job.state === "running") && (
            <button
              className="btn btn-sm"
              onClick={(e) => { e.stopPropagation(); onCancel(); }}
              title="Cancel job"
            >
              <XCircle size={12} /> Cancel
            </button>
          )}
          {showLogs && (
            <button
              className="btn btn-ghost btn-sm"
              onClick={(e) => { e.stopPropagation(); fetchLogs(true); }}
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
      {isSubmitting && job.submitPhase && (
        <div className="run-row-path" style={{ color: "var(--clr-accent)", animation: "pulse 1.5s ease-in-out infinite" }}>
          {job.submitPhase}
        </div>
      )}
      {job.state === "pending" && (
        <div className="run-row-path" style={{ color: "var(--clr-accent)", animation: "pulse 1.5s ease-in-out infinite" }}>
          Queued in Slurm — waiting for resources...
        </div>
      )}
      {isFailed && job.submitPhase && (
        <div className="error-alert-prominent" style={{ margin: "var(--space-xs) var(--space-md)" }}>
          {job.submitPhase}
        </div>
      )}
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
