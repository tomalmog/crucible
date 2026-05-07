import { useCallback, useEffect, useRef, useState } from "react";
import type { RefObject, UIEvent } from "react";
import { getJobLogs, syncJobState } from "../../api/jobsApi";
import { getCrucibleCommandStatus, startCrucibleCommand } from "../../api/studioApi";
import type { CommandTaskStatus } from "../../types";
import type { JobRecord } from "../../types/jobs";
import { ACTIVE_STATES } from "./jobRowDisplay";

interface UseUnifiedJobLogsOptions {
  dataRoot: string;
  failedOnCluster: boolean;
  isLocal: boolean;
  isRemote: boolean;
  isRunning: boolean;
  isSubmitting: boolean;
  job: JobRecord;
  localTask?: CommandTaskStatus;
  onRefresh?: () => void;
}

interface UnifiedJobLogsState {
  fetchLogs: (bypassCache?: boolean) => Promise<void>;
  handleLogScroll: (event: UIEvent<HTMLPreElement>) => void;
  loading: boolean;
  localLogRef: RefObject<HTMLPreElement | null>;
  logContainerRef: RefObject<HTMLPreElement | null>;
  logs: string;
  showLogs: boolean;
  toggleLogs: () => void;
}

export function useUnifiedJobLogs({
  dataRoot,
  failedOnCluster,
  isLocal,
  isRemote,
  isRunning,
  isSubmitting,
  job,
  localTask,
  onRefresh,
}: UseUnifiedJobLogsOptions): UnifiedJobLogsState {
  const [showLogs, setShowLogs] = useState(false);
  const [logs, setLogs] = useState("");
  const [loading, setLoading] = useState(false);
  const logContainerRef = useRef<HTMLPreElement>(null);
  const localLogRef = useRef<HTMLPreElement>(null);
  const streamTaskRef = useRef<string | null>(null);
  const streamPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isStreamingRef = useRef(false);
  const userScrolledRef = useRef(false);
  const isAutoScrollingRef = useRef(false);
  const hasAutoOpenedRef = useRef(false);

  const stopLogStream = useCallback(() => {
    if (streamPollRef.current) {
      clearInterval(streamPollRef.current);
      streamPollRef.current = null;
    }
    streamTaskRef.current = null;
    isStreamingRef.current = false;
  }, []);

  const fetchLogs = useCallback(async (_bypassCache?: boolean) => {
    if (!dataRoot || !isRemote) return;
    setLoading(true);
    try {
      const content = await getJobLogs(dataRoot, job.jobId, job.state);
      setLogs(content?.trim() || "No logs available yet.");
    } catch (err) {
      setLogs(`Error fetching logs: ${err}`);
    } finally {
      setLoading(false);
    }
  }, [dataRoot, isRemote, job.jobId, job.state]);

  const startLogStream = useCallback(async () => {
    if (!dataRoot || streamTaskRef.current || !isRemote) return;
    const legacyId = job.backendJobId;
    if (!legacyId) {
      fetchLogs().catch(console.error);
      return;
    }
    setLoading(true);
    isStreamingRef.current = true;
    userScrolledRef.current = false;
    try {
      const { task_id } = await startCrucibleCommand(dataRoot, [
        "remote",
        "logs",
        "--job-id",
        legacyId,
        "--follow",
        "--tail",
        "200",
      ]);
      streamTaskRef.current = task_id;
      const completionDetectedRef = { current: false };
      streamPollRef.current = setInterval(async () => {
        try {
          const status = await getCrucibleCommandStatus(task_id);
          const stdout = status.stdout || "";
          if (stdout) {
            const trimmed = stdout.trim();
            setLogs((previous) => (trimmed.length >= previous.length ? trimmed : previous));
            setLoading(false);
            if (
              !completionDetectedRef.current &&
              (stdout.includes("CRUCIBLE_AGENT_COMPLETE") || stdout.includes("CRUCIBLE_AGENT_ERROR"))
            ) {
              completionDetectedRef.current = true;
              syncJobState(dataRoot, job.jobId, true)
                .then(() => onRefresh?.())
                .catch(console.error);
            }
          }
          if (status.status !== "running") {
            stopLogStream();
            setLoading(false);
          }
        } catch {
          stopLogStream();
          setLoading(false);
        }
      }, 2_000);
    } catch (err) {
      setLogs(`Error starting log stream: ${err}`);
      isStreamingRef.current = false;
      setLoading(false);
    }
  }, [dataRoot, fetchLogs, isRemote, job.backendJobId, job.jobId, onRefresh, stopLogStream]);

  const toggleLogs = useCallback(() => {
    if (isLocal || job.state === "submitting") return;
    const next = !showLogs;
    setShowLogs(next);
    if (next && !logs) {
      if (ACTIVE_STATES.has(job.state)) startLogStream().catch(console.error);
      else fetchLogs().catch(console.error);
    }
    if (!next) stopLogStream();
  }, [fetchLogs, isLocal, job.state, logs, showLogs, startLogStream, stopLogStream]);

  const handleLogScroll = useCallback((event: UIEvent<HTMLPreElement>) => {
    if (!isStreamingRef.current || isAutoScrollingRef.current) return;
    const element = event.currentTarget;
    const atBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 40;
    userScrolledRef.current = !atBottom;
  }, []);

  // Stop remote log polling as soon as the job leaves an active state.
  useEffect(() => {
    if (!ACTIVE_STATES.has(job.state) && streamTaskRef.current) stopLogStream();
  }, [job.state, stopLogStream]);

  // Clear any remote polling interval when the row unmounts.
  useEffect(() => () => stopLogStream(), [stopLogStream]);

  // Keep streaming remote logs pinned to the bottom until the user scrolls away.
  useEffect(() => {
    if (!isStreamingRef.current || userScrolledRef.current) return undefined;
    const element = logContainerRef.current;
    if (!element) return undefined;
    isAutoScrollingRef.current = true;
    element.scrollTop = element.scrollHeight;
    const frame = requestAnimationFrame(() => {
      isAutoScrollingRef.current = false;
    });
    return () => cancelAnimationFrame(frame);
  }, [logs]);

  // Keep local running logs pinned to the bottom while stdout is still changing.
  useEffect(() => {
    if (!isLocal || !localTask || localTask.status !== "running") return undefined;
    const element = localLogRef.current;
    if (!element) return undefined;
    isAutoScrollingRef.current = true;
    element.scrollTop = element.scrollHeight;
    const frame = requestAnimationFrame(() => {
      isAutoScrollingRef.current = false;
    });
    return () => cancelAnimationFrame(frame);
  }, [isLocal, localTask?.stdout, localTask?.status]);

  // Open cluster failure logs automatically because the traceback is the next useful action.
  useEffect(() => {
    if (isRemote && failedOnCluster && job.backendJobId && !showLogs && !logs) {
      setShowLogs(true);
      fetchLogs().catch(console.error);
    }
  }, [failedOnCluster, fetchLogs, isRemote, job.backendJobId, logs, showLogs]);

  // Open a live stream once for already-running remote jobs so progress is visible.
  useEffect(() => {
    if (hasAutoOpenedRef.current) return;
    if (isRemote && isRunning && !isSubmitting && job.backendJobId && !showLogs) {
      hasAutoOpenedRef.current = true;
      setShowLogs(true);
      startLogStream().catch(console.error);
    }
  }, [isRemote, isRunning, isSubmitting, job.backendJobId, showLogs, startLogStream]);

  return {
    fetchLogs,
    handleLogScroll,
    loading,
    localLogRef,
    logContainerRef,
    logs,
    showLogs,
    toggleLogs,
  };
}
