import { useCallback, useEffect, useState } from "react";
import { cancelRemoteJob, deleteRemoteJob, listRemoteJobs } from "../api/remoteApi";
import type { RemoteJobRecord } from "../types/remote";

const POLL_INTERVAL_MS = 2000;

export function useRemoteJobs(dataRoot: string) {
  const [jobs, setJobs] = useState<RemoteJobRecord[]>([]);

  const refresh = useCallback(() => {
    if (!dataRoot) return;
    listRemoteJobs(dataRoot)
      .then(setJobs)
      .catch(() => setJobs([]));
  }, [dataRoot]);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [refresh]);

  const removeJob = useCallback(async (jobId: string) => {
    if (!dataRoot) return;
    await deleteRemoteJob(dataRoot, jobId);
    refresh();
  }, [dataRoot, refresh]);

  const cancelJob = useCallback(async (jobId: string) => {
    if (!dataRoot) return;
    await cancelRemoteJob(dataRoot, jobId);
    refresh();
  }, [dataRoot, refresh]);

  return { jobs, refresh, removeJob, cancelJob };
}
