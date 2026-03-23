import { useCallback, useEffect, useRef, useState } from "react";
import type { JobRecord } from "../types/jobs";
import { listJobs, deleteJob, cancelJob, syncJobState } from "../api/jobsApi";

const POLL_INTERVAL_MS = 2000;

export function useUnifiedJobs(dataRoot: string) {
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const hasFetched = useRef(false);

  const refresh = useCallback(() => {
    if (!dataRoot) return;
    listJobs(dataRoot)
      .then((result) => {
        setJobs(result);
        if (!hasFetched.current) {
          hasFetched.current = true;
          setIsLoading(false);
        }
      })
      .catch(() => {
        setJobs([]);
        if (!hasFetched.current) {
          hasFetched.current = true;
          setIsLoading(false);
        }
      });
  }, [dataRoot]);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [refresh]);

  const removeJob = useCallback(async (jobId: string) => {
    if (!dataRoot) return;
    await deleteJob(dataRoot, jobId);
    refresh();
  }, [dataRoot, refresh]);

  const cancel = useCallback(async (jobId: string) => {
    if (!dataRoot) return;
    await cancelJob(dataRoot, jobId);
    refresh();
  }, [dataRoot, refresh]);

  const sync = useCallback(async (jobId: string) => {
    if (!dataRoot) return;
    await syncJobState(dataRoot, jobId, true);
    refresh();
  }, [dataRoot, refresh]);

  return { jobs, isLoading, refresh, removeJob, cancel, sync };
}
