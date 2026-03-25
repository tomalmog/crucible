import { useCallback, useEffect, useRef, useState } from "react";
import type { JobRecord } from "../types/jobs";
import { TERMINAL_JOB_STATES } from "../types/jobs";
import { listJobs, deleteJob, cancelJob, syncJobState } from "../api/jobsApi";

const POLL_INTERVAL_MS = 2000;
const REMOTE_SYNC_INTERVAL_MS = 5_000;

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

  // Periodically sync state for active remote jobs via SSH
  const jobsRef = useRef(jobs);
  jobsRef.current = jobs;
  useEffect(() => {
    if (!dataRoot) return;
    const interval = setInterval(() => {
      const active = jobsRef.current.filter(
        (j) => j.backend !== "local" && !TERMINAL_JOB_STATES.has(j.state),
      );
      for (const j of active) {
        syncJobState(dataRoot, j.jobId).then(() => refresh()).catch(console.error);
      }
    }, REMOTE_SYNC_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [dataRoot, refresh]);

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
