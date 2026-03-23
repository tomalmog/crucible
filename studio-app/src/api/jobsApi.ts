/** Unified job API — reads from .crucible/jobs/ via Tauri commands. */

import { invoke } from "@tauri-apps/api/core";
import type { JobRecord } from "../types/jobs";
import { TERMINAL_JOB_STATES } from "../types/jobs";
import { cached, cacheSet, invalidate, sshLimited } from "./remoteCache";

export async function listJobs(dataRoot: string): Promise<JobRecord[]> {
  return invoke("list_unified_jobs", { dataRoot });
}

export async function getJob(
  dataRoot: string,
  jobId: string,
): Promise<JobRecord> {
  return invoke("get_unified_job", { dataRoot, jobId });
}

export async function syncJobState(
  dataRoot: string,
  jobId: string,
  bypassCache = false,
): Promise<JobRecord> {
  const key = `sync-job-${jobId}`;
  if (bypassCache) invalidate(key);
  const result = await cached(key, 10_000, () =>
    sshLimited(() =>
      invoke<JobRecord>("sync_unified_job_state", { dataRoot, jobId }),
    ),
  );
  if (TERMINAL_JOB_STATES.has(result.state)) {
    cacheSet(key, result, Infinity);
  }
  return result;
}

export async function cancelJob(
  dataRoot: string,
  jobId: string,
): Promise<JobRecord> {
  return sshLimited(() => {
    invalidate(`sync-job-${jobId}`);
    return invoke("cancel_unified_job", { dataRoot, jobId });
  });
}

export async function deleteJob(
  dataRoot: string,
  jobId: string,
): Promise<void> {
  invalidate(`sync-job-${jobId}`);
  return invoke("delete_unified_job", { dataRoot, jobId });
}

export async function getJobLogs(
  dataRoot: string,
  jobId: string,
): Promise<string> {
  return invoke("get_unified_job_logs", { dataRoot, jobId });
}

export async function getJobResult(
  dataRoot: string,
  jobId: string,
): Promise<Record<string, unknown>> {
  const raw: string = await invoke("get_unified_job_result", {
    dataRoot,
    jobId,
  });
  return JSON.parse(raw);
}
