/** Unified job API — reads from .crucible/jobs/ via Tauri commands. */

import { invoke } from "@tauri-apps/api/core";
import type { JobRecord } from "../types/jobs";
import { TERMINAL_JOB_STATES } from "../types/jobs";
import { cached, cacheGet, cacheSet, invalidate, sshLimited } from "./remoteCache";

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
  const result = await cached(key, 4_000, () =>
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

/** Synchronous cache read — returns undefined on miss. */
export function getCachedJobResult(jobId: string): Record<string, unknown> | undefined {
  return cacheGet<Record<string, unknown>>(`job-result-${jobId}`);
}

export function getCachedJobLogs(jobId: string): string | undefined {
  return cacheGet<string>(`job-logs-${jobId}`);
}

export async function getJobLogs(
  dataRoot: string,
  jobId: string,
  jobState?: string,
): Promise<string> {
  const key = `job-logs-${jobId}`;
  const ttl = jobState && TERMINAL_JOB_STATES.has(jobState as any) ? Infinity : 10_000;
  return cached(key, ttl, () =>
    invoke<string>("get_unified_job_logs", { dataRoot, jobId }),
  );
}

export async function getJobResult(
  dataRoot: string,
  jobId: string,
  jobState?: string,
): Promise<Record<string, unknown>> {
  const key = `job-result-${jobId}`;
  const isTerminal = jobState && TERMINAL_JOB_STATES.has(jobState as any);

  // If we have a cached non-empty result, return it immediately.
  const prior = cacheGet<Record<string, unknown>>(key);
  if (prior && Object.keys(prior).length > 0) {
    return prior;
  }

  // Clear any stale empty-result entry so `cached()` re-fetches.
  if (prior !== undefined) {
    invalidate(key);
  }

  const ttl = isTerminal ? Infinity : 10_000;
  const result = await cached(key, ttl, async () => {
    const raw: string = await invoke("get_unified_job_result", {
      dataRoot,
      jobId,
    });
    return JSON.parse(raw) as Record<string, unknown>;
  });

  // If result came back empty, use a short TTL so we retry quickly.
  // For just-completed jobs, result.json may still be writing on the cluster.
  if (Object.keys(result).length === 0) {
    cacheSet(key, result, isTerminal ? 3_000 : 10_000);
  }

  return result;
}
