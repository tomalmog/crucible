import { invoke } from "@tauri-apps/api/core";
import type { ClusterConfig, ClusterInfo, RemoteDatasetInfo, RemoteJobRecord } from "../types/remote";
import { cached, cacheGet, cacheSet, invalidate, sshLimited } from "./remoteCache";

const TERMINAL_STATES = new Set(["completed", "failed", "cancelled"]);

export async function listClusters(dataRoot: string): Promise<ClusterConfig[]> {
  return invoke<ClusterConfig[]>("list_clusters", { dataRoot });
}

export async function listRemoteJobs(dataRoot: string): Promise<RemoteJobRecord[]> {
  return invoke<RemoteJobRecord[]>("list_remote_jobs", { dataRoot });
}

export async function getRemoteJob(dataRoot: string, jobId: string): Promise<RemoteJobRecord> {
  return invoke<RemoteJobRecord>("get_remote_job", { dataRoot, jobId });
}

/** Synchronous cache read for remote job results. */
export function getCachedRemoteJobResult(dataRoot: string, jobId: string): Record<string, unknown> | undefined {
  const raw = cacheGet<string>(`jobResult:${dataRoot}:${jobId}`);
  if (raw == null) return undefined;
  try { return JSON.parse(raw) as Record<string, unknown>; } catch { return undefined; }
}

export function getCachedRemoteJobLogs(dataRoot: string, jobId: string): string | undefined {
  return cacheGet<string>(`jobLogs:${dataRoot}:${jobId}`);
}

export async function getRemoteJobResult(
  dataRoot: string,
  jobId: string,
  bypassCache?: boolean,
): Promise<Record<string, unknown>> {
  const key = `jobResult:${dataRoot}:${jobId}`;
  if (bypassCache) invalidate(key);
  const raw = await cached(key, Infinity, () =>
    sshLimited(() => invoke<string>("get_remote_job_result", { dataRoot, jobId })),
  );
  return JSON.parse(raw) as Record<string, unknown>;
}

export async function getRemoteJobLogs(
  dataRoot: string,
  jobId: string,
  jobState?: string,
  bypassCache?: boolean,
): Promise<string> {
  const key = `jobLogs:${dataRoot}:${jobId}`;
  if (bypassCache) invalidate(key);
  const wantInfinite = !!(jobState && TERMINAL_STATES.has(jobState));
  const ttl = wantInfinite ? Infinity : 10_000;
  const result = await cached(key, ttl, () => sshLimited(() => invoke<string>("get_remote_job_logs", { dataRoot, jobId })));
  // Don't permanently cache placeholder messages — they may resolve on retry
  if (wantInfinite && result.startsWith("[")) {
    invalidate(key);
  }
  return result;
}

export async function syncRemoteJobStatus(
  dataRoot: string,
  jobId: string,
  bypassCache?: boolean,
): Promise<RemoteJobRecord> {
  const key = `jobStatus:${dataRoot}:${jobId}`;
  if (bypassCache) invalidate(key);
  const result = await cached(key, 10_000, () =>
    sshLimited(() => invoke<RemoteJobRecord>("sync_remote_job_status", { dataRoot, jobId })),
  );
  // Upgrade to infinite TTL if the job reached a terminal state
  if (TERMINAL_STATES.has(result.state)) {
    cacheSet(key, result, Infinity);
    // Clear any stale log cache so the next fetch gets fresh logs
    invalidate(`jobLogs:${dataRoot}:${jobId}`);
  }
  return result;
}

export async function deleteRemoteJob(dataRoot: string, jobId: string): Promise<void> {
  const result = await sshLimited(() => invoke<void>("delete_remote_job", { dataRoot, jobId }));
  invalidate(`jobStatus:${dataRoot}:${jobId}`, `jobLogs:${dataRoot}:${jobId}`);
  return result;
}

export async function cancelRemoteJob(dataRoot: string, jobId: string): Promise<RemoteJobRecord> {
  const result = await sshLimited(() => invoke<RemoteJobRecord>("cancel_remote_job", { dataRoot, jobId }));
  invalidate(`jobStatus:${dataRoot}:${jobId}`, `jobLogs:${dataRoot}:${jobId}`);
  return result;
}

export async function listRemoteDatasets(
  dataRoot: string,
  cluster: string,
  bypassCache?: boolean,
): Promise<RemoteDatasetInfo[]> {
  const key = `datasets:${dataRoot}:${cluster}`;
  if (bypassCache) invalidate(key);
  return cached(key, 5 * 60_000, () =>
    sshLimited(() => invoke<RemoteDatasetInfo[]>("list_remote_datasets", { dataRoot, cluster })),
  );
}

export async function pushDatasetToCluster(dataRoot: string, cluster: string, dataset: string): Promise<void> {
  const result = await sshLimited(() => invoke<void>("push_dataset_to_cluster", { dataRoot, cluster, dataset }));
  invalidate(`datasets:${dataRoot}:${cluster}`);
  return result;
}

export async function pullDatasetFromCluster(dataRoot: string, cluster: string, dataset: string): Promise<void> {
  const result = await sshLimited(() => invoke<void>("pull_dataset_from_cluster", { dataRoot, cluster, dataset }));
  invalidate(`datasets:${dataRoot}:${cluster}`);
  return result;
}

export async function getRemoteModelSizes(
  dataRoot: string,
  cluster: string,
  bypassCache?: boolean,
): Promise<Record<string, number>> {
  const key = `modelSizes:${dataRoot}:${cluster}`;
  if (bypassCache) invalidate(key);
  return cached(key, 30_000, () =>
    sshLimited(() => invoke<Record<string, number>>("get_remote_model_sizes", { dataRoot, cluster })),
  );
}

export async function getClusterInfo(
  dataRoot: string,
  cluster: string,
  bypassCache?: boolean,
): Promise<ClusterInfo> {
  const key = `clusterInfo:${dataRoot}:${cluster}`;
  if (bypassCache) invalidate(key);
  return cached(key, 30_000, () =>
    sshLimited(() => invoke<ClusterInfo>("get_cluster_info", { dataRoot, cluster })),
  );
}

export async function deleteRemoteDataset(dataRoot: string, cluster: string, dataset: string): Promise<void> {
  const result = await sshLimited(() => invoke<void>("delete_remote_dataset_cmd", { dataRoot, cluster, dataset }));
  invalidate(`datasets:${dataRoot}:${cluster}`);
  return result;
}
