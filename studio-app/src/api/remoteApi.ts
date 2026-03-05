import { invoke } from "@tauri-apps/api/core";
import type { ClusterConfig, RemoteJobRecord } from "../types/remote";

export async function listClusters(dataRoot: string): Promise<ClusterConfig[]> {
  return invoke<ClusterConfig[]>("list_clusters", { dataRoot });
}

export async function listRemoteJobs(dataRoot: string): Promise<RemoteJobRecord[]> {
  return invoke<RemoteJobRecord[]>("list_remote_jobs", { dataRoot });
}

export async function getRemoteJob(dataRoot: string, jobId: string): Promise<RemoteJobRecord> {
  return invoke<RemoteJobRecord>("get_remote_job", { dataRoot, jobId });
}

export async function getRemoteJobLogs(dataRoot: string, jobId: string): Promise<string> {
  return invoke<string>("get_remote_job_logs", { dataRoot, jobId });
}

export async function syncRemoteJobStatus(dataRoot: string, jobId: string): Promise<RemoteJobRecord> {
  return invoke<RemoteJobRecord>("sync_remote_job_status", { dataRoot, jobId });
}

export async function deleteRemoteJob(dataRoot: string, jobId: string): Promise<void> {
  return invoke<void>("delete_remote_job", { dataRoot, jobId });
}

export async function cancelRemoteJob(dataRoot: string, jobId: string): Promise<RemoteJobRecord> {
  return invoke<RemoteJobRecord>("cancel_remote_job", { dataRoot, jobId });
}
