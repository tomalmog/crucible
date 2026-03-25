import { invoke } from "@tauri-apps/api/core";
import type { ClusterConfig, ClusterInfo, RemoteDatasetInfo } from "../types/remote";
import { cached, invalidate, sshLimited } from "./remoteCache";

export async function listClusters(dataRoot: string): Promise<ClusterConfig[]> {
  return invoke<ClusterConfig[]>("list_clusters", { dataRoot });
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
