import { useCallback, useEffect, useState } from "react";
import { useCrucible } from "../context/CrucibleContext";
import { getStorageBreakdown, listOrphanedRuns } from "../api/resourcesApi";
import { listClusters, getClusterInfo, getRemoteModelSizes, listRemoteDatasets } from "../api/remoteApi";
import { useJobs } from "./useJobs";
import type { StorageBreakdown, OrphanedRun } from "../types/resources";
import type { ClusterConfig, ClusterInfo, RemoteDatasetInfo } from "../types/remote";

export interface ClusterRemoteStorage {
  cluster: ClusterConfig;
  datasets: RemoteDatasetInfo[];
  modelSizes: Record<string, number>;
  clusterInfo: ClusterInfo | null;
}

export function useResourceData() {
  const { dataRoot, hardwareProfile } = useCrucible();
  const { jobs } = useJobs();

  const [storage, setStorage] = useState<StorageBreakdown | null>(null);
  const [orphans, setOrphans] = useState<OrphanedRun[]>([]);
  const [clusters, setClusters] = useState<ClusterConfig[]>([]);
  const [remoteStorage, setRemoteStorage] = useState<ClusterRemoteStorage[]>([]);
  const [loading, setLoading] = useState(true);

  const loadLocal = useCallback(async () => {
    try {
      const [sb, orph] = await Promise.all([
        getStorageBreakdown(dataRoot),
        listOrphanedRuns(dataRoot),
      ]);
      setStorage(sb);
      setOrphans(orph);
    } catch (err) {
      console.warn("[useResourceData] local load error:", err);
    }
  }, [dataRoot]);

  const loadRemote = useCallback(async (bypassCache?: boolean) => {
    try {
      const cls = await listClusters(dataRoot);
      setClusters(cls);
      const remote: ClusterRemoteStorage[] = await Promise.all(
        cls.map(async (c) => {
          const [datasets, modelSizes, clusterInfo] = await Promise.all([
            listRemoteDatasets(dataRoot, c.name, bypassCache).catch(() => [] as RemoteDatasetInfo[]),
            getRemoteModelSizes(dataRoot, c.name, bypassCache).catch(() => ({}) as Record<string, number>),
            getClusterInfo(dataRoot, c.name, bypassCache).catch(() => null as ClusterInfo | null),
          ]);
          return { cluster: c, datasets, modelSizes, clusterInfo };
        }),
      );
      setRemoteStorage(remote);
    } catch {
      setClusters([]);
      setRemoteStorage([]);
    }
  }, [dataRoot]);

  const load = useCallback(async (bypassCache?: boolean) => {
    setLoading(true);
    await Promise.all([loadLocal(), loadRemote(bypassCache)]);
    setLoading(false);
  }, [loadLocal, loadRemote]);

  const refresh = useCallback(() => load(true), [load]);

  useEffect(() => {
    load();
  }, [load]);

  return {
    storage,
    orphans,
    hardware: hardwareProfile,
    clusters,
    remoteStorage,
    localJobs: jobs,
    loading,
    refresh,
  };
}
