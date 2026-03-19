import { useCallback, useEffect, useState } from "react";
import { useCrucible } from "../context/CrucibleContext";
import { getStorageBreakdown, listOrphanedRuns } from "../api/resourcesApi";
import { listClusters, getRemoteModelSizes, listRemoteDatasets } from "../api/remoteApi";
import { useJobs } from "./useJobs";
import { useRemoteJobs } from "./useRemoteJobs";
import type { StorageBreakdown, OrphanedRun } from "../types/resources";
import type { ClusterConfig, RemoteDatasetInfo } from "../types/remote";

export interface ClusterRemoteStorage {
  cluster: ClusterConfig;
  datasets: RemoteDatasetInfo[];
  modelSizes: Record<string, number>;
}

export function useResourceData() {
  const { dataRoot, hardwareProfile } = useCrucible();
  const { jobs } = useJobs();
  const { jobs: remoteJobs } = useRemoteJobs(dataRoot);

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

  const loadRemote = useCallback(async () => {
    try {
      const cls = await listClusters(dataRoot);
      setClusters(cls);
      const remote: ClusterRemoteStorage[] = await Promise.all(
        cls.map(async (c) => {
          const [datasets, modelSizes] = await Promise.all([
            listRemoteDatasets(dataRoot, c.name).catch(() => [] as RemoteDatasetInfo[]),
            getRemoteModelSizes(dataRoot, c.name).catch(() => ({}) as Record<string, number>),
          ]);
          return { cluster: c, datasets, modelSizes };
        }),
      );
      setRemoteStorage(remote);
    } catch {
      setClusters([]);
      setRemoteStorage([]);
    }
  }, [dataRoot]);

  const refresh = useCallback(async () => {
    setLoading(true);
    await Promise.all([loadLocal(), loadRemote()]);
    setLoading(false);
  }, [loadLocal, loadRemote]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return {
    storage,
    orphans,
    hardware: hardwareProfile,
    clusters,
    remoteStorage,
    localJobs: jobs,
    remoteJobs,
    loading,
    refresh,
  };
}
