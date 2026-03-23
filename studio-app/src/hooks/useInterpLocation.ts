import { useCallback, useEffect, useMemo, useRef } from "react";
import { useCrucible } from "../context/CrucibleContext";
import { listClusters } from "../api/remoteApi";
import type { ClusterConfig } from "../types/remote";

interface InterpLocation {
  isRemote: boolean;
  clusterName: string;
}

/**
 * Detects whether a model path is local or remote and resolves the cluster.
 */
export function useInterpLocation(modelPath: string): InterpLocation {
  const { models, dataRoot } = useCrucible();
  const clustersRef = useRef<ClusterConfig[]>([]);

  useEffect(() => {
    if (!dataRoot) return;
    listClusters(dataRoot)
      .then((c) => { clustersRef.current = c; })
      .catch(() => { clustersRef.current = []; });
  }, [dataRoot]);

  const remotePathMap = useMemo(() => {
    const map = new Map<string, string>();
    for (const m of models) {
      if (m.hasRemote && m.remotePath) {
        map.set(m.remotePath, m.remoteHost);
      }
    }
    return map;
  }, [models]);

  const resolveCluster = useCallback((host: string): string => {
    const match = clustersRef.current.find((c) => c.host === host);
    return match?.name ?? "";
  }, []);

  const remoteHost = remotePathMap.get(modelPath);
  if (!remoteHost) {
    return { isRemote: false, clusterName: "" };
  }
  return { isRemote: true, clusterName: resolveCluster(remoteHost) };
}
