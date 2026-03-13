import { useEffect, useMemo, useRef } from "react";
import { useCrucible } from "../context/CrucibleContext";
import { listClusters } from "../api/remoteApi";
import type { ClusterConfig } from "../types/remote";
import type { TrainingMethod } from "../types/training";
import { PRIMARY_MODEL_KEY } from "../types/training";
import type { ClusterSubmitConfig } from "../pages/training/ClusterSubmitSection";

interface ModelLocationInfo {
  locationType: "local" | "remote";
  remoteHost: string;
}

/**
 * Auto-detects training location (local vs remote) from the selected model.
 *
 * For methods with a PRIMARY_MODEL_KEY, watches that field and sets
 * `remoteEnabled` + auto-selects a matching cluster when a remote model
 * is chosen.
 */
export function useTrainingLocation(
  method: TrainingMethod,
  extra: Record<string, string>,
  setRemoteEnabled: (enabled: boolean) => void,
  setClusterConfig: React.Dispatch<React.SetStateAction<ClusterSubmitConfig>>,
): void {
  const { models, dataRoot } = useCrucible();
  const primaryKey = PRIMARY_MODEL_KEY[method];
  const clustersRef = useRef<ClusterConfig[]>([]);

  const pathLocationMap = useMemo(() => {
    const map = new Map<string, ModelLocationInfo>();
    for (const m of models) {
      if (m.modelPath) {
        map.set(m.modelPath, { locationType: "local", remoteHost: "" });
      }
      if (m.remotePath) {
        map.set(m.remotePath, { locationType: "remote", remoteHost: m.remoteHost });
      }
    }
    return map;
  }, [models]);

  // Pre-fetch clusters so we can auto-select the matching one
  useEffect(() => {
    if (!primaryKey || !dataRoot) return;
    listClusters(dataRoot)
      .then((c) => { clustersRef.current = c; })
      .catch(() => { clustersRef.current = []; });
  }, [primaryKey, dataRoot]);

  // Watch primary model field and derive location
  const primaryModelPath = primaryKey ? (extra[primaryKey] ?? "") : "";
  useEffect(() => {
    if (!primaryKey || !primaryModelPath) return;
    const info = pathLocationMap.get(primaryModelPath);
    if (!info) return;

    const isRemote = info.locationType === "remote";
    setRemoteEnabled(isRemote);

    if (isRemote && info.remoteHost) {
      const match = clustersRef.current.find((c) => c.host === info.remoteHost);
      if (match) {
        setClusterConfig((prev) => ({ ...prev, cluster: match.name }));
      }
    }
  }, [primaryKey, primaryModelPath, pathLocationMap, setRemoteEnabled, setClusterConfig]);
}
