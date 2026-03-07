import { useEffect, useMemo, useRef } from "react";
import { useForge } from "../context/ForgeContext";
import { listClusters } from "../api/remoteApi";
import type { ClusterConfig } from "../types/remote";
import type { TrainingMethod } from "../types/training";
import { AUTO_LOCATION_METHODS, PRIMARY_MODEL_KEY } from "../types/training";
import type { ClusterMode, ClusterSubmitConfig } from "../pages/training/ClusterSubmitSection";

interface ModelLocationInfo {
  locationType: "local" | "remote";
  remoteHost: string;
}

/**
 * Derives training location (local vs remote) from the selected model.
 *
 * For auto-location methods, watches the primary model field in `extra` and
 * sets `remoteEnabled` + auto-selects a matching cluster. For manual-toggle
 * methods (train, rlhf-train, distill), returns `clusterMode: "toggle"`.
 */
export function useTrainingLocation(
  method: TrainingMethod,
  extra: Record<string, string>,
  setRemoteEnabled: (enabled: boolean) => void,
  setClusterConfig: React.Dispatch<React.SetStateAction<ClusterSubmitConfig>>,
): { clusterMode: ClusterMode } {
  const { modelGroups, dataRoot } = useForge();
  const isAutoMethod = AUTO_LOCATION_METHODS.has(method);
  const primaryKey = PRIMARY_MODEL_KEY[method];
  const clustersRef = useRef<ClusterConfig[]>([]);

  const pathLocationMap = useMemo(() => {
    const map = new Map<string, ModelLocationInfo>();
    for (const g of modelGroups) {
      if (g.activeModelPath) {
        map.set(g.activeModelPath, { locationType: "local", remoteHost: "" });
      }
      if (g.activeRemotePath) {
        map.set(g.activeRemotePath, { locationType: "remote", remoteHost: g.activeRemoteHost });
      }
    }
    return map;
  }, [modelGroups]);

  // Pre-fetch clusters so we can auto-select the matching one
  useEffect(() => {
    if (!isAutoMethod || !dataRoot) return;
    listClusters(dataRoot)
      .then((c) => { clustersRef.current = c; })
      .catch(() => { clustersRef.current = []; });
  }, [isAutoMethod, dataRoot]);

  // Watch primary model field and derive location
  const primaryModelPath = primaryKey ? (extra[primaryKey] ?? "") : "";
  useEffect(() => {
    if (!isAutoMethod || !primaryModelPath) return;
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
  }, [isAutoMethod, primaryModelPath, pathLocationMap, setRemoteEnabled, setClusterConfig]);

  return { clusterMode: isAutoMethod ? "auto" : "toggle" };
}
