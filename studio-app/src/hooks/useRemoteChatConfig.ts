import { useEffect, useState } from "react";
import { listClusters } from "../api/remoteApi";
import type { ClusterConfig } from "../types/remote";

export interface RemoteInferenceConfig {
  partition: string;
  gpuType: string;
  memory: string;
  timeLimit: string;
}

interface RemoteConfigState {
  clusterInfo: ClusterConfig | null;
  isSlurm: boolean;
  config: RemoteInferenceConfig;
  setPartition: (v: string) => void;
  setGpuType: (v: string) => void;
  setMemory: (v: string) => void;
  setTimeLimit: (v: string) => void;
}

export function useRemoteChatConfig(
  dataRoot: string,
  remoteHost: string,
  isRemote: boolean,
): RemoteConfigState {
  const [clusterInfo, setClusterInfo] = useState<ClusterConfig | null>(null);
  const [partition, setPartition] = useState("");
  const [gpuType, setGpuType] = useState("");
  const [memory, setMemory] = useState("16G");
  const [timeLimit, setTimeLimit] = useState("00:30:00");

  // Load cluster details when a remote model is selected so
  // partition and GPU type dropdowns can be populated.
  useEffect(() => {
    if (!isRemote || !dataRoot) {
      setClusterInfo(null);
      return;
    }
    listClusters(dataRoot).then((clusters) => {
      const match = clusters.find((c) => c.host === remoteHost) ?? null;
      setClusterInfo(match);
      if (match?.defaultPartition) setPartition(match.defaultPartition);
    });
  }, [isRemote, remoteHost, dataRoot]);

  return {
    clusterInfo,
    isSlurm: clusterInfo?.backend === "slurm",
    config: { partition, gpuType, memory, timeLimit },
    setPartition,
    setGpuType,
    setMemory,
    setTimeLimit,
  };
}
