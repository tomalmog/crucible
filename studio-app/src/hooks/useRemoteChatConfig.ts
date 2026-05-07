import { useEffect, useMemo, useState } from "react";
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

const REMOTE_CHAT_CONFIG_KEY = "crucible_remote_chat_config_v1";
const DEFAULT_REMOTE_CHAT_CONFIG: RemoteInferenceConfig = {
  partition: "",
  gpuType: "",
  memory: "16G",
  timeLimit: "00:30:00",
};

export function useRemoteChatConfig(
  dataRoot: string,
  remoteHost: string,
  isRemote: boolean,
): RemoteConfigState {
  const initialConfig = useMemo(loadRemoteChatConfig, []);
  const [clusterInfo, setClusterInfo] = useState<ClusterConfig | null>(null);
  const [partition, setPartition] = useState(initialConfig.partition);
  const [gpuType, setGpuType] = useState(initialConfig.gpuType);
  const [memory, setMemory] = useState(initialConfig.memory);
  const [timeLimit, setTimeLimit] = useState(initialConfig.timeLimit);

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

  // Persist remote chat resource settings so the chat screen restores after restart.
  useEffect(() => {
    saveRemoteChatConfig({ partition, gpuType, memory, timeLimit });
  }, [partition, gpuType, memory, timeLimit]);

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

function loadRemoteChatConfig(): RemoteInferenceConfig {
  if (typeof window === "undefined") return DEFAULT_REMOTE_CHAT_CONFIG;
  const rawValue = window.localStorage.getItem(REMOTE_CHAT_CONFIG_KEY);
  if (!rawValue) return DEFAULT_REMOTE_CHAT_CONFIG;
  try {
    const parsed = JSON.parse(rawValue);
    if (!isRecord(parsed)) return DEFAULT_REMOTE_CHAT_CONFIG;
    return {
      partition: asString(parsed.partition, DEFAULT_REMOTE_CHAT_CONFIG.partition),
      gpuType: asString(parsed.gpuType, DEFAULT_REMOTE_CHAT_CONFIG.gpuType),
      memory: asString(parsed.memory, DEFAULT_REMOTE_CHAT_CONFIG.memory),
      timeLimit: asString(parsed.timeLimit, DEFAULT_REMOTE_CHAT_CONFIG.timeLimit),
    };
  } catch {
    return DEFAULT_REMOTE_CHAT_CONFIG;
  }
}

function saveRemoteChatConfig(config: RemoteInferenceConfig): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(REMOTE_CHAT_CONFIG_KEY, JSON.stringify(config));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asString(value: unknown, fallback: string): string {
  return typeof value === "string" ? value : fallback;
}
