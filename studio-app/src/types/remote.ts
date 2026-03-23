export interface ClusterConfig {
  name: string;
  host: string;
  user: string;
  defaultPartition: string;
  partitions: string[];
  gpuTypes: string[];
  moduleLoads: string[];
  pythonPath: string;
  remoteWorkspace: string;
  validatedAt: string;
}

export interface RemoteDatasetInfo {
  name: string;
  sizeBytes: number;
  syncedAt: string;
}

export type RemoteJobState =
  | "pending"
  | "submitting"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface RemoteJobRecord {
  jobId: string;
  slurmJobId: string;
  clusterName: string;
  trainingMethod: string;
  state: RemoteJobState;
  submittedAt: string;
  updatedAt: string;
  remoteOutputDir: string;
  remoteLogPath: string;
  modelPathRemote: string;
  modelPathLocal: string;
  localVersionId: string;
  modelName: string;
  isSweep: boolean;
  sweepArraySize: number;
  submitPhase: string;
}

export interface ClusterValidationResult {
  clusterName: string;
  pythonOk: boolean;
  pythonVersion: string;
  torchOk: boolean;
  torchVersion: string;
  cudaOk: boolean;
  cudaVersion: string;
  slurmOk: boolean;
  partitions: string[];
  gpuTypes: string[];
  errors: string[];
}

export interface SlurmResourceConfig {
  partition: string;
  nodes: number;
  gpusPerNode: number;
  gpuType: string;
  cpusPerTask: number;
  memory: string;
  timeLimit: string;
}

export type NodeState = "idle" | "mixed" | "allocated" | "drained" | "down" | "unknown";

export interface PartitionInfo {
  name: string;
  isDefault: boolean;
  state: "up" | "down" | "inactive";
  timeLimit: string;
  totalNodes: number;
  nodesByState: Record<NodeState, number>;
  totalGpus: number;
  idleGpus: number;
  gpuConfig: string;
  memoryMb: number;
  cpusPerNode: number;
}

export interface ClusterInfo {
  clusterName: string;
  isConnected: boolean;
  partitions: PartitionInfo[];
  totalGpus: number;
  idleGpus: number;
  gpuUtilizationPct: number;
  healthyNodes: number;
  drainedNodes: number;
  downNodes: number;
  totalNodes: number;
  fetchedAt: string;
}
