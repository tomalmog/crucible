export interface ClusterConfig {
  name: string;
  host: string;
  user: string;
  defaultPartition: string;
  partitions: string[];
  gpuTypes: string[];
  pythonPath: string;
  remoteWorkspace: string;
  validatedAt: string;
}

export interface RemoteDatasetInfo {
  name: string;
  recordCount: number;
  versionId: string;
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

export type DataStrategy = "scp" | "shared" | "s3";

export interface SlurmResourceConfig {
  partition: string;
  nodes: number;
  gpusPerNode: number;
  gpuType: string;
  cpusPerTask: number;
  memory: string;
  timeLimit: string;
}
