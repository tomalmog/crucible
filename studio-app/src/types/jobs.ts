/** Unified job types matching Python core.job_types. */

export type BackendKind = "local" | "slurm" | "docker-ssh" | "http-api";
export type JobState =
  | "pending"
  | "submitting"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export const TERMINAL_JOB_STATES = new Set<JobState>([
  "completed",
  "failed",
  "cancelled",
]);

export interface JobRecord {
  jobId: string;
  backend: BackendKind;
  jobType: string;
  state: JobState;
  createdAt: string;
  updatedAt: string;
  label: string;
  backendJobId: string;
  backendCluster: string;
  backendOutputDir: string;
  backendLogPath: string;
  modelPath: string;
  modelPathLocal: string;
  modelName: string;
  errorMessage: string;
  progressPercent: number;
  submitPhase: string;
  isSweep: boolean;
  sweepTrialCount: number;
}

export interface ResourceConfig {
  partition: string;
  nodes: number;
  gpus_per_node: number;
  cpus_per_task: number;
  memory: string;
  time_limit: string;
  gpu_type: string;
}
