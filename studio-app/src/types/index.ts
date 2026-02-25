export interface VersionSummary {
  version_id: string;
  record_count: number;
  created_at: string;
  parent_version: string | null;
}

export interface SourceCount {
  source: string;
  count: number;
}

export interface DatasetDashboard {
  dataset_name: string;
  version_id: string;
  record_count: number;
  average_quality: number;
  min_quality: number;
  max_quality: number;
  language_counts: Record<string, number>;
  source_counts: SourceCount[];
}

export interface RecordSample {
  record_id: string;
  source_uri: string;
  language: string;
  quality_score: number;
  text: string;
}

export interface VersionDiff {
  dataset_name: string;
  base_version: string;
  target_version: string;
  added_records: number;
  removed_records: number;
  shared_records: number;
}

export interface CommandTaskStart {
  task_id: string;
  estimated_total_seconds: number;
}

export interface CommandTaskStatus {
  task_id: string;
  status: "running" | "completed" | "failed";
  command: string;
  args: string[];
  exit_code: number | null;
  stdout: string;
  stderr: string;
  elapsed_seconds: number;
  estimated_total_seconds: number;
  remaining_seconds: number;
  progress_percent: number;
  label: string | null;
}

export interface TrainingEpoch {
  epoch: number;
  train_loss: number;
  validation_loss: number;
}

export interface TrainingBatchLoss {
  epoch: number;
  batch_index: number;
  global_step: number;
  train_loss: number;
}

export interface TrainingHistory {
  epochs: TrainingEpoch[];
  batch_losses: TrainingBatchLoss[];
}

export interface TrainingRunSummary {
  run_id: string;
  dataset_name: string;
  dataset_version_id: string;
  state: string;
  updated_at: string;
  output_dir: string;
  artifact_contract_path: string | null;
  model_path: string | null;
}

export interface LineageRunNode {
  run_id: string;
  dataset_name: string;
  dataset_version_id: string;
  output_dir: string;
  parent_model_path: string | null;
  model_path: string | null;
  config_hash: string;
  created_at: string;
  artifact_contract_path: string | null;
}

export interface LineageEdge {
  from: string;
  to: string;
  type: string;
}

export interface LineageGraphSummary {
  run_count: number;
  edge_count: number;
  runs: LineageRunNode[];
  edges: LineageEdge[];
}
