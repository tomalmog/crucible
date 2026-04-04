export interface DatasetEntry {
  name: string;
  sizeBytes: number;
}

export interface SourceCount {
  source: string;
  count: number;
}

export interface DatasetDashboard {
  dataset_name: string;
  record_count: number;
  average_quality: number;
  min_quality: number;
  max_quality: number;
  language_counts: Record<string, number>;
  source_counts: SourceCount[];
  avg_token_length: number;
  min_token_length: number;
  max_token_length: number;
  field_names: string[];
}

export interface RecordSample {
  record_id: string;
  source_uri: string;
  language: string;
  quality_score: number;
  text: string;
  extra_fields: Record<string, string>;
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
  state: string;
  updated_at: string;
  output_dir: string;
  artifact_contract_path: string | null;
  model_path: string | null;
}

export interface LineageRunNode {
  run_id: string;
  dataset_name: string;
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
