//! Shared serialization models for Studio commands.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Serialize)]
pub struct DatasetDashboard {
    pub dataset_name: String,
    pub record_count: u64,
    pub average_quality: f64,
    pub min_quality: f64,
    pub max_quality: f64,
    pub language_counts: BTreeMap<String, u64>,
    pub source_counts: Vec<SourceCount>,
}

#[derive(Debug, Serialize)]
pub struct SourceCount {
    pub source: String,
    pub count: u64,
}

#[derive(Debug, Serialize)]
pub struct RecordSample {
    pub record_id: String,
    pub source_uri: String,
    pub language: String,
    pub quality_score: f64,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct CommandTaskStart {
    pub task_id: String,
    pub estimated_total_seconds: u64,
}

#[derive(Debug, Serialize)]
pub struct CommandTaskStatus {
    pub task_id: String,
    pub status: String,
    pub command: String,
    pub args: Vec<String>,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub elapsed_seconds: u64,
    pub estimated_total_seconds: u64,
    pub remaining_seconds: u64,
    pub progress_percent: f64,
    pub label: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingEpoch {
    pub epoch: u64,
    pub train_loss: f64,
    pub validation_loss: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingBatchLoss {
    pub epoch: u64,
    pub batch_index: u64,
    pub global_step: u64,
    pub train_loss: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingHistory {
    pub epochs: Vec<TrainingEpoch>,
    #[serde(default)]
    pub batch_losses: Vec<TrainingBatchLoss>,
}

#[derive(Debug, Serialize)]
pub struct TrainingRunSummary {
    pub run_id: String,
    pub dataset_name: String,
    pub state: String,
    pub updated_at: String,
    pub output_dir: String,
    pub artifact_contract_path: Option<String>,
    pub model_path: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct LineageRunNode {
    pub run_id: String,
    pub dataset_name: String,
    pub output_dir: String,
    pub parent_model_path: Option<String>,
    pub model_path: Option<String>,
    pub config_hash: String,
    pub created_at: String,
    pub artifact_contract_path: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct LineageEdge {
    pub from: String,
    pub to: String,
    #[serde(rename = "type")]
    pub edge_type: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelEntrySummary {
    pub model_name: String,
    pub model_path: String,
    pub run_id: Option<String>,
    pub created_at: String,
    pub location_type: String,
    pub has_local: bool,
    pub has_remote: bool,
    pub remote_host: String,
    pub remote_path: String,
    pub size_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct LineageGraphSummary {
    pub run_count: u64,
    pub edge_count: u64,
    pub runs: Vec<LineageRunNode>,
    pub edges: Vec<LineageEdge>,
}
