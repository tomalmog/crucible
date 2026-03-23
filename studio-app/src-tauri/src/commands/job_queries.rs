//! Unified job queries — reads from .crucible/jobs/ directory.
//!
//! Backend-specific operations (sync, cancel, logs, result) shell out
//! to the `crucible job` CLI subcommand.

use crate::commands::remote_queries::read_json_file;
use crate::commands::runtime_queries::resolve_data_root_path;
use serde::Serialize;
use serde_json::Value;
use std::fs;

// ── Types ──────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct JobRecordSummary {
    pub job_id: String,
    pub backend: String,
    pub job_type: String,
    pub state: String,
    pub created_at: String,
    pub updated_at: String,
    pub label: String,
    pub backend_job_id: String,
    pub backend_cluster: String,
    pub backend_output_dir: String,
    pub backend_log_path: String,
    pub model_path: String,
    pub model_path_local: String,
    pub model_name: String,
    pub error_message: String,
    pub progress_percent: f64,
    pub submit_phase: String,
    pub is_sweep: bool,
    pub sweep_trial_count: u64,
}

// ── Filesystem-only commands ───────────────────────────────────────────

#[tauri::command]
pub fn list_unified_jobs(data_root: String) -> Result<Vec<JobRecordSummary>, String> {
    let jobs_dir = resolve_data_root_path(&data_root).join("jobs");
    if !jobs_dir.exists() {
        return Ok(vec![]);
    }
    let mut summaries = Vec::new();
    for entry in fs::read_dir(&jobs_dir)
        .map_err(|e| format!("Failed to read jobs dir: {e}"))?
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().map_or(true, |ext| ext != "json") {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with("job-") && !name.starts_with("crucible-task-") && !name.starts_with("rj-") {
            continue;
        }
        let data = read_json_file(&path)?;
        summaries.push(parse_job_record(&data)?);
    }
    summaries.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(summaries)
}

#[tauri::command]
pub fn get_unified_job(data_root: String, job_id: String) -> Result<JobRecordSummary, String> {
    let job_path = resolve_data_root_path(&data_root)
        .join("jobs")
        .join(format!("{job_id}.json"));
    if !job_path.exists() {
        return Err(format!("Job '{job_id}' not found"));
    }
    let data = read_json_file(&job_path)?;
    parse_job_record(&data)
}

#[tauri::command]
pub fn delete_unified_job(data_root: String, job_id: String) -> Result<(), String> {
    let job_path = resolve_data_root_path(&data_root)
        .join("jobs")
        .join(format!("{job_id}.json"));
    if !job_path.exists() {
        return Err(format!("Job '{job_id}' not found"));
    }
    fs::remove_file(&job_path)
        .map_err(|e| format!("Failed to delete {}: {e}", job_path.display()))
}

// ── CLI-backed async commands ──────────────────────────────────────────

#[tauri::command]
pub async fn sync_unified_job_state(
    data_root: String,
    job_id: String,
) -> Result<JobRecordSummary, String> {
    let job_path = resolve_data_root_path(&data_root)
        .join("jobs")
        .join(format!("{job_id}.json"));
    if !job_path.exists() {
        return Err(format!("Job '{job_id}' not found"));
    }
    tauri::async_runtime::spawn_blocking(move || {
        run_crucible_cli(&data_root, &["job", "sync", "--job-id", &job_id])
            .map_err(|e| format!("Job sync failed: {e}"))?;
        let data = read_json_file(&job_path)?;
        parse_job_record(&data)
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))?
}

#[tauri::command]
pub async fn cancel_unified_job(
    data_root: String,
    job_id: String,
) -> Result<JobRecordSummary, String> {
    let job_path = resolve_data_root_path(&data_root)
        .join("jobs")
        .join(format!("{job_id}.json"));
    if !job_path.exists() {
        return Err(format!("Job '{job_id}' not found"));
    }
    tauri::async_runtime::spawn_blocking(move || {
        run_crucible_cli(&data_root, &["job", "cancel", "--job-id", &job_id])
            .map_err(|e| format!("Job cancel failed: {e}"))?;
        let data = read_json_file(&job_path)?;
        parse_job_record(&data)
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))?
}

#[tauri::command]
pub async fn get_unified_job_logs(
    data_root: String,
    job_id: String,
) -> Result<String, String> {
    let job_path = resolve_data_root_path(&data_root)
        .join("jobs")
        .join(format!("{job_id}.json"));
    if !job_path.exists() {
        return Err(format!("Job '{job_id}' not found"));
    }
    tauri::async_runtime::spawn_blocking(move || {
        run_crucible_cli(&data_root, &["job", "logs", "--job-id", &job_id])
            .map_err(|e| format!("Log fetch failed: {e}"))
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))?
}

#[tauri::command]
pub async fn get_unified_job_result(
    data_root: String,
    job_id: String,
) -> Result<String, String> {
    let job_path = resolve_data_root_path(&data_root)
        .join("jobs")
        .join(format!("{job_id}.json"));
    if !job_path.exists() {
        return Err(format!("Job '{job_id}' not found"));
    }
    tauri::async_runtime::spawn_blocking(move || {
        let stdout = run_crucible_cli(&data_root, &["job", "result", "--job-id", &job_id])
            .map_err(|e| format!("Result fetch failed: {e}"))?;
        for line in stdout.lines() {
            if let Some(json_str) = line.strip_prefix("CRUCIBLE_JSON:") {
                return Ok(json_str.to_string());
            }
        }
        Err("No CRUCIBLE_JSON line found in result output".to_string())
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))?
}

// ── Helpers ────────────────────────────────────────────────────────────

fn parse_job_record(data: &Value) -> Result<JobRecordSummary, String> {
    let obj = data
        .as_object()
        .ok_or_else(|| "Job entry is not an object".to_string())?;
    Ok(JobRecordSummary {
        job_id: str_field(obj, "job_id"),
        backend: str_field(obj, "backend"),
        job_type: str_field(obj, "job_type"),
        state: str_field(obj, "state"),
        created_at: str_field(obj, "created_at"),
        updated_at: str_field(obj, "updated_at"),
        label: str_field(obj, "label"),
        backend_job_id: str_field(obj, "backend_job_id"),
        backend_cluster: str_field(obj, "backend_cluster"),
        backend_output_dir: str_field(obj, "backend_output_dir"),
        backend_log_path: str_field(obj, "backend_log_path"),
        model_path: str_field(obj, "model_path"),
        model_path_local: str_field(obj, "model_path_local"),
        model_name: str_field(obj, "model_name"),
        error_message: str_field(obj, "error_message"),
        progress_percent: obj
            .get("progress_percent")
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        submit_phase: str_field(obj, "submit_phase"),
        is_sweep: obj
            .get("is_sweep")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        sweep_trial_count: obj
            .get("sweep_trial_count")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    })
}

fn str_field(map: &serde_json::Map<String, Value>, key: &str) -> String {
    map.get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn run_crucible_cli(data_root: &str, args: &[&str]) -> Result<String, String> {
    let workspace_root = super::crucible_task_store::workspace_root_dir();
    let crucible_bin = super::crucible_task_store::resolve_crucible_binary(&workspace_root);
    let output = std::process::Command::new(&crucible_bin)
        .current_dir(&workspace_root)
        .env("PYTHONUNBUFFERED", "1")
        .arg("--data-root")
        .arg(data_root)
        .args(args)
        .output()
        .map_err(|e| format!("Failed to run crucible CLI: {e}"))?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}
