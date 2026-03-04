//! Read-only queries for remote clusters and Slurm jobs.

use crate::commands::runtime_queries::resolve_data_root_path;
use serde::Serialize;
use serde_json::Value;
use std::fs;
use std::path::Path;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClusterSummary {
    pub name: String,
    pub host: String,
    pub user: String,
    pub default_partition: String,
    pub partitions: Vec<String>,
    pub gpu_types: Vec<String>,
    pub python_path: String,
    pub remote_workspace: String,
    pub validated_at: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RemoteJobSummary {
    pub job_id: String,
    pub slurm_job_id: String,
    pub cluster_name: String,
    pub training_method: String,
    pub state: String,
    pub submitted_at: String,
    pub updated_at: String,
    pub remote_output_dir: String,
    pub remote_log_path: String,
    pub model_path_remote: String,
    pub model_path_local: String,
    pub local_version_id: String,
    pub model_name: String,
    pub is_sweep: bool,
    pub sweep_array_size: u64,
}

#[tauri::command]
pub fn list_clusters(data_root: String) -> Result<Vec<ClusterSummary>, String> {
    let clusters_dir = resolve_data_root_path(&data_root).join("clusters");
    if !clusters_dir.exists() {
        return Ok(vec![]);
    }
    let mut summaries = Vec::new();
    let mut entries: Vec<_> = fs::read_dir(&clusters_dir)
        .map_err(|e| format!("Failed to read clusters dir: {e}"))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "json"))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let data = read_json_file(&entry.path())?;
        summaries.push(parse_cluster(&data)?);
    }
    Ok(summaries)
}

#[tauri::command]
pub fn list_remote_jobs(data_root: String) -> Result<Vec<RemoteJobSummary>, String> {
    let jobs_dir = resolve_data_root_path(&data_root).join("remote-jobs");
    if !jobs_dir.exists() {
        return Ok(vec![]);
    }
    let mut summaries = Vec::new();
    for entry in fs::read_dir(&jobs_dir)
        .map_err(|e| format!("Failed to read remote-jobs dir: {e}"))?
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().map_or(true, |ext| ext != "json") {
            continue;
        }
        let data = read_json_file(&path)?;
        summaries.push(parse_remote_job(&data)?);
    }
    summaries.sort_by(|a, b| b.submitted_at.cmp(&a.submitted_at));
    Ok(summaries)
}

#[tauri::command]
pub fn get_remote_job(data_root: String, job_id: String) -> Result<RemoteJobSummary, String> {
    let job_path = resolve_data_root_path(&data_root)
        .join("remote-jobs")
        .join(format!("{job_id}.json"));
    if !job_path.exists() {
        return Err(format!("Remote job '{job_id}' not found"));
    }
    let data = read_json_file(&job_path)?;
    parse_remote_job(&data)
}

#[tauri::command]
pub fn get_remote_job_logs(data_root: String, job_id: String) -> Result<String, String> {
    let job_path = resolve_data_root_path(&data_root)
        .join("remote-jobs")
        .join(format!("{job_id}.json"));
    if !job_path.exists() {
        return Err(format!("Remote job '{job_id}' not found"));
    }
    // Call the Python CLI to SSH and fetch logs from the remote cluster
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let venv_binary = workspace_root.join(".venv/bin/forge");
    let forge_bin = if venv_binary.exists() {
        venv_binary
    } else {
        std::path::PathBuf::from("forge")
    };
    let output = std::process::Command::new(&forge_bin)
        .current_dir(&workspace_root)
        .env("PYTHONUNBUFFERED", "1")
        .arg("--data-root")
        .arg(&data_root)
        .args(["remote", "logs", "--job-id", &job_id, "--tail", "200"])
        .output()
        .map_err(|e| format!("Failed to run forge CLI: {e}"))?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(stdout)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        Err(format!("Log fetch failed: {stderr}"))
    }
}

#[tauri::command]
pub fn sync_remote_job_status(data_root: String, job_id: String) -> Result<RemoteJobSummary, String> {
    let job_path = resolve_data_root_path(&data_root)
        .join("remote-jobs")
        .join(format!("{job_id}.json"));
    if !job_path.exists() {
        return Err(format!("Remote job '{job_id}' not found"));
    }
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let venv_binary = workspace_root.join(".venv/bin/forge");
    let forge_bin = if venv_binary.exists() {
        venv_binary
    } else {
        std::path::PathBuf::from("forge")
    };
    let output = std::process::Command::new(&forge_bin)
        .current_dir(&workspace_root)
        .env("PYTHONUNBUFFERED", "1")
        .arg("--data-root")
        .arg(&data_root)
        .args(["remote", "status", "--job-id", &job_id])
        .output()
        .map_err(|e| format!("Failed to run forge CLI: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(format!("Status sync failed: {stderr}"));
    }
    // Re-read the updated JSON file
    let data = read_json_file(&job_path)?;
    parse_remote_job(&data)
}

#[tauri::command]
pub fn delete_remote_job(data_root: String, job_id: String) -> Result<(), String> {
    let job_path = resolve_data_root_path(&data_root)
        .join("remote-jobs")
        .join(format!("{job_id}.json"));
    if !job_path.exists() {
        return Err(format!("Remote job '{job_id}' not found"));
    }
    fs::remove_file(&job_path)
        .map_err(|e| format!("Failed to delete {}: {e}", job_path.display()))
}

fn parse_cluster(data: &Value) -> Result<ClusterSummary, String> {
    let obj = data
        .as_object()
        .ok_or_else(|| "Cluster entry is not an object".to_string())?;
    Ok(ClusterSummary {
        name: str_field(obj, "name"),
        host: str_field(obj, "host"),
        user: str_field(obj, "user"),
        default_partition: str_field(obj, "default_partition"),
        partitions: str_array(obj, "partitions"),
        gpu_types: str_array(obj, "gpu_types"),
        python_path: str_field(obj, "python_path"),
        remote_workspace: str_field(obj, "remote_workspace"),
        validated_at: str_field(obj, "validated_at"),
    })
}

fn parse_remote_job(data: &Value) -> Result<RemoteJobSummary, String> {
    let obj = data
        .as_object()
        .ok_or_else(|| "Remote job entry is not an object".to_string())?;
    Ok(RemoteJobSummary {
        job_id: str_field(obj, "job_id"),
        slurm_job_id: str_field(obj, "slurm_job_id"),
        cluster_name: str_field(obj, "cluster_name"),
        training_method: str_field(obj, "training_method"),
        state: str_field(obj, "state"),
        submitted_at: str_field(obj, "submitted_at"),
        updated_at: str_field(obj, "updated_at"),
        remote_output_dir: str_field(obj, "remote_output_dir"),
        remote_log_path: str_field(obj, "remote_log_path"),
        model_path_remote: str_field(obj, "model_path_remote"),
        model_path_local: str_field(obj, "model_path_local"),
        local_version_id: str_field(obj, "local_version_id"),
        model_name: str_field(obj, "model_name"),
        is_sweep: obj
            .get("is_sweep")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        sweep_array_size: obj
            .get("sweep_array_size")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    })
}

fn read_json_file(path: &Path) -> Result<Value, String> {
    let payload = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    serde_json::from_str(&payload)
        .map_err(|e| format!("Failed to parse {}: {e}", path.display()))
}

fn str_field(map: &serde_json::Map<String, Value>, key: &str) -> String {
    map.get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn str_array(map: &serde_json::Map<String, Value>, key: &str) -> Vec<String> {
    map.get(key)
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}
