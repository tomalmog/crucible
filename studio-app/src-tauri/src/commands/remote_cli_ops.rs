//! Async Tauri commands that shell out to the crucible CLI for remote ops.
//!
//! Each command runs the subprocess on a blocking thread via
//! `spawn_blocking` so the Tauri main thread stays responsive.

use crate::commands::remote_queries::{
    parse_remote_job, read_json_file, RemoteDatasetSummary, RemoteJobSummary,
};
use crate::commands::runtime_queries::resolve_data_root_path;

#[tauri::command]
pub async fn get_remote_job_result(data_root: String, job_id: String) -> Result<String, String> {
    let job_path = resolve_data_root_path(&data_root)
        .join("remote-jobs")
        .join(format!("{job_id}.json"));
    if !job_path.exists() {
        return Err(format!("Remote job '{job_id}' not found"));
    }
    tauri::async_runtime::spawn_blocking(move || {
        let stdout = run_crucible_cli(&data_root, &["remote", "result", "--job-id", &job_id])
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

#[tauri::command]
pub async fn get_remote_job_logs(data_root: String, job_id: String) -> Result<String, String> {
    let job_path = resolve_data_root_path(&data_root)
        .join("remote-jobs")
        .join(format!("{job_id}.json"));
    if !job_path.exists() {
        return Err(format!("Remote job '{job_id}' not found"));
    }
    tauri::async_runtime::spawn_blocking(move || {
        run_crucible_cli(&data_root, &["remote", "logs", "--job-id", &job_id, "--tail", "200"])
            .map_err(|e| format!("Log fetch failed: {e}"))
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))?
}

#[tauri::command]
pub async fn sync_remote_job_status(
    data_root: String,
    job_id: String,
) -> Result<RemoteJobSummary, String> {
    let job_path = resolve_data_root_path(&data_root)
        .join("remote-jobs")
        .join(format!("{job_id}.json"));
    if !job_path.exists() {
        return Err(format!("Remote job '{job_id}' not found"));
    }
    tauri::async_runtime::spawn_blocking(move || {
        run_crucible_cli(&data_root, &["remote", "status", "--job-id", &job_id])
            .map_err(|e| format!("Status sync failed: {e}"))?;
        let data = read_json_file(&job_path)?;
        parse_remote_job(&data)
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))?
}

#[tauri::command]
pub async fn cancel_remote_job(
    data_root: String,
    job_id: String,
) -> Result<RemoteJobSummary, String> {
    let job_path = resolve_data_root_path(&data_root)
        .join("remote-jobs")
        .join(format!("{job_id}.json"));
    if !job_path.exists() {
        return Err(format!("Remote job '{job_id}' not found"));
    }
    tauri::async_runtime::spawn_blocking(move || {
        run_crucible_cli(&data_root, &["remote", "cancel", "--job-id", &job_id])
            .map_err(|e| format!("Cancel failed: {e}"))?;
        let data = read_json_file(&job_path)?;
        parse_remote_job(&data)
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))?
}

#[tauri::command]
pub async fn list_remote_datasets(
    data_root: String,
    cluster: String,
) -> Result<Vec<RemoteDatasetSummary>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let stdout = run_crucible_cli(
            &data_root,
            &["remote", "dataset-list", "--cluster", &cluster],
        )
        .map_err(|e| format!("dataset-list failed: {e}"))?;
        for line in stdout.lines() {
            if let Some(json_str) = line.strip_prefix("CRUCIBLE_JSON:") {
                let items: Vec<RemoteDatasetSummary> = serde_json::from_str(json_str)
                    .map_err(|e| format!("Failed to parse dataset list JSON: {e}"))?;
                return Ok(items);
            }
        }
        Err("No CRUCIBLE_JSON line found in dataset-list output".to_string())
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))?
}

#[tauri::command]
pub async fn push_dataset_to_cluster(
    data_root: String,
    cluster: String,
    dataset: String,
) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        run_crucible_cli(
            &data_root,
            &["remote", "dataset-push", "--cluster", &cluster, "--dataset", &dataset],
        )
        .map_err(|e| format!("dataset-push failed: {e}"))?;
        Ok(())
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))?
}

#[tauri::command]
pub async fn pull_dataset_from_cluster(
    data_root: String,
    cluster: String,
    dataset: String,
) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        run_crucible_cli(
            &data_root,
            &["remote", "dataset-pull", "--cluster", &cluster, "--dataset", &dataset],
        )
        .map_err(|e| format!("dataset-pull failed: {e}"))?;
        Ok(())
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))?
}

#[tauri::command]
pub async fn delete_remote_dataset_cmd(
    data_root: String,
    cluster: String,
    dataset: String,
) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        run_crucible_cli(
            &data_root,
            &["remote", "dataset-delete", "--cluster", &cluster, "--dataset", &dataset],
        )
        .map_err(|e| format!("dataset-delete failed: {e}"))?;
        Ok(())
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))?
}

/// Get sizes of remote models on a cluster.
///
/// Returns a JSON map of { model_name: size_bytes }.
#[tauri::command]
pub async fn get_remote_model_sizes(
    data_root: String,
    cluster: String,
) -> Result<std::collections::HashMap<String, u64>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let stdout = run_crucible_cli(
            &data_root,
            &["model", "remote-sizes", "--cluster", &cluster],
        )
        .map_err(|e| format!("remote-sizes failed: {e}"))?;
        for line in stdout.lines() {
            if let Some(json_str) = line.strip_prefix("CRUCIBLE_JSON:") {
                let map: std::collections::HashMap<String, u64> =
                    serde_json::from_str(json_str)
                        .map_err(|e| format!("Failed to parse remote-sizes JSON: {e}"))?;
                return Ok(map);
            }
        }
        Err("No CRUCIBLE_JSON line found in remote-sizes output".to_string())
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))?
}

/// Run a crucible CLI subprocess and return stdout on success, stderr on error.
fn run_crucible_cli(data_root: &str, args: &[&str]) -> Result<String, String> {
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let venv_binary = workspace_root.join(".venv/bin/crucible");
    let crucible_bin = if venv_binary.exists() {
        venv_binary
    } else {
        std::path::PathBuf::from("crucible")
    };
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
