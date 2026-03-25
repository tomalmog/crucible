//! Async Tauri commands that shell out to the crucible CLI for remote ops.
//!
//! Each command runs the subprocess on a blocking thread via
//! `spawn_blocking` so the Tauri main thread stays responsive.

use crate::commands::remote_queries::RemoteDatasetSummary;

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

/// Fetch live cluster GPU and node status via sinfo.
///
/// Returns the raw JSON value — TypeScript owns the shape.
#[tauri::command]
pub async fn get_cluster_info(
    data_root: String,
    cluster: String,
) -> Result<serde_json::Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let stdout = run_crucible_cli(
            &data_root,
            &["remote", "cluster-info", "--cluster", &cluster],
        )
        .map_err(|e| format!("cluster-info failed: {e}"))?;
        for line in stdout.lines() {
            if let Some(json_str) = line.strip_prefix("CRUCIBLE_JSON:") {
                let value: serde_json::Value = serde_json::from_str(json_str)
                    .map_err(|e| format!("Failed to parse cluster-info JSON: {e}"))?;
                return Ok(value);
            }
        }
        Err("No CRUCIBLE_JSON line found in cluster-info output".to_string())
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
