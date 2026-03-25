//! Read-only queries for remote clusters and Slurm jobs.
//!
//! Filesystem-only commands live here.  CLI subprocess commands that
//! require SSH are in [`remote_cli_ops`].

use crate::commands::runtime_queries::resolve_data_root_path;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::Path;

// ── Types ──────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClusterSummary {
    pub name: String,
    pub host: String,
    pub user: String,
    pub ssh_port: u16,
    pub default_partition: String,
    pub partitions: Vec<String>,
    pub gpu_types: Vec<String>,
    pub module_loads: Vec<String>,
    pub python_path: String,
    pub remote_workspace: String,
    pub validated_at: String,
    pub backend: String,
    pub docker_image: String,
    pub api_endpoint: String,
    pub api_token: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RemoteDatasetSummary {
    pub name: String,
    #[serde(alias = "size_bytes")]
    #[serde(rename(serialize = "sizeBytes"))]
    pub size_bytes: u64,
    #[serde(alias = "synced_at")]
    #[serde(rename(serialize = "syncedAt"))]
    pub synced_at: String,
}

// ── Filesystem-only commands ───────────────────────────────────────────

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

// ── JSON parsing helpers (pub(crate) for remote_cli_ops) ───────────────

pub(crate) fn parse_cluster(data: &Value) -> Result<ClusterSummary, String> {
    let obj = data
        .as_object()
        .ok_or_else(|| "Cluster entry is not an object".to_string())?;
    Ok(ClusterSummary {
        name: str_field(obj, "name"),
        host: str_field(obj, "host"),
        user: str_field(obj, "user"),
        ssh_port: obj
            .get("ssh_port")
            .and_then(Value::as_u64)
            .unwrap_or(22) as u16,
        default_partition: str_field(obj, "default_partition"),
        partitions: str_array(obj, "partitions"),
        gpu_types: str_array(obj, "gpu_types"),
        module_loads: str_array(obj, "module_loads"),
        python_path: str_field(obj, "python_path"),
        remote_workspace: str_field(obj, "remote_workspace"),
        validated_at: str_field(obj, "validated_at"),
        backend: {
            let b = str_field(obj, "backend");
            if b.is_empty() { "slurm".to_string() } else { b }
        },
        docker_image: str_field(obj, "docker_image"),
        api_endpoint: str_field(obj, "api_endpoint"),
        api_token: str_field(obj, "api_token"),
    })
}

pub(crate) fn read_json_file(path: &Path) -> Result<Value, String> {
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
