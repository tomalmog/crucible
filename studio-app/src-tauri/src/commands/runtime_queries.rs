//! Runtime metadata commands for lifecycle, lineage, and hardware profile views.

use crate::models::{LineageEdge, LineageGraphSummary, LineageRunNode, TrainingRunSummary};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[tauri::command]
pub fn list_training_runs(data_root: String) -> Result<Vec<TrainingRunSummary>, String> {
    let resolved_data_root = resolve_data_root_path(&data_root);
    let runs_root = resolved_data_root.join("runs");
    let index_path = runs_root.join("index.json");
    if !index_path.exists() {
        return Ok(vec![]);
    }
    let index_payload = read_json_file(&index_path)?;
    let run_ids = index_payload
        .get("runs")
        .and_then(Value::as_array)
        .ok_or_else(|| "Run index is missing runs array".to_string())?;
    let model_paths = load_lineage_model_paths(&resolved_data_root)?;
    let mut rows = Vec::new();
    for run_id_value in run_ids.iter().rev() {
        let run_id = run_id_value
            .as_str()
            .ok_or_else(|| "Run id entry must be a string".to_string())?;
        let lifecycle_path = runs_root.join(run_id).join("lifecycle.json");
        if !lifecycle_path.exists() {
            continue;
        }
        let payload = read_json_file(&lifecycle_path)?;
        let object = payload
            .as_object()
            .ok_or_else(|| "Lifecycle payload must be a JSON object".to_string())?;
        rows.push(TrainingRunSummary {
            run_id: run_id.to_string(),
            dataset_name: required_string(object, "dataset_name")?,
            state: required_string(object, "state")?,
            updated_at: required_string(object, "updated_at")?,
            output_dir: required_string(object, "output_dir")?,
            artifact_contract_path: optional_string(object, "artifact_contract_path"),
            model_path: model_paths.get(run_id).cloned(),
        });
    }
    Ok(rows)
}

#[tauri::command]
pub fn get_lineage_graph(data_root: String) -> Result<LineageGraphSummary, String> {
    let resolved_data_root = resolve_data_root_path(&data_root);
    let graph_path = resolved_data_root.join("lineage").join("model_lineage.json");
    if !graph_path.exists() {
        return Ok(empty_lineage_graph());
    }
    let payload = read_json_file(&graph_path)?;
    let root = payload
        .as_object()
        .ok_or_else(|| "Lineage payload must be a JSON object".to_string())?;
    let runs_map = root
        .get("runs")
        .and_then(Value::as_object)
        .ok_or_else(|| "Lineage payload missing runs map".to_string())?;
    let edges_rows = root
        .get("edges")
        .and_then(Value::as_array)
        .ok_or_else(|| "Lineage payload missing edges array".to_string())?;
    let mut runs = Vec::with_capacity(runs_map.len());
    for (run_id, raw_run) in runs_map {
        let run_payload = raw_run
            .as_object()
            .ok_or_else(|| "Lineage run payload must be object".to_string())?;
        runs.push(LineageRunNode {
            run_id: run_id.to_string(),
            dataset_name: required_string(run_payload, "dataset_name")?,
            output_dir: required_string(run_payload, "output_dir")?,
            parent_model_path: optional_string(run_payload, "parent_model_path"),
            model_path: optional_string(run_payload, "model_path"),
            config_hash: required_string(run_payload, "config_hash")?,
            created_at: required_string(run_payload, "created_at")?,
            artifact_contract_path: optional_string(run_payload, "artifact_contract_path"),
        });
    }
    runs.sort_by(|left, right| right.created_at.cmp(&left.created_at));
    let mut edges = Vec::with_capacity(edges_rows.len());
    for edge_row in edges_rows {
        let edge = edge_row
            .as_object()
            .ok_or_else(|| "Lineage edge row must be object".to_string())?;
        edges.push(LineageEdge {
            from: required_string(edge, "from")?,
            to: required_string(edge, "to")?,
            edge_type: required_string(edge, "type")?,
        });
    }
    Ok(LineageGraphSummary {
        run_count: runs.len() as u64,
        edge_count: edges.len() as u64,
        runs,
        edges,
    })
}

#[tauri::command]
pub fn get_hardware_profile(data_root: String) -> Result<BTreeMap<String, String>, String> {
    let resolved_data_root = resolve_data_root_path(&data_root);
    let workspace_root = workspace_root_dir();
    let crucible_bin = super::crucible_task_store::resolve_crucible_binary(&workspace_root);
    let output = Command::new(&crucible_bin)
        .current_dir(&workspace_root)
        .arg("--data-root")
        .arg(resolved_data_root.as_os_str())
        .arg("hardware-profile")
        .output()
        .map_err(|error| format!("Failed to run crucible hardware-profile: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(format!(
            "crucible hardware-profile failed with status {}: {}",
            output.status.code().unwrap_or(-1),
            stderr
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let profile = parse_hardware_profile_output(&stdout);
    if profile.is_empty() {
        return Err("Hardware profile output was empty".to_string());
    }
    Ok(profile)
}

fn load_lineage_model_paths(data_root: &Path) -> Result<HashMap<String, String>, String> {
    let graph_path = data_root.join("lineage").join("model_lineage.json");
    if !graph_path.exists() {
        return Ok(HashMap::new());
    }
    let payload = read_json_file(&graph_path)?;
    let root = payload
        .as_object()
        .ok_or_else(|| "Lineage payload must be a JSON object".to_string())?;
    let runs_map = root
        .get("runs")
        .and_then(Value::as_object)
        .ok_or_else(|| "Lineage payload missing runs map".to_string())?;
    let mut model_paths = HashMap::new();
    for (run_id, raw_run) in runs_map {
        let run_payload = raw_run
            .as_object()
            .ok_or_else(|| "Lineage run payload must be object".to_string())?;
        if let Some(model_path) = optional_string(run_payload, "model_path") {
            model_paths.insert(run_id.to_string(), model_path);
        }
    }
    Ok(model_paths)
}

fn parse_hardware_profile_output(stdout: &str) -> BTreeMap<String, String> {
    let mut profile = BTreeMap::new();
    for line in stdout.lines() {
        if let Some((key, value)) = line.split_once('=') {
            let normalized_key = key.trim();
            let normalized_value = value.trim();
            if !normalized_key.is_empty() {
                profile.insert(normalized_key.to_string(), normalized_value.to_string());
            }
        }
    }
    profile
}

fn required_string(payload: &Map<String, Value>, key: &str) -> Result<String, String> {
    payload
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("Field '{key}' is missing or invalid"))
}

fn optional_string(payload: &Map<String, Value>, key: &str) -> Option<String> {
    payload.get(key).and_then(Value::as_str).map(str::to_string)
}

fn empty_lineage_graph() -> LineageGraphSummary {
    LineageGraphSummary {
        run_count: 0,
        edge_count: 0,
        runs: vec![],
        edges: vec![],
    }
}

fn read_json_file(payload_path: &Path) -> Result<Value, String> {
    let payload = fs::read_to_string(payload_path).map_err(|error| {
        format!(
            "Failed to read JSON file {}: {error}",
            payload_path.to_string_lossy()
        )
    })?;
    serde_json::from_str::<Value>(&payload).map_err(|error| {
        format!(
            "Failed to parse JSON file {}: {error}",
            payload_path.to_string_lossy()
        )
    })
}

fn workspace_root_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn resolve_data_root_path(data_root: &str) -> PathBuf {
    let raw_path = Path::new(data_root);
    if raw_path.is_absolute() {
        return raw_path.to_path_buf();
    }
    workspace_root_dir().join(raw_path)
}

#[cfg(test)]
mod tests {
    use super::{parse_hardware_profile_output, resolve_data_root_path};
    use std::path::Path;

    #[test]
    fn parse_hardware_profile_output_reads_key_value_rows() {
        let profile = parse_hardware_profile_output(
            "accelerator=cuda\ngpu_count=1\nrecommended_precision_mode=bf16\n",
        );
        assert_eq!(profile.get("accelerator"), Some(&"cuda".to_string()));
        assert_eq!(profile.get("gpu_count"), Some(&"1".to_string()));
        assert_eq!(
            profile.get("recommended_precision_mode"),
            Some(&"bf16".to_string())
        );
    }

    #[test]
    fn resolve_data_root_path_keeps_absolute_paths() {
        let absolute_path = resolve_data_root_path("/tmp/crucible-data-root");
        assert_eq!(absolute_path, Path::new("/tmp/crucible-data-root"));
    }

    #[test]
    fn resolve_data_root_path_anchors_relative_paths_to_workspace_root() {
        let relative_path = resolve_data_root_path(".crucible");
        assert!(relative_path.ends_with(Path::new(".crucible")));
        assert!(relative_path.is_absolute());
    }
}
