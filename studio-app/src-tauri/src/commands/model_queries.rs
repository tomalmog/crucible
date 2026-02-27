//! Model registry query commands used by Studio panels.

use crate::commands::runtime_queries::resolve_data_root_path;
use crate::models::ModelVersionSummary;
use serde_json::Value;
use std::fs;
use std::path::Path;

#[tauri::command]
pub fn list_model_versions(data_root: String) -> Result<Vec<ModelVersionSummary>, String> {
    let models_dir = resolve_data_root_path(&data_root).join("models");
    let index_path = models_dir.join("index.json");
    if !index_path.exists() {
        return Ok(vec![]);
    }
    let index = read_json_file(&index_path)?;
    let active_version_id = index
        .get("active_version_id")
        .and_then(Value::as_str)
        .map(str::to_string);
    let version_ids = index
        .get("version_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| "Model index is missing version_ids array".to_string())?;

    let versions_dir = models_dir.join("versions");
    let mut summaries = Vec::with_capacity(version_ids.len());
    for raw_id in version_ids {
        let version_id = raw_id
            .as_str()
            .ok_or_else(|| "version_id entry is not a string".to_string())?;
        let version_path = versions_dir.join(format!("{version_id}.json"));
        if !version_path.exists() {
            continue;
        }
        let version_data = read_json_file(&version_path)?;
        let summary = parse_model_version(&version_data, &active_version_id)?;
        summaries.push(summary);
    }
    summaries.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(summaries)
}

fn parse_model_version(
    raw: &Value,
    active_version_id: &Option<String>,
) -> Result<ModelVersionSummary, String> {
    let obj = raw
        .as_object()
        .ok_or_else(|| "Model version entry is not an object".to_string())?;
    let version_id = string_field(obj, "version_id")?;
    let is_active = active_version_id
        .as_ref()
        .map_or(false, |active| active == &version_id);
    Ok(ModelVersionSummary {
        version_id,
        model_path: string_field(obj, "model_path")?,
        run_id: obj.get("run_id").and_then(Value::as_str).map(str::to_string),
        parent_version_id: obj
            .get("parent_version_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        created_at: obj
            .get("created_at")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        is_active,
    })
}

fn read_json_file(path: &Path) -> Result<Value, String> {
    let payload = fs::read_to_string(path)
        .map_err(|error| format!("Failed to read {}: {error}", path.display()))?;
    serde_json::from_str(&payload)
        .map_err(|error| format!("Failed to parse {}: {error}", path.display()))
}

fn string_field(map: &serde_json::Map<String, Value>, key: &str) -> Result<String, String> {
    map.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("Field '{key}' is missing or invalid"))
}
