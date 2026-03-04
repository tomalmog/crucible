//! Model registry query commands used by Studio panels.

use crate::commands::runtime_queries::resolve_data_root_path;
use crate::models::{ModelGroupSummary, ModelVersionSummary};
use serde_json::Value;
use std::fs;
use std::path::Path;

#[tauri::command]
pub fn list_model_groups(data_root: String) -> Result<Vec<ModelGroupSummary>, String> {
    let models_dir = resolve_data_root_path(&data_root).join("models");
    let index_path = models_dir.join("index.json");
    if !index_path.exists() {
        return Ok(vec![]);
    }
    let index = read_json_file(&index_path)?;

    // Handle both old format (version_ids) and new format (model_names)
    if let Some(model_names) = index.get("model_names").and_then(Value::as_array) {
        let groups_dir = models_dir.join("groups");
        let mut summaries = Vec::with_capacity(model_names.len());
        for raw_name in model_names {
            let name = raw_name
                .as_str()
                .ok_or_else(|| "model_name entry is not a string".to_string())?;
            let group_path = groups_dir.join(format!("{}.json", safe_filename(name)));
            let group = if group_path.exists() {
                read_json_file(&group_path)?
            } else {
                Value::Object(serde_json::Map::new())
            };
            let version_ids = group
                .get("version_ids")
                .and_then(Value::as_array)
                .map_or(0, |a| a.len());
            let active_version_id = group
                .get("active_version_id")
                .and_then(Value::as_str)
                .map(str::to_string);
            // Use first version's created_at as group created_at
            let created_at = first_version_created_at(&models_dir, &group);
            summaries.push(ModelGroupSummary {
                model_name: name.to_string(),
                version_count: version_ids as u64,
                active_version_id,
                created_at,
            });
        }
        Ok(summaries)
    } else if index.get("version_ids").and_then(Value::as_array).is_some() {
        // Old flat format — present as a single "default" group
        let version_ids = index
            .get("version_ids")
            .and_then(Value::as_array)
            .unwrap();
        let active_version_id = index
            .get("active_version_id")
            .and_then(Value::as_str)
            .map(str::to_string);
        let created_at = if let Some(first_id) = version_ids.first().and_then(Value::as_str) {
            load_version_created_at(&models_dir, first_id)
        } else {
            String::new()
        };
        Ok(vec![ModelGroupSummary {
            model_name: "default".to_string(),
            version_count: version_ids.len() as u64,
            active_version_id,
            created_at,
        }])
    } else {
        Ok(vec![])
    }
}

#[tauri::command]
pub fn list_model_versions(
    data_root: String,
    model_name: String,
) -> Result<Vec<ModelVersionSummary>, String> {
    let models_dir = resolve_data_root_path(&data_root).join("models");

    // Try new grouped format first
    let group_path = models_dir.join("groups").join(format!("{}.json", safe_filename(&model_name)));
    let (version_ids, active_version_id) = if group_path.exists() {
        let group = read_json_file(&group_path)?;
        let vids = group
            .get("version_ids")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let active = group
            .get("active_version_id")
            .and_then(Value::as_str)
            .map(str::to_string);
        (vids, active)
    } else {
        // Fallback to old flat index for "default" model
        let index_path = models_dir.join("index.json");
        if !index_path.exists() {
            return Ok(vec![]);
        }
        let index = read_json_file(&index_path)?;
        let vids = index
            .get("version_ids")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let active = index
            .get("active_version_id")
            .and_then(Value::as_str)
            .map(str::to_string);
        (vids, active)
    };

    let versions_dir = models_dir.join("versions");
    let mut summaries = Vec::with_capacity(version_ids.len());
    for raw_id in &version_ids {
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
        model_name: obj
            .get("model_name")
            .and_then(Value::as_str)
            .unwrap_or("default")
            .to_string(),
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
        location_type: obj
            .get("location_type")
            .and_then(Value::as_str)
            .unwrap_or("local")
            .to_string(),
        remote_host: obj
            .get("remote_host")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        remote_path: obj
            .get("remote_path")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
    })
}

#[tauri::command]
pub fn get_model_architecture(model_path: String) -> Result<Value, String> {
    let model = std::path::PathBuf::from(&model_path);
    // model_path may be a directory (hub downloads) or a file (trained models)
    // Check the path itself first (if directory), then its parent (if file)
    for dir in [model.as_path(), model.parent().unwrap_or(model.as_path())] {
        let training_config = dir.join("training_config.json");
        if training_config.exists() {
            return read_json_file(&training_config);
        }
        let hf_config = dir.join("config.json");
        if hf_config.exists() {
            return read_json_file(&hf_config);
        }
    }
    Ok(Value::Null)
}

fn first_version_created_at(models_dir: &Path, group: &Value) -> String {
    if let Some(first_id) = group
        .get("version_ids")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_str)
    {
        load_version_created_at(models_dir, first_id)
    } else {
        String::new()
    }
}

fn load_version_created_at(models_dir: &Path, version_id: &str) -> String {
    let version_path = models_dir.join("versions").join(format!("{version_id}.json"));
    if let Ok(data) = read_json_file(&version_path) {
        data.get("created_at")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    } else {
        String::new()
    }
}

fn read_json_file(path: &Path) -> Result<Value, String> {
    let payload = fs::read_to_string(path)
        .map_err(|error| format!("Failed to read {}: {error}", path.display()))?;
    serde_json::from_str(&payload)
        .map_err(|error| format!("Failed to parse {}: {error}", path.display()))
}

fn safe_filename(name: &str) -> String {
    name.replace('/', "--")
}

fn string_field(map: &serde_json::Map<String, Value>, key: &str) -> Result<String, String> {
    map.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("Field '{key}' is missing or invalid"))
}
