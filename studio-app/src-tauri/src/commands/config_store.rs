//! Persistent training config storage (draft + history per method).

use serde_json::Value;
use std::fs;

use super::runtime_queries::resolve_data_root_path;

fn configs_dir(data_root: &str) -> std::path::PathBuf {
    resolve_data_root_path(data_root)
        .join("studio")
        .join("configs")
}

#[tauri::command]
pub fn load_training_config(
    data_root: String,
    method: String,
) -> Result<Value, String> {
    let path = configs_dir(&data_root).join(format!("{method}.json"));
    if !path.exists() {
        return Ok(serde_json::json!({ "draft": null, "history": [] }));
    }
    let raw = fs::read_to_string(&path).map_err(|e| {
        format!("Failed to read config {}: {e}", path.display())
    })?;
    serde_json::from_str::<Value>(&raw).map_err(|e| {
        format!("Failed to parse config {}: {e}", path.display())
    })
}

#[tauri::command]
pub fn save_training_config(
    data_root: String,
    method: String,
    payload: Value,
) -> Result<(), String> {
    let dir = configs_dir(&data_root);
    fs::create_dir_all(&dir).map_err(|e| {
        format!("Failed to create configs dir {}: {e}", dir.display())
    })?;
    let path = dir.join(format!("{method}.json"));
    let json = serde_json::to_string_pretty(&payload).map_err(|e| {
        format!("Failed to serialize config: {e}")
    })?;
    fs::write(&path, json).map_err(|e| {
        format!("Failed to write config {}: {e}", path.display())
    })
}

#[tauri::command]
pub fn write_text_file(file_path: String, contents: String) -> Result<(), String> {
    let path = std::path::Path::new(&file_path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            format!("Failed to create directory {}: {e}", parent.display())
        })?;
    }
    fs::write(path, contents).map_err(|e| {
        format!("Failed to write file {}: {e}", path.display())
    })
}
