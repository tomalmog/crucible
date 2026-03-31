//! Model registry query commands used by Studio panels.

use crate::commands::runtime_queries::resolve_data_root_path;
use crate::models::ModelEntrySummary;
use serde_json::Value;
use std::fs;
use std::path::Path;

#[tauri::command]
pub fn list_models(data_root: String) -> Result<Vec<ModelEntrySummary>, String> {
    let models_dir = resolve_data_root_path(&data_root).join("models");
    let index_path = models_dir.join("index.json");
    if !index_path.exists() {
        return Ok(vec![]);
    }
    let index = read_json_file(&index_path)?;

    let model_names = match index.get("model_names").and_then(Value::as_array) {
        Some(names) => names.clone(),
        None => {
            // Old flat format with version_ids — treat as single "default" model
            return list_models_from_old_flat_index(&models_dir, &index);
        }
    };

    let entries_dir = models_dir.join("entries");
    let has_entries_dir = entries_dir.is_dir();

    let mut summaries = Vec::with_capacity(model_names.len());
    for raw_name in &model_names {
        let name = raw_name
            .as_str()
            .ok_or_else(|| "model_name entry is not a string".to_string())?;

        // Try new flat entries format first
        if has_entries_dir {
            let entry_path = entries_dir.join(format!("{}.json", safe_filename(name)));
            if entry_path.exists() {
                if let Ok(summary) = parse_entry_file(name, &entry_path) {
                    summaries.push(summary);
                    continue;
                }
            }
        }

        // Fallback: read from old grouped format (groups/ + versions/)
        if let Ok(summary) = read_from_old_grouped_format(&models_dir, name) {
            summaries.push(summary);
        }
    }
    Ok(summaries)
}

/// Parse a flat entry JSON file into a ModelEntrySummary.
fn parse_entry_file(name: &str, entry_path: &Path) -> Result<ModelEntrySummary, String> {
    let entry_data = read_json_file(entry_path)?;
    let obj = entry_data
        .as_object()
        .ok_or_else(|| "Model entry is not an object".to_string())?;

    let location_type = obj
        .get("location_type")
        .and_then(Value::as_str)
        .unwrap_or("local")
        .to_string();
    let has_local = location_type == "local" || location_type == "both";
    let has_remote = location_type == "remote" || location_type == "both";

    let model_path = obj
        .get("model_path")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let size_bytes = model_size_bytes(&model_path);

    Ok(ModelEntrySummary {
        model_name: name.to_string(),
        model_path,
        run_id: obj.get("run_id").and_then(Value::as_str).map(str::to_string),
        created_at: obj
            .get("created_at")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        location_type,
        has_local,
        has_remote,
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
        size_bytes,
    })
}

/// Read a model from the old grouped format (groups/{name}.json → versions/{vid}.json).
fn read_from_old_grouped_format(
    models_dir: &Path,
    name: &str,
) -> Result<ModelEntrySummary, String> {
    let group_path = models_dir
        .join("groups")
        .join(format!("{}.json", safe_filename(name)));
    let group = if group_path.exists() {
        read_json_file(&group_path)?
    } else {
        return Err(format!("No group file for {name}"));
    };

    let version_ids = group
        .get("version_ids")
        .and_then(Value::as_array);
    let active_id = group
        .get("active_version_id")
        .and_then(Value::as_str);

    // Pick the active version, or the last one
    let vid = active_id
        .or_else(|| version_ids.and_then(|a| a.last()).and_then(Value::as_str));

    let versions_dir = models_dir.join("versions");
    if let Some(vid) = vid {
        let vpath = versions_dir.join(format!("{vid}.json"));
        if vpath.exists() {
            return summary_from_version_file(name, &vpath, version_ids);
        }
    }

    // No version data — return stub
    Ok(ModelEntrySummary {
        model_name: name.to_string(),
        model_path: String::new(),
        run_id: None,
        created_at: String::new(),
        location_type: "local".to_string(),
        has_local: true,
        has_remote: false,
        remote_host: String::new(),
        remote_path: String::new(),
        size_bytes: 0,
    })
}

/// Build a summary from an old-format version JSON file.
fn summary_from_version_file(
    name: &str,
    vpath: &Path,
    all_version_ids: Option<&Vec<Value>>,
) -> Result<ModelEntrySummary, String> {
    let vdata = read_json_file(vpath)?;
    let obj = vdata
        .as_object()
        .ok_or_else(|| "Version data is not an object".to_string())?;

    let location_type = obj
        .get("location_type")
        .and_then(Value::as_str)
        .unwrap_or("local")
        .to_string();

    // Scan all versions for has_local/has_remote
    let (mut has_local, mut has_remote) = (false, false);
    let versions_dir = vpath.parent().unwrap_or(vpath);
    if let Some(vids) = all_version_ids {
        for vid_val in vids {
            if let Some(vid) = vid_val.as_str() {
                let p = versions_dir.join(format!("{vid}.json"));
                if let Ok(d) = read_json_file(&p) {
                    match d.get("location_type").and_then(Value::as_str).unwrap_or("local") {
                        "remote" => has_remote = true,
                        "both" => { has_local = true; has_remote = true; }
                        _ => has_local = true,
                    }
                }
                if has_local && has_remote { break; }
            }
        }
    } else {
        has_local = location_type == "local" || location_type == "both";
        has_remote = location_type == "remote" || location_type == "both";
    }

    let model_path = obj
        .get("model_path")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let size_bytes = model_size_bytes(&model_path);

    Ok(ModelEntrySummary {
        model_name: name.to_string(),
        model_path,
        run_id: obj.get("run_id").and_then(Value::as_str).map(str::to_string),
        created_at: obj
            .get("created_at")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        location_type,
        has_local,
        has_remote,
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
        size_bytes,
    })
}

/// Handle the oldest index format where index.json has version_ids directly.
fn list_models_from_old_flat_index(
    models_dir: &Path,
    index: &Value,
) -> Result<Vec<ModelEntrySummary>, String> {
    let version_ids = match index.get("version_ids").and_then(Value::as_array) {
        Some(vids) => vids,
        None => return Ok(vec![]),
    };
    let active_id = index.get("active_version_id").and_then(Value::as_str);
    let vid = active_id
        .or_else(|| version_ids.last().and_then(Value::as_str));

    if let Some(vid) = vid {
        let vpath = models_dir.join("versions").join(format!("{vid}.json"));
        if vpath.exists() {
            let summary = summary_from_version_file("default", &vpath, Some(version_ids))?;
            return Ok(vec![summary]);
        }
    }
    Ok(vec![])
}

#[tauri::command]
pub fn get_model_architecture(model_path: String) -> Result<Value, String> {
    let model = std::path::PathBuf::from(&model_path);
    // model_path may be a directory (hub downloads) or a file (trained models)
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

#[tauri::command]
pub fn get_model_index_mtime(data_root: String) -> Result<String, String> {
    let index_path = resolve_data_root_path(&data_root).join("models").join("index.json");
    let meta = fs::metadata(&index_path).map_err(|e| e.to_string())?;
    let mtime = meta
        .modified()
        .map_err(|e| e.to_string())?
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?;
    Ok(format!("{}.{}", mtime.as_secs(), mtime.subsec_nanos()))
}

fn read_json_file(path: &Path) -> Result<Value, String> {
    let payload = fs::read_to_string(path)
        .map_err(|error| format!("Failed to read {}: {error}", path.display()))?;
    serde_json::from_str(&payload)
        .map_err(|error| format!("Failed to parse {}: {error}", path.display()))
}

fn safe_filename(name: &str) -> String {
    // Must match Python's sanitize_remote_name: re.sub(r"[^a-zA-Z0-9_.\-]", "_", name)
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-' { c } else { '_' })
        .collect()
}

/// Get the size of a model path (file or directory).
fn model_size_bytes(model_path: &str) -> u64 {
    if model_path.is_empty() {
        return 0;
    }
    let p = std::path::PathBuf::from(model_path);
    if p.is_file() {
        fs::metadata(&p).map(|m| m.len()).unwrap_or(0)
    } else if p.is_dir() {
        dir_size_bytes(&p)
    } else {
        0
    }
}

fn dir_size_bytes(dir: &Path) -> u64 {
    let mut total: u64 = 0;
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                total += fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            } else if path.is_dir() {
                total += dir_size_bytes(&path);
            }
        }
    }
    total
}
