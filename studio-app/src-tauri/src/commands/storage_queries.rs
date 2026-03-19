//! Storage analysis and cleanup commands for the Resources page.

use crate::commands::runtime_queries::resolve_data_root_path;
use serde::Serialize;
use serde_json::Value;
use std::fs;
use std::path::Path;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StorageBreakdown {
    pub datasets_bytes: u64,
    pub runs_bytes: u64,
    pub models_bytes: u64,
    pub cache_bytes: u64,
    pub total_bytes: u64,
    pub disk_available_bytes: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OrphanedRun {
    pub run_id: String,
    pub dataset_name: String,
    pub size_bytes: u64,
    pub updated_at: String,
}

#[tauri::command]
pub fn get_storage_breakdown(data_root: String) -> Result<StorageBreakdown, String> {
    let root = resolve_data_root_path(&data_root);
    let datasets_bytes = dir_size_bytes(&root.join("datasets"));
    let runs_bytes = dir_size_bytes(&root.join("runs"));
    let models_bytes = compute_models_bytes(&root);
    let cache_bytes = dir_size_bytes(&root.join("cache"));
    let total_bytes = datasets_bytes + runs_bytes + models_bytes + cache_bytes;
    let disk_available_bytes = get_disk_available(&root);
    Ok(StorageBreakdown {
        datasets_bytes,
        runs_bytes,
        models_bytes,
        cache_bytes,
        total_bytes,
        disk_available_bytes,
    })
}

#[tauri::command]
pub fn list_orphaned_runs(data_root: String) -> Result<Vec<OrphanedRun>, String> {
    let root = resolve_data_root_path(&data_root);
    let runs_root = root.join("runs");
    let index_path = runs_root.join("index.json");
    if !index_path.exists() {
        return Ok(vec![]);
    }
    let index_payload = read_json_file(&index_path)?;
    let run_ids = index_payload
        .get("runs")
        .and_then(Value::as_array)
        .ok_or_else(|| "Run index missing runs array".to_string())?;

    // Collect all run_ids referenced by model entries
    let referenced = collect_model_run_ids(&root);

    let mut orphans = Vec::new();
    for run_id_value in run_ids {
        let run_id = match run_id_value.as_str() {
            Some(s) => s,
            None => continue,
        };
        if referenced.contains(&run_id.to_string()) {
            continue;
        }
        let lifecycle_path = runs_root.join(run_id).join("lifecycle.json");
        let (dataset_name, updated_at) = if lifecycle_path.exists() {
            let payload = read_json_file(&lifecycle_path).unwrap_or(Value::Null);
            let obj = payload.as_object();
            let ds = obj
                .and_then(|o| o.get("dataset_name"))
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string();
            let ua = obj
                .and_then(|o| o.get("updated_at"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            (ds, ua)
        } else {
            ("unknown".to_string(), String::new())
        };
        let size_bytes = dir_size_bytes(&runs_root.join(run_id));
        orphans.push(OrphanedRun {
            run_id: run_id.to_string(),
            dataset_name,
            size_bytes,
            updated_at,
        });
    }
    Ok(orphans)
}

#[tauri::command]
pub fn delete_orphaned_runs(data_root: String, run_ids: Vec<String>) -> Result<u64, String> {
    let root = resolve_data_root_path(&data_root);
    let runs_root = root.join("runs");
    let index_path = runs_root.join("index.json");

    // Remove from index.json
    if index_path.exists() {
        let index_payload = read_json_file(&index_path)?;
        if let Some(obj) = index_payload.as_object() {
            let mut updated = obj.clone();
            if let Some(runs_array) = updated.get_mut("runs").and_then(Value::as_array_mut) {
                let to_delete: std::collections::HashSet<&str> =
                    run_ids.iter().map(|s| s.as_str()).collect();
                runs_array.retain(|v| {
                    v.as_str().map(|s| !to_delete.contains(s)).unwrap_or(true)
                });
            }
            let json = serde_json::to_string_pretty(&updated)
                .map_err(|e| format!("Failed to serialize index: {e}"))?;
            fs::write(&index_path, json)
                .map_err(|e| format!("Failed to write index: {e}"))?;
        }
    }

    // Delete run directories
    let mut deleted = 0u64;
    for run_id in &run_ids {
        let run_dir = runs_root.join(run_id);
        if run_dir.exists() {
            if fs::remove_dir_all(&run_dir).is_ok() {
                deleted += 1;
            }
        }
    }
    Ok(deleted)
}

#[tauri::command]
pub fn clear_cache(data_root: String) -> Result<u64, String> {
    let root = resolve_data_root_path(&data_root);
    let cache_dir = root.join("cache");
    if !cache_dir.exists() {
        return Ok(0);
    }
    let size = dir_size_bytes(&cache_dir);
    let entries = fs::read_dir(&cache_dir)
        .map_err(|e| format!("Failed to read cache dir: {e}"))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let _ = fs::remove_dir_all(&path);
        } else {
            let _ = fs::remove_file(&path);
        }
    }
    Ok(size)
}

/// Compute total size of model entry directories by reading model_path from each entry JSON.
fn compute_models_bytes(root: &Path) -> u64 {
    let entries_dir = root.join("models").join("entries");
    if !entries_dir.exists() {
        // Try versions directory (grouped registry format)
        let versions_dir = root.join("models").join("versions");
        if !versions_dir.exists() {
            return dir_size_bytes(&root.join("models"));
        }
        return compute_model_versions_bytes(&versions_dir);
    }
    compute_model_entries_bytes(&entries_dir)
}

fn compute_model_entries_bytes(entries_dir: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = fs::read_dir(entries_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "json").unwrap_or(false) {
                total += fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                if let Ok(payload) = read_json_file(&path) {
                    if let Some(model_path) = payload.get("model_path").and_then(Value::as_str) {
                        let mp = Path::new(model_path);
                        if mp.is_dir() {
                            total += dir_size_bytes(mp);
                        }
                    }
                }
            }
        }
    }
    total
}

fn compute_model_versions_bytes(versions_dir: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = fs::read_dir(versions_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "json").unwrap_or(false) {
                total += fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                if let Ok(payload) = read_json_file(&path) {
                    if let Some(model_path) = payload.get("model_path").and_then(Value::as_str) {
                        let mp = Path::new(model_path);
                        if mp.is_dir() {
                            total += dir_size_bytes(mp);
                        }
                    }
                }
            }
        }
    }
    total
}

/// Collect all run_ids referenced by model entry/version JSON files.
fn collect_model_run_ids(root: &Path) -> std::collections::HashSet<String> {
    let mut ids = std::collections::HashSet::new();
    // Check entries/ directory
    let entries_dir = root.join("models").join("entries");
    collect_run_ids_from_dir(&entries_dir, &mut ids);
    // Check versions/ directory (grouped registry)
    let versions_dir = root.join("models").join("versions");
    collect_run_ids_from_dir(&versions_dir, &mut ids);
    ids
}

fn collect_run_ids_from_dir(dir: &Path, ids: &mut std::collections::HashSet<String>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "json").unwrap_or(false) {
                if let Ok(payload) = read_json_file(&path) {
                    if let Some(run_id) = payload.get("run_id").and_then(Value::as_str) {
                        if !run_id.is_empty() {
                            ids.insert(run_id.to_string());
                        }
                    }
                }
            }
        }
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

fn get_disk_available(path: &Path) -> u64 {
    use std::ffi::CString;
    let c_path = match CString::new(path.to_string_lossy().as_bytes()) {
        Ok(p) => p,
        Err(_) => return 0,
    };
    unsafe {
        let mut stat: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(c_path.as_ptr(), &mut stat) == 0 {
            stat.f_bavail as u64 * stat.f_frsize as u64
        } else {
            0
        }
    }
}

fn read_json_file(path: &Path) -> Result<Value, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse {}: {e}", path.display()))
}
