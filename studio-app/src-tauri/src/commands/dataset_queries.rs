//! Dataset query commands used by Studio panels.

use crate::commands::runtime_queries::resolve_data_root_path;
use crate::models::{DatasetDashboard, RecordSample, SourceCount, TrainingHistory};
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetEntry {
    pub name: String,
    pub size_bytes: u64,
}

#[tauri::command]
pub fn list_datasets(data_root: String) -> Result<Vec<DatasetEntry>, String> {
    let datasets_dir = resolve_data_root_path(&data_root).join("datasets");
    if !datasets_dir.exists() {
        return Ok(vec![]);
    }
    let mut names = read_child_dirs(&datasets_dir)?;
    // Only include datasets that have a records.jsonl file — empty directories
    // (e.g. from failed ingests) should not appear in the list.
    names.retain(|name| datasets_dir.join(name).join("records.jsonl").exists());
    names.sort();
    let mut entries = Vec::with_capacity(names.len());
    for name in names {
        let size = dataset_records_size(&datasets_dir, &name);
        entries.push(DatasetEntry { name, size_bytes: size });
    }
    Ok(entries)
}

#[tauri::command]
pub fn get_dataset_dashboard(
    data_root: String,
    dataset_name: String,
) -> Result<DatasetDashboard, String> {
    let records = read_records(&data_root, &dataset_name)?;
    if records.is_empty() {
        return Err("Dataset has no records".to_string());
    }
    let record_count = records.len() as u64;
    let mut language_counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut source_counts: HashMap<String, u64> = HashMap::new();
    let mut quality_sum = 0.0;
    let mut min_quality = f64::INFINITY;
    let mut max_quality = f64::NEG_INFINITY;
    let mut token_sum: u64 = 0;
    let mut min_tokens: u64 = u64::MAX;
    let mut max_tokens: u64 = 0;
    let mut field_names_set: BTreeMap<String, bool> = BTreeMap::new();
    for record in &records {
        let metadata = record
            .get("metadata")
            .and_then(Value::as_object)
            .ok_or_else(|| "Record metadata is missing".to_string())?;
        let language = string_field(metadata, "language")?;
        *language_counts.entry(language).or_insert(0) += 1;
        let source_uri = string_field(metadata, "source_uri")?;
        *source_counts.entry(source_uri).or_insert(0) += 1;
        let quality = float_field(metadata, "quality_score")?;
        quality_sum += quality;
        if quality < min_quality {
            min_quality = quality;
        }
        if quality > max_quality {
            max_quality = quality;
        }
        // Token length approximation: split on whitespace
        let text = record.get("text").and_then(Value::as_str).unwrap_or("");
        let tokens = text.split_whitespace().count() as u64;
        token_sum += tokens;
        if tokens < min_tokens { min_tokens = tokens; }
        if tokens > max_tokens { max_tokens = tokens; }
        // Collect field names from extra_fields
        if let Some(extras) = metadata.get("extra_fields").and_then(Value::as_object) {
            for k in extras.keys() {
                if k != "quality_model" {
                    field_names_set.insert(k.clone(), true);
                }
            }
        }
    }
    let average_quality = quality_sum / record_count as f64;
    let avg_token_length = token_sum / record_count;
    let field_names: Vec<String> = field_names_set.into_keys().collect();
    let mut source_rows: Vec<SourceCount> = source_counts
        .into_iter()
        .map(|(source, count)| SourceCount { source, count })
        .collect();
    source_rows.sort_by(|left, right| right.count.cmp(&left.count));
    source_rows.truncate(12);
    Ok(DatasetDashboard {
        dataset_name,
        record_count,
        average_quality,
        min_quality,
        max_quality,
        language_counts,
        source_counts: source_rows,
        avg_token_length,
        min_token_length: if min_tokens == u64::MAX { 0 } else { min_tokens },
        max_token_length: max_tokens,
        field_names,
    })
}

#[tauri::command]
pub fn sample_records(
    data_root: String,
    dataset_name: String,
    offset: usize,
    limit: usize,
) -> Result<Vec<RecordSample>, String> {
    let records = read_records(&data_root, &dataset_name)?;
    let safe_limit = limit.min(200);
    let mut samples: Vec<RecordSample> = Vec::new();
    for record in records.iter().skip(offset).take(safe_limit) {
        let record_object = record
            .as_object()
            .ok_or_else(|| "Record entry is not an object".to_string())?;
        let metadata = record
            .get("metadata")
            .and_then(Value::as_object)
            .ok_or_else(|| "Record metadata is missing".to_string())?;
        let mut extra_fields = BTreeMap::new();
        if let Some(extras) = metadata.get("extra_fields").and_then(Value::as_object) {
            for (k, v) in extras {
                if let Some(s) = v.as_str() {
                    extra_fields.insert(k.clone(), s.to_string());
                } else if let Some(b) = v.as_bool() {
                    extra_fields.insert(k.clone(), b.to_string());
                } else if let Some(n) = v.as_f64() {
                    extra_fields.insert(k.clone(), n.to_string());
                }
            }
        }
        samples.push(RecordSample {
            record_id: string_field(record_object, "record_id")?,
            source_uri: string_field(metadata, "source_uri")?,
            language: string_field(metadata, "language")?,
            quality_score: float_field(metadata, "quality_score")?,
            text: string_field(record_object, "text")?,
            extra_fields,
        });
    }
    Ok(samples)
}

#[tauri::command]
pub fn get_dataset_record_count(data_root: String, dataset_name: String) -> Result<u64, String> {
    let path = records_path(&data_root, &dataset_name);
    let payload = fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    Ok(payload.lines().filter(|l| !l.trim().is_empty()).count() as u64)
}

#[tauri::command]
pub fn dataset_columns(data_root: String, dataset_name: String) -> Result<Vec<String>, String> {
    let records = read_records(&data_root, &dataset_name)?;
    let first = records.first().ok_or_else(|| "Dataset is empty".to_string())?;
    let mut cols = Vec::new();
    // Top-level string fields (e.g. "text")
    if let Some(obj) = first.as_object() {
        for (k, v) in obj {
            if k != "metadata" && k != "record_id" && v.is_string() {
                cols.push(k.clone());
            }
        }
    }
    // metadata.extra_fields keys
    if let Some(extra) = first
        .get("metadata")
        .and_then(|m| m.get("extra_fields"))
        .and_then(Value::as_object)
    {
        for (k, v) in extra {
            if v.is_string() {
                cols.push(k.clone());
            }
        }
    }
    cols.sort();
    Ok(cols)
}

#[tauri::command]
pub fn load_training_history(history_path: String) -> Result<TrainingHistory, String> {
    let payload = fs::read_to_string(&history_path)
        .map_err(|error| format!("Failed to read history file {history_path}: {error}"))?;
    serde_json::from_str(&payload)
        .map_err(|error| format!("Failed to parse history file {history_path}: {error}"))
}

#[tauri::command]
pub fn delete_dataset(data_root: String, dataset_name: String) -> Result<(), String> {
    let dataset_dir = dataset_root(&data_root, &dataset_name);
    if !dataset_dir.exists() {
        return Err(format!("Dataset '{}' not found", dataset_name));
    }
    fs::remove_dir_all(&dataset_dir)
        .map_err(|e| format!("Failed to delete dataset '{}': {}", dataset_name, e))
}

fn dataset_root(data_root: &str, dataset_name: &str) -> PathBuf {
    resolve_data_root_path(data_root).join("datasets").join(dataset_name)
}

fn records_path(data_root: &str, dataset_name: &str) -> PathBuf {
    dataset_root(data_root, dataset_name).join("records.jsonl")
}

fn read_records(data_root: &str, dataset_name: &str) -> Result<Vec<Value>, String> {
    let records_path = records_path(data_root, dataset_name);
    let payload = fs::read_to_string(&records_path)
        .map_err(|error| format!("Failed to read records {}: {error}", records_path.display()))?;
    let mut rows = Vec::new();
    for line in payload.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let row = serde_json::from_str::<Value>(line)
            .map_err(|error| format!("Failed to parse record json in {}: {error}", records_path.display()))?;
        rows.push(row);
    }
    Ok(rows)
}

fn dataset_records_size(datasets_dir: &Path, name: &str) -> u64 {
    let dir = datasets_dir.join(name);
    dir_size_bytes(&dir)
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

fn read_child_dirs(parent: &Path) -> Result<Vec<String>, String> {
    let entries = fs::read_dir(parent)
        .map_err(|error| format!("Failed to read {}: {error}", parent.display()))?;
    let mut rows = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| format!("Failed to read dir entry: {error}"))?;
        let path = entry.path();
        if path.is_dir() {
            if let Some(name) = path.file_name().and_then(|value| value.to_str()) {
                rows.push(name.to_string());
            }
        }
    }
    Ok(rows)
}

fn string_field(map: &serde_json::Map<String, Value>, key: &str) -> Result<String, String> {
    map.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("Field '{key}' is missing or invalid"))
}

fn float_field(map: &serde_json::Map<String, Value>, key: &str) -> Result<f64, String> {
    map.get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("Field '{key}' is missing or invalid"))
}
