//! Dataset query commands used by Studio panels.

use crate::commands::runtime_queries::resolve_data_root_path;
use crate::models::{DatasetDashboard, RecordSample, SourceCount, TrainingHistory, VersionDiff, VersionSummary};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

#[tauri::command]
pub fn list_datasets(data_root: String) -> Result<Vec<String>, String> {
    let datasets_dir = resolve_data_root_path(&data_root).join("datasets");
    if !datasets_dir.exists() {
        return Ok(vec![]);
    }
    let mut names = read_child_dirs(&datasets_dir)?;
    names.sort();
    Ok(names)
}

#[tauri::command]
pub fn list_versions(data_root: String, dataset_name: String) -> Result<Vec<VersionSummary>, String> {
    let catalog = read_catalog(&dataset_root(&data_root, &dataset_name))?;
    let versions = catalog
        .get("versions")
        .and_then(Value::as_array)
        .ok_or_else(|| "Catalog is missing versions array".to_string())?;
    let mut summaries = Vec::with_capacity(versions.len());
    for version in versions {
        summaries.push(parse_version_summary(version)?);
    }
    Ok(summaries)
}

#[tauri::command]
pub fn get_dataset_dashboard(
    data_root: String,
    dataset_name: String,
    version_id: Option<String>,
) -> Result<DatasetDashboard, String> {
    let selected_version = resolve_version(&data_root, &dataset_name, version_id)?;
    let records = read_records(&data_root, &dataset_name, &selected_version)?;
    if records.is_empty() {
        return Err("Dataset version has no records".to_string());
    }
    let record_count = records.len() as u64;
    let mut language_counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut source_counts: HashMap<String, u64> = HashMap::new();
    let mut quality_sum = 0.0;
    let mut min_quality = f64::INFINITY;
    let mut max_quality = f64::NEG_INFINITY;
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
    }
    let average_quality = quality_sum / record_count as f64;
    let mut source_rows: Vec<SourceCount> = source_counts
        .into_iter()
        .map(|(source, count)| SourceCount { source, count })
        .collect();
    source_rows.sort_by(|left, right| right.count.cmp(&left.count));
    source_rows.truncate(12);
    Ok(DatasetDashboard {
        dataset_name,
        version_id: selected_version,
        record_count,
        average_quality,
        min_quality,
        max_quality,
        language_counts,
        source_counts: source_rows,
    })
}

#[tauri::command]
pub fn sample_records(
    data_root: String,
    dataset_name: String,
    version_id: Option<String>,
    offset: usize,
    limit: usize,
) -> Result<Vec<RecordSample>, String> {
    let selected_version = resolve_version(&data_root, &dataset_name, version_id)?;
    let records = read_records(&data_root, &dataset_name, &selected_version)?;
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
        samples.push(RecordSample {
            record_id: string_field(record_object, "record_id")?,
            source_uri: string_field(metadata, "source_uri")?,
            language: string_field(metadata, "language")?,
            quality_score: float_field(metadata, "quality_score")?,
            text: string_field(record_object, "text")?,
        });
    }
    Ok(samples)
}

#[tauri::command]
pub fn version_diff(
    data_root: String,
    dataset_name: String,
    base_version: String,
    target_version: String,
) -> Result<VersionDiff, String> {
    let base_ids = record_id_set(&data_root, &dataset_name, &base_version)?;
    let target_ids = record_id_set(&data_root, &dataset_name, &target_version)?;
    let shared_records = base_ids.intersection(&target_ids).count() as u64;
    let removed_records = base_ids.difference(&target_ids).count() as u64;
    let added_records = target_ids.difference(&base_ids).count() as u64;
    Ok(VersionDiff {
        dataset_name,
        base_version,
        target_version,
        added_records,
        removed_records,
        shared_records,
    })
}

#[tauri::command]
pub fn load_training_history(history_path: String) -> Result<TrainingHistory, String> {
    let payload = fs::read_to_string(&history_path)
        .map_err(|error| format!("Failed to read history file {history_path}: {error}"))?;
    serde_json::from_str(&payload)
        .map_err(|error| format!("Failed to parse history file {history_path}: {error}"))
}

fn dataset_root(data_root: &str, dataset_name: &str) -> PathBuf {
    resolve_data_root_path(data_root).join("datasets").join(dataset_name)
}

fn records_path(data_root: &str, dataset_name: &str, version_id: &str) -> PathBuf {
    dataset_root(data_root, dataset_name)
        .join("versions")
        .join(version_id)
        .join("records.jsonl")
}

fn read_catalog(dataset_root: &Path) -> Result<Value, String> {
    let catalog_path = dataset_root.join("catalog.json");
    let payload = fs::read_to_string(&catalog_path)
        .map_err(|error| format!("Failed to read catalog {}: {error}", catalog_path.display()))?;
    serde_json::from_str::<Value>(&payload)
        .map_err(|error| format!("Failed to parse catalog {}: {error}", catalog_path.display()))
}

fn resolve_version(
    data_root: &str,
    dataset_name: &str,
    explicit_version: Option<String>,
) -> Result<String, String> {
    if let Some(version_id) = explicit_version {
        return Ok(version_id);
    }
    let catalog = read_catalog(&dataset_root(data_root, dataset_name))?;
    catalog
        .get("latest_version")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| "Catalog is missing latest_version".to_string())
}

fn read_records(data_root: &str, dataset_name: &str, version_id: &str) -> Result<Vec<Value>, String> {
    let records_path = records_path(data_root, dataset_name, version_id);
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

fn record_id_set(data_root: &str, dataset_name: &str, version_id: &str) -> Result<HashSet<String>, String> {
    let records = read_records(data_root, dataset_name, version_id)?;
    let mut ids = HashSet::with_capacity(records.len());
    for record in records {
        let id = record
            .get("record_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "Record is missing record_id".to_string())?;
        ids.insert(id.to_string());
    }
    Ok(ids)
}

fn parse_version_summary(raw: &Value) -> Result<VersionSummary, String> {
    let object = raw
        .as_object()
        .ok_or_else(|| "Version entry is not an object".to_string())?;
    let parent = object.get("parent_version").and_then(Value::as_str).map(str::to_string);
    let record_count = object
        .get("record_count")
        .and_then(Value::as_u64)
        .ok_or_else(|| "Version record_count is missing".to_string())?;
    Ok(VersionSummary {
        version_id: string_field(object, "version_id")?,
        record_count,
        created_at: string_field(object, "created_at")?,
        parent_version: parent,
    })
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
