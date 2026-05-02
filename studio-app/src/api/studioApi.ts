import { invoke } from "@tauri-apps/api/core";
import {
  CommandTaskStart,
  CommandTaskStatus,
  DatasetDashboard,
  DatasetEntry,
  LineageGraphSummary,
  RecordSample,
  TrainingRunSummary,
  TrainingHistory,
} from "../types";
import type { ModelEntry } from "../types/models";

export async function listModels(dataRoot: string): Promise<ModelEntry[]> {
  return invoke<ModelEntry[]>("list_models", { dataRoot });
}

export async function getModelArchitecture(modelPath: string): Promise<Record<string, unknown> | null> {
  return invoke<Record<string, unknown> | null>("get_model_architecture", { modelPath });
}

export async function deleteDataset(dataRoot: string, datasetName: string, deleteFiles = false): Promise<void> {
  return invoke<void>("delete_dataset", { dataRoot, datasetName, deleteFiles });
}

export async function listDatasets(dataRoot: string): Promise<DatasetEntry[]> {
  return invoke<DatasetEntry[]>("list_datasets", { dataRoot });
}

export async function getDatasetDashboard(
  dataRoot: string,
  datasetName: string,
): Promise<DatasetDashboard> {
  return invoke<DatasetDashboard>("get_dataset_dashboard", {
    dataRoot,
    datasetName,
  });
}

export async function sampleRecords(
  dataRoot: string,
  datasetName: string,
  offset: number,
  limit: number,
): Promise<RecordSample[]> {
  return invoke<RecordSample[]>("sample_records", {
    dataRoot,
    datasetName,
    offset,
    limit,
  });
}

export async function datasetColumns(
  dataRoot: string,
  datasetName: string,
): Promise<string[]> {
  return invoke<string[]>("dataset_columns", { dataRoot, datasetName });
}

export async function getDatasetRecordCount(
  dataRoot: string,
  datasetName: string,
): Promise<number> {
  return invoke<number>("get_dataset_record_count", { dataRoot, datasetName });
}

export interface BenchmarkRegistryEntry {
  name: string;
  displayName: string;
  type: string;
  entries: number;
  description: string;
  createdAt: string;
  localCompatible: boolean;
}

export async function listBenchmarks(dataRoot: string): Promise<BenchmarkRegistryEntry[]> {
  return invoke<BenchmarkRegistryEntry[]>("list_benchmarks", { dataRoot });
}

export async function startCrucibleCommand(
  dataRoot: string,
  args: string[],
  label?: string,
  config?: Record<string, unknown>,
): Promise<CommandTaskStart> {
  const configJson = config ? JSON.stringify(config) : undefined;
  return invoke<CommandTaskStart>("start_crucible_command", { dataRoot, args, label, configJson });
}

export async function getCrucibleCommandStatus(
  taskId: string,
): Promise<CommandTaskStatus> {
  return invoke<CommandTaskStatus>("get_crucible_command_status", { taskId });
}

export async function listCrucibleTasks(): Promise<CommandTaskStatus[]> {
  return invoke<CommandTaskStatus[]>("list_crucible_tasks");
}

export async function killCrucibleTask(taskId: string): Promise<void> {
  return invoke<void>("kill_crucible_task", { taskId });
}

export async function renameCrucibleTask(taskId: string, label: string): Promise<void> {
  return invoke<void>("rename_crucible_task", { taskId, label });
}

export async function deleteCrucibleTask(taskId: string): Promise<void> {
  return invoke<void>("delete_crucible_task", { taskId });
}

export async function loadTrainingHistory(
  historyPath: string,
): Promise<TrainingHistory> {
  return invoke<TrainingHistory>("load_training_history", { historyPath });
}

export async function listTrainingRuns(
  dataRoot: string,
): Promise<TrainingRunSummary[]> {
  return invoke<TrainingRunSummary[]>("list_training_runs", { dataRoot });
}

export async function getLineageGraph(
  dataRoot: string,
): Promise<LineageGraphSummary> {
  return invoke<LineageGraphSummary>("get_lineage_graph", { dataRoot });
}

export async function getHardwareProfile(
  dataRoot: string,
): Promise<Record<string, string>> {
  return invoke<Record<string, string>>("get_hardware_profile", { dataRoot });
}

export async function loadTrainingConfig(
  dataRoot: string,
  method: string,
): Promise<import("../types/training").TrainingConfigFile> {
  return invoke("load_training_config", { dataRoot, method });
}

export async function saveTrainingConfig(
  dataRoot: string,
  method: string,
  payload: object,
): Promise<void> {
  return invoke("save_training_config", { dataRoot, method, payload });
}

export async function getModelIndexMtime(dataRoot: string): Promise<string> {
  return invoke<string>("get_model_index_mtime", { dataRoot });
}

export async function writeTextFile(filePath: string, contents: string): Promise<void> {
  return invoke<void>("write_text_file", { filePath, contents });
}
