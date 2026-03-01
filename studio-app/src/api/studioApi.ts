import { invoke } from "@tauri-apps/api/core";
import {
  CommandTaskStart,
  CommandTaskStatus,
  DatasetDashboard,
  LineageGraphSummary,
  RecordSample,
  TrainingRunSummary,
  TrainingHistory,
  VersionDiff,
  VersionSummary,
} from "../types";
import type { ModelGroup, ModelVersion } from "../types/models";

export async function listModelGroups(dataRoot: string): Promise<ModelGroup[]> {
  return invoke<ModelGroup[]>("list_model_groups", { dataRoot });
}

export async function listModelVersions(dataRoot: string, modelName: string): Promise<ModelVersion[]> {
  return invoke<ModelVersion[]>("list_model_versions", { dataRoot, modelName });
}

export async function getModelArchitecture(modelPath: string): Promise<Record<string, unknown> | null> {
  return invoke<Record<string, unknown> | null>("get_model_architecture", { modelPath });
}

export async function deleteDataset(dataRoot: string, datasetName: string): Promise<void> {
  return invoke<void>("delete_dataset", { dataRoot, datasetName });
}

export async function listDatasets(dataRoot: string): Promise<string[]> {
  return invoke<string[]>("list_datasets", { dataRoot });
}

export async function listVersions(
  dataRoot: string,
  datasetName: string,
): Promise<VersionSummary[]> {
  return invoke<VersionSummary[]>("list_versions", { dataRoot, datasetName });
}

export async function getDatasetDashboard(
  dataRoot: string,
  datasetName: string,
  versionId: string | null,
): Promise<DatasetDashboard> {
  return invoke<DatasetDashboard>("get_dataset_dashboard", {
    dataRoot,
    datasetName,
    versionId,
  });
}

export async function sampleRecords(
  dataRoot: string,
  datasetName: string,
  versionId: string | null,
  offset: number,
  limit: number,
): Promise<RecordSample[]> {
  return invoke<RecordSample[]>("sample_records", {
    dataRoot,
    datasetName,
    versionId,
    offset,
    limit,
  });
}

export async function versionDiff(
  dataRoot: string,
  datasetName: string,
  baseVersion: string,
  targetVersion: string,
): Promise<VersionDiff> {
  return invoke<VersionDiff>("version_diff", {
    dataRoot,
    datasetName,
    baseVersion,
    targetVersion,
  });
}

export async function startForgeCommand(
  dataRoot: string,
  args: string[],
): Promise<CommandTaskStart> {
  return invoke<CommandTaskStart>("start_forge_command", { dataRoot, args });
}

export async function getForgeCommandStatus(
  taskId: string,
): Promise<CommandTaskStatus> {
  return invoke<CommandTaskStatus>("get_forge_command_status", { taskId });
}

export async function listForgeTasks(): Promise<CommandTaskStatus[]> {
  return invoke<CommandTaskStatus[]>("list_forge_tasks");
}

export async function killForgeTask(taskId: string): Promise<void> {
  return invoke<void>("kill_forge_task", { taskId });
}

export async function renameForgeTask(taskId: string, label: string): Promise<void> {
  return invoke<void>("rename_forge_task", { taskId, label });
}

export async function deleteForgeTask(taskId: string): Promise<void> {
  return invoke<void>("delete_forge_task", { taskId });
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
