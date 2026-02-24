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
