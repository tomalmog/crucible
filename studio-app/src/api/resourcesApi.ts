import { invoke } from "@tauri-apps/api/core";
import type { StorageBreakdown, OrphanedRun } from "../types/resources";

export async function getStorageBreakdown(dataRoot: string): Promise<StorageBreakdown> {
  return invoke<StorageBreakdown>("get_storage_breakdown", { dataRoot });
}

export async function listOrphanedRuns(dataRoot: string): Promise<OrphanedRun[]> {
  return invoke<OrphanedRun[]>("list_orphaned_runs", { dataRoot });
}

export async function deleteOrphanedRuns(dataRoot: string, runIds: string[]): Promise<number> {
  return invoke<number>("delete_orphaned_runs", { dataRoot, runIds });
}

export async function clearCache(dataRoot: string): Promise<number> {
  return invoke<number>("clear_cache", { dataRoot });
}
