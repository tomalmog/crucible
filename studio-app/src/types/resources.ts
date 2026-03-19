export interface StorageBreakdown {
  datasetsBytes: number;
  runsBytes: number;
  modelsBytes: number;
  cacheBytes: number;
  totalBytes: number;
  diskAvailableBytes: number;
}

export interface OrphanedRun {
  runId: string;
  datasetName: string;
  sizeBytes: number;
  updatedAt: string;
}
