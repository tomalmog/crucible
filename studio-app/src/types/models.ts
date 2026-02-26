export interface ModelVersion {
  versionId: string;
  modelPath: string;
  runId: string | null;
  parentVersionId: string | null;
  createdAt: string;
  isActive: boolean;
}
