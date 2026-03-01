export interface ModelGroup {
  modelName: string;
  versionCount: number;
  activeVersionId: string | null;
  createdAt: string;
}

export interface ModelVersion {
  versionId: string;
  modelName: string;
  modelPath: string;
  runId: string | null;
  parentVersionId: string | null;
  createdAt: string;
  isActive: boolean;
}
