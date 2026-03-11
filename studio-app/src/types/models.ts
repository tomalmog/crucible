export interface ModelEntry {
  modelName: string;
  modelPath: string;
  runId: string | null;
  createdAt: string;
  locationType: string;
  hasLocal: boolean;
  hasRemote: boolean;
  remoteHost: string;
  remotePath: string;
  sizeBytes: number;
}
