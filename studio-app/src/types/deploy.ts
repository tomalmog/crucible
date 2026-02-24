export interface PackageConfig {
  modelPath: string;
  outputPath: string;
  format: string;
}

export interface QuantizeConfig {
  modelPath: string;
  quantizationType: string;
  outputPath: string;
}

export interface LatencyProfileConfig {
  modelPath: string;
  batchSizes: string;
  warmupRuns: string;
}
