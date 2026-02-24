export interface SafetyEvalConfig {
  modelPath: string;
  dataset: string;
  categories: string;
  threshold: string;
}

export interface SafetyGateConfig {
  modelPath: string;
  minScore: string;
  maxToxicity: string;
  requireCategories: string;
}
