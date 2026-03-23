/** Interpretability analysis result types. */

export interface LogitLensTopToken {
  token: string;
  prob: number;
}

export interface LogitLensPrediction {
  token_position: number;
  top_k: LogitLensTopToken[];
}

export interface LogitLensLayer {
  layer_index: number;
  layer_name: string;
  predictions: LogitLensPrediction[];
}

export interface LogitLensResult {
  input_tokens: string[];
  layers: LogitLensLayer[];
  warning?: string;
}

export interface PcaPoint {
  x: number;
  y: number;
  label: string;
  text: string;
}

export interface PcaResult {
  layer_name: string;
  layer_index: number;
  granularity: string;
  explained_variance: number[];
  points: PcaPoint[];
}

export interface PatchingLayerResult {
  layer_index: number;
  layer_name: string;
  patched_metric: number;
  recovery: number;
}

export interface PatchingResult {
  clean_text: string;
  corrupted_text: string;
  metric: string;
  clean_metric: number;
  corrupted_metric: number;
  layer_results: PatchingLayerResult[];
}
