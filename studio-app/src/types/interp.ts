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

// Linear Probe

export interface ProbeLayerResult {
  layer_index: number;
  layer_name: string;
  accuracy: number;
  num_classes: number;
  class_names: string[];
  confusion_matrix: number[][];
  error?: string;
}

export interface LinearProbeResult {
  layers: ProbeLayerResult[];
}

// Sparse Autoencoder

export interface SaeTrainResult {
  sae_path: string;
  layer_name: string;
  layer_index: number;
  input_dim: number;
  latent_dim: number;
  epochs: number;
  final_loss: number;
  final_recon_loss: number;
  final_sparsity_loss: number;
  history: { epoch: number; loss: number; recon_loss: number; sparsity_loss: number }[];
}

export interface SaeFeature {
  feature_index: number;
  activation: number;
  associated_texts?: string[];
}

export interface SaeAnalyzeResult {
  input_text: string;
  layer_name: string;
  reconstruction_error: number;
  sparsity: number;
  active_features: number;
  total_features: number;
  top_features: SaeFeature[];
}

// Activation Steering

export interface SteerComputeResult {
  steering_vector_path: string;
  layer_name: string;
  layer_index: number;
  vector_norm: number;
  cosine_similarity: number;
  num_positive: number;
  num_negative: number;
}

export interface SteerApplyResult {
  input_text: string;
  original_text: string;
  steered_text: string;
  coefficient: number;
  layer_name: string;
  max_new_tokens: number;
}
