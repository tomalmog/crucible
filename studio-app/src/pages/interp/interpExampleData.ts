import type {
  LinearProbeResult,
  LogitLensResult,
  PatchingResult,
  PcaResult,
  SaeAnalyzeResult,
  SaeTrainResult,
  SteerApplyResult,
  SteerComputeResult,
} from "../../types/interp";

export const SAMPLE_LOGIT_LENS: LogitLensResult = {
  input_tokens: ["The", " API", " returns", " JSON"],
  layers: [0, 1, 2, 3, 4].map((layer) => ({
    layer_index: layer,
    layer_name: `blocks.${layer}`,
    predictions: [
      { token_position: 0, top_k: [{ token: layer < 2 ? "A" : "The", prob: 0.22 + layer * 0.08 }] },
      { token_position: 1, top_k: [{ token: layer < 2 ? " model" : " API", prob: 0.18 + layer * 0.1 }] },
      { token_position: 2, top_k: [{ token: layer < 3 ? " prints" : " returns", prob: 0.21 + layer * 0.09 }] },
      { token_position: 3, top_k: [{ token: layer < 3 ? " text" : " JSON", prob: 0.16 + layer * 0.11 }] },
    ],
  })),
};

export const SAMPLE_PCA: PcaResult = {
  layer_name: "transformer.h.3.mlp",
  layer_index: 3,
  granularity: "sequence",
  explained_variance: [0.42, 0.18],
  points: [
    { x: -2.1, y: 1.2, label: "code", text: "Parse JSON and retry failed API calls." },
    { x: -1.8, y: 0.7, label: "code", text: "Generate a SQL query from English." },
    { x: -1.3, y: 1.5, label: "code", text: "Explain a TypeScript type error." },
    { x: 1.4, y: -0.9, label: "medical", text: "Summarize symptoms from a patient note." },
    { x: 1.9, y: -1.2, label: "medical", text: "Identify cardiac risk factors." },
    { x: 2.2, y: -0.5, label: "medical", text: "Classify radiology findings." },
    { x: 0.1, y: 2.0, label: "robotics", text: "Plan a warehouse grasp sequence." },
    { x: 0.4, y: 1.6, label: "robotics", text: "Recover from a failed pick action." },
    { x: 0.8, y: 2.3, label: "robotics", text: "Navigate around a blocked aisle." },
  ],
};

export const SAMPLE_PATCHING: PatchingResult = {
  clean_text: "The API returned JSON with a valid user_id field.",
  corrupted_text: "The API returned poetry with a valid user_id field.",
  metric: "target logit recovery",
  clean_token: "JSON",
  corrupt_token: "poetry",
  clean_metric: 0.91,
  corrupted_metric: 0.21,
  layer_results: [0, 1, 2, 3, 4, 5, 6, 7].map((layer) => ({
    layer_index: layer,
    layer_name: `blocks.${layer}.resid_post`,
    patched_metric: [0.24, 0.32, 0.55, 0.83, 0.89, 0.71, 0.48, 0.35][layer],
    recovery: [0.04, 0.16, 0.49, 0.88, 0.97, 0.72, 0.39, 0.2][layer],
  })),
};

export const SAMPLE_PROBE: LinearProbeResult = {
  layers: [0, 1, 2, 3, 4, 5].map((layer) => ({
    layer_index: layer,
    layer_name: `blocks.${layer}.resid_post`,
    accuracy: [0.38, 0.51, 0.64, 0.82, 0.77, 0.69][layer],
    num_classes: 3,
    class_names: ["code", "medical", "robotics"],
    confusion_matrix: [
      [5, 1, 0],
      [1, 4, 1],
      [0, 1, 5],
    ],
  })),
};

export const SAMPLE_SAE_TRAIN: SaeTrainResult = {
  sae_path: "outputs/interp-fixtures/sae_model.pt",
  layer_name: "transformer.h.4.mlp",
  layer_index: 4,
  input_dim: 768,
  latent_dim: 12288,
  epochs: 8,
  final_loss: 0.084,
  final_recon_loss: 0.071,
  final_sparsity_loss: 0.013,
  average_l0: 17.4,
  dead_features: 211,
  fvu: 0.19,
  history: [
    { epoch: 1, loss: 0.33, recon_loss: 0.29, sparsity_loss: 0.04 },
    { epoch: 2, loss: 0.24, recon_loss: 0.21, sparsity_loss: 0.03 },
    { epoch: 3, loss: 0.18, recon_loss: 0.155, sparsity_loss: 0.025 },
    { epoch: 4, loss: 0.14, recon_loss: 0.121, sparsity_loss: 0.019 },
    { epoch: 5, loss: 0.115, recon_loss: 0.099, sparsity_loss: 0.016 },
    { epoch: 6, loss: 0.101, recon_loss: 0.087, sparsity_loss: 0.014 },
    { epoch: 7, loss: 0.091, recon_loss: 0.078, sparsity_loss: 0.013 },
    { epoch: 8, loss: 0.084, recon_loss: 0.071, sparsity_loss: 0.013 },
  ],
};

export const SAMPLE_SAE_ANALYZE: SaeAnalyzeResult = {
  input_text: "The warehouse robot should retry the grasp after detecting the object slipped.",
  layer_name: "transformer.h.4.mlp",
  reconstruction_error: 0.067,
  sparsity: 0.018,
  active_features: 42,
  total_features: 12288,
  top_features: [
    { feature_index: 271, activation: 2.31, concept: "robot grasp recovery", associated_texts: ["The arm retries the grasp after slip detection."] },
    { feature_index: 918, activation: 1.94, concept: "warehouse navigation", associated_texts: ["A robot routes around a blocked aisle."] },
    { feature_index: 1204, activation: 1.37, concept: "API retry logic", associated_texts: ["Retry failed requests with exponential backoff."] },
    { feature_index: 443, activation: 1.11, concept: "safety constraint", associated_texts: ["Stop movement when a human enters the workspace."] },
  ],
};

export const SAMPLE_STEER_COMPUTE: SteerComputeResult = {
  steering_vector_path: "outputs/interp-fixtures/helpful_robotics_vector.pt",
  layer_name: "transformer.h.5.resid_post",
  layer_index: 5,
  vector_norm: 8.4123,
  cosine_similarity: 0.2841,
  num_positive: 24,
  num_negative: 24,
};

export const SAMPLE_STEER_APPLY: SteerApplyResult = {
  input_text: "A warehouse robot fails to pick up a fragile object.",
  original_text: "It should continue the task and attempt another pickup immediately.",
  steered_text: "It should stop, verify the gripper force, inspect whether the object slipped, and retry with a safer grasp.",
  coefficient: 1.6,
  layer_name: "transformer.h.5.resid_post",
  max_new_tokens: 64,
};
