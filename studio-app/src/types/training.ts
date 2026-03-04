export type TrainingMethod =
  | "train"
  | "sft"
  | "dpo-train"
  | "rlhf-train"
  | "lora-train"
  | "distill"
  | "domain-adapt"
  | "grpo-train"
  | "qlora-train"
  | "kto-train"
  | "orpo-train"
  | "multimodal-train"
  | "rlvr-train";

export type TrainingMethodCategory =
  | "pre-training"
  | "fine-tuning"
  | "alignment"
  | "knowledge-transfer";

export interface TrainingMethodCategoryInfo {
  id: TrainingMethodCategory;
  label: string;
}

export const TRAINING_METHOD_CATEGORIES: TrainingMethodCategoryInfo[] = [
  { id: "pre-training", label: "Pre-Training" },
  { id: "fine-tuning", label: "Fine-Tuning" },
  { id: "alignment", label: "Alignment" },
  { id: "knowledge-transfer", label: "Knowledge Transfer" },
];

export interface TrainingMethodInfo {
  id: TrainingMethod;
  name: string;
  description: string;
  category: TrainingMethodCategory;
}

export const TRAINING_METHODS: TrainingMethodInfo[] = [
  { id: "train", name: "Basic Training", description: "Standard supervised training from scratch", category: "pre-training" },
  { id: "sft", name: "SFT", description: "Supervised fine-tuning on instruction data", category: "fine-tuning" },
  { id: "lora-train", name: "LoRA", description: "Low-rank adaptation for parameter-efficient fine-tuning", category: "fine-tuning" },
  { id: "qlora-train", name: "QLoRA", description: "Quantized LoRA for memory-efficient fine-tuning", category: "fine-tuning" },
  { id: "domain-adapt", name: "Domain Adaptation", description: "Adapt a pre-trained model to a new domain", category: "fine-tuning" },
  { id: "multimodal-train", name: "Multimodal", description: "Vision-language model fine-tuning", category: "fine-tuning" },
  { id: "dpo-train", name: "DPO", description: "Direct preference optimization with chosen/rejected pairs", category: "alignment" },
  { id: "rlhf-train", name: "RLHF", description: "Reinforcement learning from human feedback", category: "alignment" },
  { id: "kto-train", name: "KTO", description: "Kahneman-Tversky optimization with unpaired preferences", category: "alignment" },
  { id: "orpo-train", name: "ORPO", description: "Odds ratio preference optimization (SFT + preference)", category: "alignment" },
  { id: "grpo-train", name: "GRPO", description: "Group relative policy optimization with reward functions", category: "alignment" },
  { id: "rlvr-train", name: "RLVR", description: "RL with verifiable rewards for code and math", category: "alignment" },
  { id: "distill", name: "Distillation", description: "Knowledge distillation from a teacher model", category: "knowledge-transfer" },
];

export interface SharedTrainingConfig {
  epochs: string;
  learningRate: string;
  batchSize: string;
  optimizer: string;
  precision: string;
  outputDir: string;
  maxTokenLength: string;
  embeddingDim: string;
  numHeads: string;
  numLayers: string;
  checkpointEvery: string;
  mlpHiddenDim: string;
  mlpLayers: string;
  resumeCheckpointPath: string;
  wandbProject: string;
  tensorboardDir: string;
}

export const DEFAULT_SHARED_CONFIG: SharedTrainingConfig = {
  epochs: "3",
  learningRate: "0.001",
  batchSize: "32",
  optimizer: "adamw",
  precision: "fp32",
  outputDir: "./outputs/train",
  maxTokenLength: "256",
  embeddingDim: "128",
  numHeads: "4",
  numLayers: "4",
  checkpointEvery: "1",
  mlpHiddenDim: "512",
  mlpLayers: "1",
  resumeCheckpointPath: "",
  wandbProject: "",
  tensorboardDir: "",
};

/** Per-method overrides for shared config defaults. Fine-tuning methods
 *  need much lower learning rates than training from scratch. */
const METHOD_CONFIG_OVERRIDES: Partial<Record<TrainingMethod, Partial<SharedTrainingConfig>>> = {
  "dpo-train":  { learningRate: "5e-5" },
  "rlhf-train": { learningRate: "1e-5" },
  "lora-train": { learningRate: "2e-4" },
  distill:      { learningRate: "5e-5" },
  sft:          { learningRate: "2e-5" },
  "domain-adapt": { learningRate: "5e-5" },
  "grpo-train": { learningRate: "5e-5" },
  "qlora-train": { learningRate: "2e-4" },
  "kto-train":  { learningRate: "5e-5" },
  "orpo-train": { learningRate: "5e-5" },
  "multimodal-train": { learningRate: "2e-5" },
  "rlvr-train": { learningRate: "5e-5" },
};

export function getDefaultConfigForMethod(method: TrainingMethod): SharedTrainingConfig {
  return { ...DEFAULT_SHARED_CONFIG, ...METHOD_CONFIG_OVERRIDES[method] };
}

/** Required extra fields per training method (CLI flags).
 *  Data path fields (e.g. --sft-data-path) are auto-resolved from the
 *  dataset's source URI and excluded here. */
export const REQUIRED_METHOD_FIELDS: Record<TrainingMethod, string[]> = {
  train: ["--dataset"],
  sft: ["--dataset", "--base-model"],
  "dpo-train": ["--dataset", "--base-model"],
  "rlhf-train": ["--dataset", "--policy-model-path"],
  "lora-train": ["--dataset", "--base-model-path"],
  distill: ["--dataset", "--teacher-model-path"],
  "domain-adapt": ["--dataset", "--base-model-path"],
  "grpo-train": ["--dataset", "--base-model"],
  "qlora-train": ["--dataset", "--base-model-path"],
  "kto-train": ["--dataset", "--base-model"],
  "orpo-train": ["--dataset", "--base-model"],
  "multimodal-train": ["--dataset", "--base-model"],
  "rlvr-train": ["--dataset", "--base-model"],
};

export interface TrainingConfigDraft {
  shared: SharedTrainingConfig;
  extra: Record<string, string>;
}

export interface TrainingConfigFile {
  draft: TrainingConfigDraft | null;
}
