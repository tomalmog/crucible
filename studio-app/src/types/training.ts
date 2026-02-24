export type TrainingMethod =
  | "train"
  | "sft"
  | "dpo-train"
  | "rlhf-train"
  | "lora-train"
  | "distill"
  | "domain-adapt";

export interface TrainingMethodInfo {
  id: TrainingMethod;
  name: string;
  description: string;
}

export const TRAINING_METHODS: TrainingMethodInfo[] = [
  { id: "train", name: "Basic Training", description: "Standard supervised training from scratch" },
  { id: "sft", name: "SFT", description: "Supervised fine-tuning on instruction data" },
  { id: "dpo-train", name: "DPO", description: "Direct preference optimization with chosen/rejected pairs" },
  { id: "rlhf-train", name: "RLHF", description: "Reinforcement learning from human feedback" },
  { id: "lora-train", name: "LoRA", description: "Low-rank adaptation for parameter-efficient fine-tuning" },
  { id: "distill", name: "Distillation", description: "Knowledge distillation from a teacher model" },
  { id: "domain-adapt", name: "Domain Adaptation", description: "Adapt a pre-trained model to a new domain" },
];

export interface SharedTrainingConfig {
  dataset: string;
  versionId: string;
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
}

export const DEFAULT_SHARED_CONFIG: SharedTrainingConfig = {
  dataset: "",
  versionId: "",
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
  checkpointEvery: "0",
};
