export type DocCategory =
  | "Getting Started"
  | "Training"
  | "Data"
  | "Concepts"
  | "Studio Guide"
  | "Evaluation & Safety"
  | "Deployment";

export interface DocEntry {
  slug: string;
  title: string;
  category: DocCategory;
  content: string;
}

export const CATEGORY_ORDER: DocCategory[] = [
  "Getting Started",
  "Training",
  "Data",
  "Concepts",
  "Studio Guide",
  "Evaluation & Safety",
  "Deployment",
];

// Content imports — one per doc page
import { gettingStarted } from "./content/getting-started";
import { trainingOverview } from "./content/training-overview";
import { trainingBasic } from "./content/training-basic";
import { trainingSft } from "./content/training-sft";
import { trainingDpo } from "./content/training-dpo";
import { trainingRlhf } from "./content/training-rlhf";
import { trainingLora } from "./content/training-lora";
import { trainingQlora } from "./content/training-qlora";
import { trainingDistillation } from "./content/training-distillation";
import { trainingDomainAdapt } from "./content/training-domain-adapt";
import { trainingGrpo } from "./content/training-grpo";
import { trainingKto } from "./content/training-kto";
import { trainingOrpo } from "./content/training-orpo";
import { trainingMultimodal } from "./content/training-multimodal";
import { trainingRlvr } from "./content/training-rlvr";
import { trainingCommonOptions } from "./content/training-common-options";
import { hyperparameterSweeps } from "./content/hyperparameter-sweeps";
import { experimentTracking } from "./content/experiment-tracking";
import { dataFormats } from "./content/data-formats";
import { dataManagement } from "./content/data-management";
import { concepts } from "./content/concepts";
import { studioGuide } from "./content/studio-guide";
import { evaluationSafety } from "./content/evaluation-safety";
import { deployment } from "./content/deployment";

export const DOC_ENTRIES: DocEntry[] = [
  gettingStarted,
  trainingOverview,
  trainingBasic,
  trainingSft,
  trainingDpo,
  trainingRlhf,
  trainingLora,
  trainingQlora,
  trainingDistillation,
  trainingDomainAdapt,
  trainingGrpo,
  trainingKto,
  trainingOrpo,
  trainingMultimodal,
  trainingRlvr,
  trainingCommonOptions,
  hyperparameterSweeps,
  experimentTracking,
  dataFormats,
  dataManagement,
  concepts,
  studioGuide,
  evaluationSafety,
  deployment,
];

/** Maps TrainingMethod id to the doc slug for linking from the training picker. */
export const TRAINING_METHOD_ANCHORS: Record<string, string> = {
  train: "training-basic",
  sft: "training-sft",
  "dpo-train": "training-dpo",
  "rlhf-train": "training-rlhf",
  "lora-train": "training-lora",
  distill: "training-distillation",
  "domain-adapt": "training-domain-adapt",
  "grpo-train": "training-grpo",
  "qlora-train": "training-qlora",
  "kto-train": "training-kto",
  "orpo-train": "training-orpo",
  "multimodal-train": "training-multimodal",
  "rlvr-train": "training-rlvr",
};
