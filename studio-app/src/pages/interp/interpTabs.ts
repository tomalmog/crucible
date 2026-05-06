export type InterpTab =
  | "logit-lens"
  | "activation-pca"
  | "activation-patching"
  | "linear-probe"
  | "sae"
  | "steering";

export interface InterpWorkflowSpec {
  tab: InterpTab;
  eyebrow: string;
  title: string;
  description: string;
  signal: string;
}

export const INTERP_TABS: readonly InterpTab[] = [
  "logit-lens",
  "activation-pca",
  "activation-patching",
  "linear-probe",
  "sae",
  "steering",
];

const INTERP_TAB_SET = new Set<string>(INTERP_TABS);

export function isInterpTab(value: unknown): value is InterpTab {
  return typeof value === "string" && INTERP_TAB_SET.has(value);
}

export const INTERP_TAB_LABELS: Record<InterpTab, string> = {
  "logit-lens": "Logit Lens",
  "activation-pca": "Activation PCA",
  "activation-patching": "Activation Patching",
  "linear-probe": "Linear Probe",
  "sae": "Sparse Autoencoder",
  "steering": "Activation Steering",
};

export const INTERP_WORKFLOWS: readonly InterpWorkflowSpec[] = [
  {
    tab: "logit-lens",
    eyebrow: "Prompt",
    title: "Logit Lens",
    description: "Input text, top-k decoded predictions, optional layer subset.",
    signal: "Output: token x layer probability grid",
  },
  {
    tab: "activation-patching",
    eyebrow: "Prompt pair",
    title: "Activation Patching",
    description: "Clean text, corrupted text, target token index, metric.",
    signal: "Output: recovery by layer",
  },
  {
    tab: "sae",
    eyebrow: "Dataset / prompt",
    title: "Sparse Autoencoder",
    description: "Train an SAE on activations, then analyze one prompt with the saved SAE.",
    signal: "Output: loss curves and feature activations",
  },
  {
    tab: "activation-pca",
    eyebrow: "Dataset",
    title: "Activation PCA",
    description: "Dataset records, layer index, sample limit, optional color field.",
    signal: "Output: 2D activation projection",
  },
  {
    tab: "linear-probe",
    eyebrow: "Labeled dataset",
    title: "Linear Probe",
    description: "Dataset records with a label field and probe training parameters.",
    signal: "Output: accuracy by layer",
  },
  {
    tab: "steering",
    eyebrow: "Contrast",
    title: "Activation Steering",
    description: "Positive/negative examples for vector compute, then coefficient for apply.",
    signal: "Output: vector metrics and generation comparison",
  },
];
