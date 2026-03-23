/** Auto-generated job labels for the Jobs page. */

const METHOD_DISPLAY: Record<string, string> = {
  train: "Train",
  sft: "SFT",
  "dpo-train": "DPO",
  "rlhf-train": "RLHF",
  "lora-train": "LoRA",
  "qlora-train": "QLoRA",
  "grpo-train": "GRPO",
  "kto-train": "KTO",
  "orpo-train": "ORPO",
  distill: "Distill",
  "domain-adapt": "Domain Adapt",
  "distributed-train": "Distributed",
  "multimodal-train": "Multimodal",
  "rlvr-train": "RLVR",
  "logit-lens": "Logit Lens",
  "activation-pca": "Activation PCA",
  "activation-patch": "Activation Patching",
  eval: "Eval",
  sweep: "Sweep",
};

/**
 * Build a display label for any job from its method/type and model name.
 * Returns "Method · Name" or just "Method" if no name is available.
 */
export function jobLabel(method: string, modelNameOrPath: string): string {
  const display = METHOD_DISPLAY[method] || method;
  const name = modelBasename(modelNameOrPath);
  return name ? `${display} · ${name}` : display;
}

export function modelBasename(path: string): string {
  return path.replace(/\/+$/, "").split("/").pop() || path;
}

export function trainingLabel(method: string, modelName: string): string {
  const display = METHOD_DISPLAY[method] || method;
  return `${display} · ${modelName}`;
}

export function sweepLabel(method: string, modelName: string): string {
  const display = METHOD_DISPLAY[method] || method;
  return `Sweep · ${display} · ${modelName}`;
}

export function logitLensLabel(modelPath: string): string {
  return `Logit Lens · ${modelBasename(modelPath)}`;
}

export function activationPcaLabel(modelPath: string): string {
  return `Activation PCA · ${modelBasename(modelPath)}`;
}

export function activationPatchingLabel(modelPath: string): string {
  return `Activation Patching · ${modelBasename(modelPath)}`;
}

export function evalLabel(modelName: string): string {
  return `Eval · ${modelName}`;
}
