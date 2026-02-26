import { SharedTrainingConfig, TrainingMethod } from "../types/training";

function appendOptional(args: string[], flag: string, value: string): void {
  const trimmed = value.trim();
  if (trimmed.length > 0 && trimmed !== "0") {
    args.push(flag, trimmed);
  }
}

function appendOptionalRaw(args: string[], flag: string, value: string): void {
  const trimmed = value.trim();
  if (trimmed.length > 0) {
    args.push(flag, trimmed);
  }
}

export function buildSharedTrainingArgs(
  config: SharedTrainingConfig,
): string[] {
  const args: string[] = [];
  appendOptionalRaw(args, "--epochs", config.epochs);
  appendOptionalRaw(args, "--learning-rate", config.learningRate);
  appendOptionalRaw(args, "--batch-size", config.batchSize);
  appendOptionalRaw(args, "--optimizer-type", config.optimizer);
  appendOptionalRaw(args, "--precision-mode", config.precision);
  appendOptionalRaw(args, "--output-dir", config.outputDir);
  appendOptionalRaw(args, "--max-token-length", config.maxTokenLength);
  appendOptionalRaw(args, "--hidden-dim", config.embeddingDim);
  appendOptionalRaw(args, "--attention-heads", config.numHeads);
  appendOptionalRaw(args, "--num-layers", config.numLayers);
  if (config.mlpHiddenDim && config.mlpHiddenDim !== "512") {
    args.push("--mlp-hidden-dim", config.mlpHiddenDim);
  }
  if (config.mlpLayers && config.mlpLayers !== "1") {
    args.push("--mlp-layers", config.mlpLayers);
  }
  appendOptional(args, "--checkpoint-every-epochs", config.checkpointEvery);
  if (config.resumeCheckpointPath) {
    args.push("--resume-checkpoint-path", config.resumeCheckpointPath);
  }
  return args;
}

const BOOLEAN_FLAGS: ReadonlySet<string> = new Set([
  "--train-reward-model",
]);

export function buildTrainingArgs(
  method: TrainingMethod,
  shared: SharedTrainingConfig,
  extra: Record<string, string>,
): string[] {
  const args = [method, ...buildSharedTrainingArgs(shared)];
  for (const [key, value] of Object.entries(extra)) {
    if (BOOLEAN_FLAGS.has(key)) {
      if (value === "true") args.push(key);
    } else {
      appendOptionalRaw(args, key, value);
    }
  }
  return args;
}

export function buildIngestArgs(
  source: string,
  dataset: string,
  extra: Record<string, string>,
): string[] {
  const args = ["ingest", source, "--dataset", dataset];
  for (const [key, value] of Object.entries(extra)) {
    appendOptionalRaw(args, key, value);
  }
  return args;
}

export function buildFilterArgs(
  dataset: string,
  extra: Record<string, string>,
): string[] {
  const args = ["filter", "--dataset", dataset];
  for (const [key, value] of Object.entries(extra)) {
    appendOptionalRaw(args, key, value);
  }
  return args;
}

export function buildSafetyEvalArgs(extra: Record<string, string>): string[] {
  const args = ["safety-eval"];
  for (const [key, value] of Object.entries(extra)) {
    appendOptionalRaw(args, key, value);
  }
  return args;
}

export function buildSafetyGateArgs(extra: Record<string, string>): string[] {
  const args = ["safety-gate"];
  for (const [key, value] of Object.entries(extra)) {
    appendOptionalRaw(args, key, value);
  }
  return args;
}

export function buildDeployArgs(extra: Record<string, string>): string[] {
  const args = ["deploy"];
  for (const [key, value] of Object.entries(extra)) {
    appendOptionalRaw(args, key, value);
  }
  return args;
}

export function buildModelArgs(subcommand: string, extra: Record<string, string>): string[] {
  const args = ["model", subcommand];
  for (const [key, value] of Object.entries(extra)) {
    appendOptionalRaw(args, key, value);
  }
  return args;
}

export function buildExperimentArgs(subcommand: string, extra: Record<string, string>): string[] {
  const args = ["experiment", subcommand];
  for (const [key, value] of Object.entries(extra)) {
    appendOptionalRaw(args, key, value);
  }
  return args;
}

export function buildHubArgs(subcommand: string, extra: Record<string, string>): string[] {
  const args = ["hub", subcommand];
  for (const [key, value] of Object.entries(extra)) {
    appendOptionalRaw(args, key, value);
  }
  return args;
}

export function buildEvalArgs(extra: Record<string, string>): string[] {
  const args = ["eval"];
  for (const [key, value] of Object.entries(extra)) {
    appendOptionalRaw(args, key, value);
  }
  return args;
}

export function buildJudgeArgs(extra: Record<string, string>): string[] {
  const args = ["judge"];
  for (const [key, value] of Object.entries(extra)) {
    appendOptionalRaw(args, key, value);
  }
  return args;
}

export function buildCurateArgs(subcommand: string, extra: Record<string, string>): string[] {
  const args = ["curate", subcommand];
  for (const [key, value] of Object.entries(extra)) {
    appendOptionalRaw(args, key, value);
  }
  return args;
}

export function buildMergeArgs(extra: Record<string, string>): string[] {
  const args = ["merge"];
  for (const [key, value] of Object.entries(extra)) {
    appendOptionalRaw(args, key, value);
  }
  return args;
}

export function buildSweepArgs(extra: Record<string, string>): string[] {
  const args = ["sweep"];
  for (const [key, value] of Object.entries(extra)) {
    appendOptionalRaw(args, key, value);
  }
  return args;
}

export function buildRecipeArgs(subcommand: string, extra: Record<string, string>): string[] {
  const args = ["recipe", subcommand];
  for (const [key, value] of Object.entries(extra)) {
    appendOptionalRaw(args, key, value);
  }
  return args;
}

export function buildCloudArgs(subcommand: string, extra: Record<string, string>): string[] {
  const args = ["cloud", subcommand];
  for (const [key, value] of Object.entries(extra)) {
    appendOptionalRaw(args, key, value);
  }
  return args;
}

export function buildCostArgs(subcommand: string, extra: Record<string, string>): string[] {
  const args = ["cost", subcommand];
  for (const [key, value] of Object.entries(extra)) {
    appendOptionalRaw(args, key, value);
  }
  return args;
}

export function buildSuggestArgs(extra: Record<string, string>): string[] {
  const args = ["suggest"];
  for (const [key, value] of Object.entries(extra)) {
    appendOptionalRaw(args, key, value);
  }
  return args;
}

export function buildSyntheticArgs(extra: Record<string, string>): string[] {
  const args = ["synthetic"];
  for (const [key, value] of Object.entries(extra)) {
    appendOptionalRaw(args, key, value);
  }
  return args;
}

export function buildAbChatArgs(extra: Record<string, string>): string[] {
  const args = ["ab-chat"];
  for (const [key, value] of Object.entries(extra)) {
    appendOptionalRaw(args, key, value);
  }
  return args;
}
