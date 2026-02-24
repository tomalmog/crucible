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

export function buildSharedTrainingArgs(config: SharedTrainingConfig): string[] {
  const args: string[] = [];
  appendOptionalRaw(args, "--dataset", config.dataset);
  appendOptionalRaw(args, "--version-id", config.versionId);
  appendOptionalRaw(args, "--epochs", config.epochs);
  appendOptionalRaw(args, "--learning-rate", config.learningRate);
  appendOptionalRaw(args, "--batch-size", config.batchSize);
  appendOptionalRaw(args, "--optimizer", config.optimizer);
  appendOptionalRaw(args, "--precision", config.precision);
  appendOptionalRaw(args, "--output-dir", config.outputDir);
  appendOptionalRaw(args, "--max-token-length", config.maxTokenLength);
  appendOptionalRaw(args, "--embedding-dim", config.embeddingDim);
  appendOptionalRaw(args, "--num-heads", config.numHeads);
  appendOptionalRaw(args, "--num-layers", config.numLayers);
  appendOptional(args, "--checkpoint-every", config.checkpointEvery);
  return args;
}

export function buildTrainingArgs(
  method: TrainingMethod,
  shared: SharedTrainingConfig,
  extra: Record<string, string>,
): string[] {
  const args = [method, ...buildSharedTrainingArgs(shared)];
  for (const [key, value] of Object.entries(extra)) {
    appendOptionalRaw(args, key, value);
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
