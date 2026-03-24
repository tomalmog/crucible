import { SharedTrainingConfig, TrainingMethod } from "../types/training";
import type { ClusterSubmitConfig } from "../pages/training/ClusterSubmitSection";
import type { BackendKind, ResourceConfig } from "../types/jobs";

function appendOptional(args: string[], flag: string, value: string | undefined): void {
  const trimmed = (value ?? "").trim();
  if (trimmed.length > 0 && trimmed !== "0") {
    args.push(flag, trimmed);
  }
}

function appendOptionalRaw(args: string[], flag: string, value: string | undefined): void {
  const trimmed = (value ?? "").trim();
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
  appendOptionalRaw(args, "--wandb-project", config.wandbProject);
  appendOptionalRaw(args, "--tensorboard-dir", config.tensorboardDir);
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


export function buildModelArgs(subcommand: string, extra: Record<string, string>): string[] {
  const args = ["model", subcommand];
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


/** Flag-to-dataclass-field mapping for extra (method-specific) flags. */
const EXTRA_FLAG_TO_FIELD: Record<string, string> = {
  "--dataset": "dataset_name",
  "--base-model": "base_model",
  "--base-model-path": "base_model_path",
  "--policy-model-path": "policy_model_path",
  "--teacher-model-path": "teacher_model_path",
  "--train-reward-model": "train_reward_model",
};

function flagToField(flag: string): string {
  if (EXTRA_FLAG_TO_FIELD[flag]) return EXTRA_FLAG_TO_FIELD[flag];
  return flag.replace(/^--/, "").replace(/-/g, "_");
}

function addIfPresent(out: Record<string, unknown>, key: string, raw: string | undefined, parse?: (v: string) => unknown): void {
  const v = (raw ?? "").trim();
  if (!v) return;
  out[key] = parse ? parse(v) : v;
}

/** Convert the wizard's shared + extra state into a JSON-serializable dict
 *  with Python dataclass field names for remote dispatch. */
export function buildRemoteMethodArgs(
  shared: SharedTrainingConfig,
  extra: Record<string, string>,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  const int = (v: string) => parseInt(v, 10);
  const float = (v: string) => parseFloat(v);

  addIfPresent(out, "epochs", shared.epochs, int);
  addIfPresent(out, "learning_rate", shared.learningRate, float);
  addIfPresent(out, "batch_size", shared.batchSize, int);
  addIfPresent(out, "optimizer_type", shared.optimizer);
  addIfPresent(out, "precision_mode", shared.precision);
  addIfPresent(out, "output_dir", shared.outputDir);
  addIfPresent(out, "max_token_length", shared.maxTokenLength, int);
  addIfPresent(out, "hidden_dim", shared.embeddingDim, int);
  addIfPresent(out, "attention_heads", shared.numHeads, int);
  addIfPresent(out, "num_layers", shared.numLayers, int);
  addIfPresent(out, "checkpoint_every_epochs", shared.checkpointEvery, (v) => Math.max(1, int(v)));
  addIfPresent(out, "mlp_hidden_dim", shared.mlpHiddenDim, int);
  addIfPresent(out, "mlp_layers", shared.mlpLayers, int);
  addIfPresent(out, "resume_checkpoint_path", shared.resumeCheckpointPath);

  for (const [flag, value] of Object.entries(extra)) {
    const v = (value ?? "").trim();
    if (!v) continue;
    if (BOOLEAN_FLAGS.has(flag)) {
      if (v === "true") out[flagToField(flag)] = true;
    } else {
      out[flagToField(flag)] = v;
    }
  }

  return out;
}

/** Build CLI args for `crucible remote eval-submit ...`. */
export function buildRemoteEvalArgs(
  cluster: string,
  modelPath: string,
  benchmarks: string,
  opts: {
    baseModel?: string;
    maxSamples?: string;
    modelName?: string;
    partition?: string;
    gpusPerNode?: string;
    gpuType?: string;
    cpusPerTask?: string;
    memory?: string;
    timeLimit?: string;
  } = {},
): string[] {
  const args = [
    "remote", "eval-submit",
    "--cluster", cluster,
    "--model-path", modelPath,
    "--benchmarks", benchmarks,
  ];
  if (opts.modelName) args.push("--model-name", opts.modelName);
  if (opts.baseModel) args.push("--base-model", opts.baseModel);
  if (opts.maxSamples) args.push("--max-samples", opts.maxSamples);
  if (opts.partition) args.push("--partition", opts.partition);
  if (opts.gpusPerNode) args.push("--gpus-per-node", opts.gpusPerNode);
  if (opts.gpuType) args.push("--gpu-type", opts.gpuType);
  if (opts.cpusPerTask) args.push("--cpus-per-task", opts.cpusPerTask);
  if (opts.memory) args.push("--memory", opts.memory);
  if (opts.timeLimit) args.push("--time-limit", opts.timeLimit);
  return args;
}

/** Build CLI args for `crucible remote interp-submit ...`. */
export function buildRemoteInterpArgs(
  cluster: string,
  interpMethod: string,
  methodArgsJson: string,
  opts: {
    partition?: string;
    gpusPerNode?: string;
    gpuType?: string;
    cpusPerTask?: string;
    memory?: string;
    timeLimit?: string;
  } = {},
): string[] {
  const args = [
    "remote", "interp-submit",
    "--cluster", cluster,
    "--interp-method", interpMethod,
    "--method-args", methodArgsJson,
  ];
  if (opts.partition) args.push("--partition", opts.partition);
  if (opts.gpusPerNode) args.push("--gpus-per-node", opts.gpusPerNode);
  if (opts.gpuType) args.push("--gpu-type", opts.gpuType);
  if (opts.cpusPerTask) args.push("--cpus-per-task", opts.cpusPerTask);
  if (opts.memory) args.push("--memory", opts.memory);
  if (opts.timeLimit) args.push("--time-limit", opts.timeLimit);
  return args;
}

/** Canonical method args dict — alias for buildRemoteMethodArgs. */
export const buildMethodArgs = buildRemoteMethodArgs;

/** Build CLI args for `crucible dispatch --spec <json>`. */
export function buildDispatchSpec(
  jobType: string,
  methodArgs: Record<string, unknown>,
  backend: BackendKind,
  opts?: {
    label?: string;
    clusterName?: string;
    resources?: ResourceConfig;
    isSweep?: boolean;
    sweepTrials?: Record<string, unknown>[];
    config?: Record<string, unknown>;
  },
): string[] {
  const spec: Record<string, unknown> = {
    job_type: jobType,
    method_args: methodArgs,
    backend,
  };
  if (opts?.label) spec.label = opts.label;
  if (opts?.clusterName) spec.cluster_name = opts.clusterName;
  if (opts?.resources) spec.resources = opts.resources;
  if (opts?.isSweep) spec.is_sweep = true;
  if (opts?.sweepTrials) spec.sweep_trials = opts.sweepTrials;
  if (opts?.config) spec.config = opts.config;
  return ["dispatch", "--spec", JSON.stringify(spec)];
}

/** Build the full CLI args array for `crucible remote submit ...`. */
export function buildRemoteSubmitArgs(
  method: TrainingMethod,
  methodArgsJson: string,
  config: ClusterSubmitConfig,
  modelName?: string,
): string[] {
  const args = [
    "remote", "submit",
    "--cluster", config.cluster,
    "--method", method,
    "--method-args", methodArgsJson,
    "--nodes", config.nodes,
    "--gpus-per-node", config.gpusPerNode,
    "--cpus-per-task", config.cpusPerTask,
    "--memory", config.memory,
    "--time-limit", config.timeLimit,
  ];
  if (config.partition) args.push("--partition", config.partition);
  if (config.gpuType) args.push("--gpu-type", config.gpuType);
  if (config.pullModel) args.push("--pull-model");
  if (modelName) args.push("--model-name", modelName);
  return args;
}
