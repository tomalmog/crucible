import { getJobResult } from "./jobsApi";
import type { AgentJobPreview } from "../types/agent";
import type { TrainingBatchLoss, TrainingEpoch, TrainingHistory } from "../types";
import type { JobRecord } from "../types/jobs";

const TRAINING_JOB_TYPES = new Set([
  "train", "sft", "dpo-train", "rlhf-train", "lora-train",
  "distill", "domain-adapt", "grpo-train", "qlora-train",
  "kto-train", "orpo-train", "multimodal-train", "rlvr-train",
]);

interface AgentJobCompletion {
  content: string;
  modelPath: string | null;
  preview: AgentJobPreview | null;
}

interface EvalBenchmark {
  name: string;
  num_examples: number;
  correct: number;
  error?: string;
}

const RESULT_ATTEMPTS = 3;
const RESULT_RETRY_MS = 1200;

export async function loadAgentJobCompletion(
  dataRoot: string,
  job: JobRecord,
): Promise<AgentJobCompletion> {
  const result = await loadResultWithRetry(dataRoot, job);
  const preview = buildAgentJobPreview(job, result);
  const modelPath = readString(result.model_path) ?? job.modelPath ?? null;
  return {
    content: buildCompletionMessage(job, preview),
    modelPath,
    preview,
  };
}

async function loadResultWithRetry(
  dataRoot: string,
  job: JobRecord,
): Promise<Record<string, unknown>> {
  for (let attempt = 0; attempt < RESULT_ATTEMPTS; attempt += 1) {
    const result = await getJobResult(dataRoot, job.jobId, job.state);
    if (Object.keys(result).length > 0) {
      return result;
    }
    if (attempt < RESULT_ATTEMPTS - 1) {
      await new Promise((resolve) => window.setTimeout(resolve, RESULT_RETRY_MS));
    }
  }
  return {};
}

function buildAgentJobPreview(
  job: JobRecord,
  result: Record<string, unknown>,
): AgentJobPreview | null {
  if (TRAINING_JOB_TYPES.has(job.jobType)) {
    return buildTrainingPreview(job, result);
  }
  if (job.jobType === "eval") {
    return buildEvalPreview(job, result);
  }
  return buildInterpPreview(job, result);
}

function buildTrainingPreview(
  job: JobRecord,
  result: Record<string, unknown>,
): AgentJobPreview | null {
  const history = readTrainingHistory(result.training_history);
  if (!history || history.epochs.length === 0) {
    return null;
  }
  const finalEpoch = history.epochs[history.epochs.length - 1];
  return {
    kind: "training",
    jobId: job.jobId,
    title: job.label || "Training completed",
    jobType: job.jobType,
    cluster: job.backendCluster || null,
    history,
    finalTrainLoss: finalEpoch.train_loss,
    finalValidationLoss: Number.isFinite(finalEpoch.validation_loss)
      ? finalEpoch.validation_loss
      : null,
    modelPath: readString(result.model_path) ?? job.modelPath ?? null,
  };
}

function buildEvalPreview(
  job: JobRecord,
  result: Record<string, unknown>,
): AgentJobPreview | null {
  const benchmarks = readEvalBenchmarks(result.benchmarks);
  if (benchmarks.length === 0) {
    return null;
  }
  const successful = benchmarks.filter((benchmark) => !benchmark.error && benchmark.num_examples > 0);
  if (successful.length === 0) {
    return null;
  }
  const topBenchmarks = [...successful]
    .map((benchmark) => ({
      name: benchmark.name,
      score: (benchmark.correct / benchmark.num_examples) * 100,
    }))
    .sort((left, right) => right.score - left.score)
    .slice(0, 3);
  const scoredBenchmarks = successful
    .map((benchmark) => ({
      name: benchmark.name,
      score: (benchmark.correct / benchmark.num_examples) * 100,
      correct: benchmark.correct,
      numExamples: benchmark.num_examples,
    }))
    .sort((left, right) => right.score - left.score);
  const averageScore = topBenchmarks.length > 0
    ? successful.reduce((sum, benchmark) => sum + (benchmark.correct / benchmark.num_examples) * 100, 0) / successful.length
    : 0;
  return {
    kind: "eval",
    jobId: job.jobId,
    title: job.label || "Evaluation completed",
    cluster: job.backendCluster || null,
    averageScore,
    benchmarkCount: successful.length,
    topBenchmarks,
    benchmarks: scoredBenchmarks,
  };
}

function buildInterpPreview(
  job: JobRecord,
  result: Record<string, unknown>,
): AgentJobPreview | null {
  const summaryLines = buildInterpSummary(job.jobType, result);
  if (summaryLines.length === 0) {
    return null;
  }
  return {
    kind: "interp",
    jobId: job.jobId,
    title: job.label || "Interpretability completed",
    cluster: job.backendCluster || null,
    jobType: readString(result.job_type) ?? job.jobType,
    summaryLines,
    result,
  };
}

function buildCompletionMessage(
  job: JobRecord,
  preview: AgentJobPreview | null,
): string {
  const clusterSuffix = job.backendCluster ? ` on ${job.backendCluster}` : "";
  if (!preview) {
    return `Job \`${job.jobId}\` completed${clusterSuffix}. Continuing automatically.`;
  }
  if (preview.kind === "training") {
    return `Training finished${clusterSuffix}. Here’s the loss curve before I continue.`;
  }
  if (preview.kind === "eval") {
    return `Evaluation finished${clusterSuffix}. Here’s the current score snapshot before I continue.`;
  }
  return `Interpretability finished${clusterSuffix}. Here’s the high-signal summary before I continue.`;
}

function buildInterpSummary(jobType: string, result: Record<string, unknown>): string[] {
  switch (jobType) {
    case "logit-lens":
      return [
        summarizeCount(readArray(result.input_tokens)?.length, "input token"),
        summarizeCount(readArray(result.layers)?.length, "layer"),
      ].filter(Boolean) as string[];
    case "activation-pca":
      return [
        summarizeLabel("Layer", readString(result.layer_name)),
        summarizeCount(readArray(result.points)?.length, "projection point"),
      ].filter(Boolean) as string[];
    case "activation-patch":
      return [
        summarizeLabel("Metric", readString(result.metric)),
        summarizeNumber("Clean score", readNumber(result.clean_metric)),
        summarizeNumber("Corrupted score", readNumber(result.corrupted_metric)),
      ].filter(Boolean) as string[];
    case "linear-probe":
      return [summarizeCount(readArray(result.layers)?.length, "probe layer")].filter(Boolean) as string[];
    case "sae-train":
      return [
        summarizeNumber("Final loss", readNumber(result.final_loss)),
        summarizeNumber("Latent dim", readNumber(result.latent_dim)),
      ].filter(Boolean) as string[];
    case "sae-analyze":
      return [
        summarizeFraction(
          "Active features",
          readNumber(result.active_features),
          readNumber(result.total_features),
        ),
        summarizeNumber("Reconstruction error", readNumber(result.reconstruction_error)),
      ].filter(Boolean) as string[];
    case "steer-compute":
      return [
        summarizeNumber("Vector norm", readNumber(result.vector_norm)),
        summarizeNumber("Cosine similarity", readNumber(result.cosine_similarity)),
      ].filter(Boolean) as string[];
    case "steer-apply":
      return [
        summarizeNumber("Coefficient", readNumber(result.coefficient)),
        summarizeLabel("Preview", readString(result.steered_text)),
      ].filter(Boolean) as string[];
    default:
      return [];
  }
}

function summarizeCount(count: number | undefined, noun: string): string | null {
  if (count == null) return null;
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

function summarizeFraction(
  label: string,
  numerator: number | undefined,
  denominator: number | undefined,
): string | null {
  if (numerator == null || denominator == null) return null;
  return `${label}: ${numerator} / ${denominator}`;
}

function summarizeLabel(label: string, value: string | undefined): string | null {
  if (!value) return null;
  return `${label}: ${truncate(value)}`;
}

function summarizeNumber(label: string, value: number | undefined): string | null {
  if (value == null) return null;
  return `${label}: ${value.toFixed(3)}`;
}

function truncate(value: string): string {
  return value.length > 72 ? `${value.slice(0, 69)}...` : value;
}

function readTrainingHistory(value: unknown): TrainingHistory | null {
  if (!isRecord(value)) return null;
  const epochsValue = value.epochs;
  const batchLossesValue = value.batch_losses;
  if (!Array.isArray(epochsValue) || !Array.isArray(batchLossesValue)) {
    return null;
  }
  const epochs = epochsValue.filter(isTrainingEpoch);
  const batchLosses = batchLossesValue.filter(isTrainingBatchLoss);
  if (epochs.length === 0) {
    return null;
  }
  return { epochs, batch_losses: batchLosses };
}

function readEvalBenchmarks(value: unknown): EvalBenchmark[] {
  if (!Array.isArray(value)) return [];
  return value.filter(isEvalBenchmark);
}

function isTrainingEpoch(value: unknown): value is TrainingEpoch {
  if (!isRecord(value)) return false;
  return typeof value.epoch === "number"
    && typeof value.train_loss === "number"
    && typeof value.validation_loss === "number";
}

function isTrainingBatchLoss(value: unknown): value is TrainingBatchLoss {
  if (!isRecord(value)) return false;
  return typeof value.epoch === "number"
    && typeof value.batch_index === "number"
    && typeof value.global_step === "number"
    && typeof value.train_loss === "number";
}

function isEvalBenchmark(value: unknown): value is EvalBenchmark {
  if (!isRecord(value)) return false;
  return typeof value.name === "string"
    && typeof value.num_examples === "number"
    && typeof value.correct === "number"
    && (value.error === undefined || typeof value.error === "string");
}

function readString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function readNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function readArray(value: unknown): unknown[] | undefined {
  return Array.isArray(value) ? value : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
