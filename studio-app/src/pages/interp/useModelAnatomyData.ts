import { useEffect, useMemo, useState } from "react";
import { getModelArchitecture } from "../../api/studioApi";
import { useCrucible } from "../../context/CrucibleContext";
import { useUnifiedJobs } from "../../hooks/useUnifiedJobs";
import type { JobRecord } from "../../types/jobs";
import type { ModelEntry } from "../../types/models";
import type {
  ModelAnatomyData,
  ModelAnatomyLayer,
  ModelArchitectureConfig,
  ModelLayerEvidence,
} from "./modelAnatomyTypes";

const INTERP_JOB_TYPES = new Set([
  "logit-lens",
  "activation-pca",
  "activation-patch",
  "linear-probe",
  "sae-train",
  "sae-analyze",
  "steer-compute",
  "steer-apply",
]);

export function useModelAnatomyData(): ModelAnatomyData | null {
  const { dataRoot, selectedModel } = useCrucible();
  const { jobs } = useUnifiedJobs(dataRoot);
  const [architecture, setArchitecture] = useState<ModelArchitectureConfig | null>(null);

  // Reload architecture metadata when the selected local model changes.
  // Remote-only models may not have readable config files on this machine.
  useEffect(() => {
    let cancelled = false;
    setArchitecture(null);
    if (!selectedModel?.hasLocal || !selectedModel.modelPath) {
      return () => { cancelled = true; };
    }
    getModelArchitecture(selectedModel.modelPath)
      .then((config) => {
        if (!cancelled) setArchitecture(toArchitectureConfig(config));
      })
      .catch(() => {
        if (!cancelled) setArchitecture(null);
      });
    return () => { cancelled = true; };
  }, [selectedModel]);

  return useMemo(() => {
    if (!selectedModel) return null;
    return buildAnatomyData(selectedModel, architecture, jobs);
  }, [architecture, jobs, selectedModel]);
}

function buildAnatomyData(
  model: ModelEntry,
  architecture: ModelArchitectureConfig | null,
  jobs: JobRecord[],
): ModelAnatomyData {
  const layerCount = readFirstNumber(architecture, ["num_layers", "num_hidden_layers", "n_layer"]) ?? estimateLayers(model);
  const hiddenSize = readFirstNumber(architecture, ["hidden_dim", "hidden_size", "n_embd", "d_model"]);
  const attentionHeads = readFirstNumber(architecture, ["attention_heads", "num_attention_heads", "n_head"]);
  const evidenceByLayer = collectLayerEvidence(model, jobs);
  const layers = Array.from({ length: layerCount }, (_, index) => makeLayer(index, architecture, evidenceByLayer));
  return {
    architectureLabel: architecture?.model_type ?? architecture?.architectures?.[0] ?? "estimated layer map",
    attentionHeads,
    hiddenSize,
    layerCount,
    layers,
    locationLabel: model.hasLocal ? "local registry model" : "remote registry model",
    modelName: model.modelName,
    parameterLabel: formatBytesAsParams(model.sizeBytes),
    statusLabel: architecture ? "architecture config loaded" : "estimated from registry",
  };
}

function makeLayer(
  index: number,
  architecture: ModelArchitectureConfig | null,
  evidenceByLayer: Map<number, ModelLayerEvidence[]>,
): ModelAnatomyLayer {
  const isMoe = Number(architecture?.num_experts ?? 0) > 0;
  return {
    index,
    kind: isMoe ? "moe" : "block",
    label: `L${String(index).padStart(2, "0")}`,
    detail: isMoe ? "block with MoE metadata" : "transformer block",
    evidence: evidenceByLayer.get(index) ?? [],
  };
}

function collectLayerEvidence(model: ModelEntry, jobs: JobRecord[]): Map<number, ModelLayerEvidence[]> {
  const byLayer = new Map<number, ModelLayerEvidence[]>();
  for (const job of jobs) {
    if (job.state !== "completed" || !INTERP_JOB_TYPES.has(job.jobType) || !matchesModel(job, model)) continue;
    for (const item of readEvidence(job)) {
      const existing = byLayer.get(Number(item.label.replace("L", ""))) ?? [];
      existing.push(item);
      byLayer.set(Number(item.label.replace("L", "")), existing);
    }
  }
  return byLayer;
}

function readEvidence(job: JobRecord): ModelLayerEvidence[] {
  const parsed = parseJson(job.stdout);
  if (!isRecord(parsed)) return [];
  if (Array.isArray(parsed.layers)) {
    return parsed.layers.flatMap((layer) => layerEvidence(job, layer));
  }
  if (Array.isArray(parsed.layer_results)) {
    return parsed.layer_results.flatMap((layer) => layerEvidence(job, layer));
  }
  return layerEvidence(job, parsed);
}

function layerEvidence(job: JobRecord, value: unknown): ModelLayerEvidence[] {
  if (!isRecord(value)) return [];
  const layer = typeof value.layer_index === "number"
    ? value.layer_index
    : typeof value.layer_name === "string" ? parseLayerIndex(value.layer_name) : null;
  if (layer === null) return [];
  const metric = typeof value.recovery === "number"
    ? value.recovery
    : typeof value.accuracy === "number" ? value.accuracy : undefined;
  return [{ jobId: job.jobId, jobType: job.jobType, label: `L${layer}`, metric }];
}

function parseLayerIndex(layerName: string): number | null {
  const match = layerName.match(/(?:^|\.)(\d+)(?:\.|$)/);
  if (!match) return null;
  return Number.parseInt(match[1], 10);
}

function matchesModel(job: JobRecord, model: ModelEntry): boolean {
  return job.modelName === model.modelName
    || job.modelPath === model.modelPath
    || job.modelPathLocal === model.modelPath
    || job.config.model_path === model.modelPath
    || job.config.modelPath === model.modelPath;
}

function toArchitectureConfig(value: Record<string, unknown> | null): ModelArchitectureConfig | null {
  if (!value) return null;
  return {
    architectures: readStringArray(value.architectures),
    model_type: readString(value.model_type),
    hidden_dim: readNumber(value.hidden_dim),
    hidden_size: readNumber(value.hidden_size),
    n_embd: readNumber(value.n_embd),
    d_model: readNumber(value.d_model),
    num_layers: readNumber(value.num_layers),
    num_hidden_layers: readNumber(value.num_hidden_layers),
    n_layer: readNumber(value.n_layer),
    attention_heads: readNumber(value.attention_heads),
    num_attention_heads: readNumber(value.num_attention_heads),
    n_head: readNumber(value.n_head),
    num_experts: readNumber(value.num_experts),
    torch_dtype: readString(value.torch_dtype),
  };
}

function parseJson(value: string): unknown {
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function readFirstNumber(
  config: ModelArchitectureConfig | null,
  keys: readonly (keyof ModelArchitectureConfig)[],
): number | null {
  if (!config) return null;
  for (const key of keys) {
    const value = config[key];
    if (typeof value === "number") return value;
  }
  return null;
}

function estimateLayers(model: ModelEntry): number {
  const name = model.modelName.toLowerCase();
  if (name.includes("tiny")) return 2;
  if (name.includes("distil")) return 6;
  if (name.includes("gpt2")) return 12;
  if (name.includes("7b") || name.includes("8b")) return 32;
  return 12;
}

function formatBytesAsParams(sizeBytes: number): string {
  if (sizeBytes <= 0) return "size unknown";
  const params = sizeBytes / 2;
  if (params >= 1_000_000_000) return `${(params / 1_000_000_000).toFixed(1)}B est. params`;
  if (params >= 1_000_000) return `${(params / 1_000_000).toFixed(0)}M est. params`;
  return `${Math.round(params).toLocaleString()} est. params`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function readNumber(value: unknown): number | undefined {
  return typeof value === "number" ? value : undefined;
}

function readString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function readStringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const strings = value.filter((entry): entry is string => typeof entry === "string");
  return strings.length > 0 ? strings : undefined;
}
