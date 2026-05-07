import type { ModelEntry } from "../types/models";

export function buildAgentChatContext(
  currentPage: string,
  selectedModel: string | null,
  selectedDataset: string | null,
  models: ModelEntry[],
  datasets: { name: string }[],
): Record<string, unknown> {
  const context: Record<string, unknown> = {
    currentPage,
    selectedModel,
    selectedDataset,
    modelNames: models.map((model) => model.modelName).slice(0, 20),
    modelPaths: models.slice(0, 20).reduce<Record<string, string>>((acc, model) => {
      if (model.modelPath) acc[model.modelName] = model.modelPath;
      if (model.remotePath) acc[`${model.modelName} (remote)`] = model.remotePath;
      return acc;
    }, {}),
    datasetNames: datasets.map((dataset) => dataset.name).slice(0, 20),
  };
  return context;
}

interface ChainPromptInput {
  jobId: string;
  jobState: string | null;
  jobModelPath: string | null;
  jobModelName: string | null;
  steps: string[];
}

export function buildChainContinuationPrompt(chain: ChainPromptInput): string {
  const parts = [`[Chain continuation] Job ${chain.jobId} finished with state: ${chain.jobState}.`];
  if (chain.jobModelPath) parts.push(`Output model path: ${chain.jobModelPath}`);
  if (chain.jobModelName) parts.push(`Registered model name: ${chain.jobModelName}`);
  parts.push(`\nPlease proceed with the next step: ${chain.steps[0]}`);
  if (chain.steps.length > 1) {
    parts.push(`\nRemaining steps after this: ${chain.steps.slice(1).join("; ")}`);
  }
  return parts.join("\n");
}
