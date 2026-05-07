import { listClusters } from "../../api/remoteApi";
import type { RemoteInferenceConfig } from "../../hooks/useRemoteChatConfig";
import type { ChatMessage } from "./chatPersistence";

export const SAMPLING_PRESETS = {
  deterministic: { maxNewTokens: "80", temperature: "0", topK: "0" },
  balanced: { maxNewTokens: "120", temperature: "0.7", topK: "40" },
  creative: { maxNewTokens: "160", temperature: "1.0", topK: "80" },
} as const;

export interface LocalChatArgsOptions {
  dataset: string;
  tokenizer: string;
  model: string;
  prompt: string;
  maxTokens: string;
  temperature: string;
  topK: string;
  maxTokenLength: string;
  positionEmbeddingType: string;
  weights: string;
}

export interface RemoteChatArgsOptions {
  cluster: string;
  modelPath: string;
  prompt: string;
  maxTokens: string;
  temperature: string;
  topK: string;
  resources: RemoteInferenceConfig;
}

export function extractStatusLine(stderr: string): string {
  const lines = stderr.split("\n").filter((line) => line.startsWith("CRUCIBLE_"));
  return lines.length > 0 ? lines[lines.length - 1].replace(/^CRUCIBLE_\w+:\s*/, "") : "";
}

export function formatChatFailure(stderr: string): string {
  const cleaned = stderr
    .replace(/\r?Loading weights:[^\n\r]*/g, "")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .filter((line) => !line.includes("Materializing param="))
    .filter((line) => !line.includes("resource_tracker:"));
  if (cleaned.length === 0) return "Chat command failed.";
  return cleaned.slice(-6).join("\n");
}

export function buildPromptText(_messages: ChatMessage[], currentText: string): string {
  return currentText;
}

export async function resolveClusterName(dataRoot: string, host: string): Promise<string> {
  const clusters = await listClusters(dataRoot);
  const match = clusters.find((cluster) => cluster.host === host);
  if (!match) {
    throw new Error(`No registered cluster found for host "${host}".`);
  }
  return match.name;
}

export function buildRemoteChatArgs(options: RemoteChatArgsOptions): string[] {
  const args = [
    "remote", "chat",
    "--cluster", options.cluster,
    "--model-path", options.modelPath.trim(),
    "--prompt", options.prompt,
    "--max-new-tokens", options.maxTokens.trim() || "80",
    "--temperature", options.temperature.trim() || "0.7",
    "--top-k", options.topK.trim() || "40",
  ];
  if (options.resources.partition) args.push("--partition", options.resources.partition);
  if (options.resources.gpuType) args.push("--gpu-type", options.resources.gpuType);
  if (options.resources.memory) args.push("--memory", options.resources.memory);
  if (options.resources.timeLimit) args.push("--time-limit", options.resources.timeLimit);
  return args;
}

export function buildChatArgs(options: LocalChatArgsOptions): string[] {
  const args = ["chat", "--model-path", options.model.trim(), "--prompt", options.prompt];
  const optionals: [string, string][] = [
    ["--dataset", options.dataset],
    ["--tokenizer-path", options.tokenizer],
    ["--max-new-tokens", options.maxTokens],
    ["--temperature", options.temperature],
    ["--top-k", options.topK],
    ["--max-token-length", options.maxTokenLength],
    ["--position-embedding-type", options.positionEmbeddingType],
    ["--weights-path", options.weights],
  ];
  for (const [flag, value] of optionals) {
    if (value.trim()) args.push(flag, value.trim());
  }
  return args;
}
