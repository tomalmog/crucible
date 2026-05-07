export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export type SamplingPreset = "deterministic" | "balanced" | "creative" | "custom";

export interface ChatScreenState {
  datasetName: string;
  tokenizerPath: string;
  weightsPath: string;
  modelPath: string;
  maxNewTokens: string;
  temperature: string;
  topK: string;
  maxTokenLength: string;
  positionEmbeddingType: string;
  samplingPreset: SamplingPreset;
  draftMessage: string;
}

const CHAT_MESSAGES_KEY = "crucible_chat_messages";
const CHAT_STATE_KEY = "crucible_chat_screen_state_v1";
const LEGACY_MODEL_KEY = "crucible_chat_model";

export const DEFAULT_CHAT_SCREEN_STATE: ChatScreenState = {
  datasetName: "",
  tokenizerPath: "",
  weightsPath: "",
  modelPath: "",
  maxNewTokens: "120",
  temperature: "0.7",
  topK: "40",
  maxTokenLength: "256",
  positionEmbeddingType: "learned",
  samplingPreset: "balanced",
  draftMessage: "",
};

export function loadChatMessages(): ChatMessage[] {
  const rawValue = readStorageValue(CHAT_MESSAGES_KEY);
  if (!rawValue) return [];
  try {
    const parsed = JSON.parse(rawValue);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isChatMessage);
  } catch {
    return [];
  }
}

export function saveChatMessages(messages: ChatMessage[]): void {
  writeLocalStorageValue(CHAT_MESSAGES_KEY, JSON.stringify(messages));
}

export function loadChatScreenState(): ChatScreenState {
  const parsed = parseStoredScreenState(readStorageValue(CHAT_STATE_KEY));
  const legacyModel = readStorageValue(LEGACY_MODEL_KEY) ?? "";
  return {
    ...DEFAULT_CHAT_SCREEN_STATE,
    ...parsed,
    modelPath: parsed.modelPath || legacyModel,
  };
}

export function saveChatScreenState(state: ChatScreenState): void {
  writeLocalStorageValue(CHAT_STATE_KEY, JSON.stringify(state));
}

function readStorageValue(key: string): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(key) ?? window.sessionStorage.getItem(key);
}

function writeLocalStorageValue(key: string, value: string): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(key, value);
}

function parseStoredScreenState(rawValue: string | null): Partial<ChatScreenState> {
  if (!rawValue) return {};
  try {
    const parsed = JSON.parse(rawValue);
    if (!isRecord(parsed)) return {};
    return {
      datasetName: asString(parsed.datasetName),
      tokenizerPath: asString(parsed.tokenizerPath),
      weightsPath: asString(parsed.weightsPath),
      modelPath: asString(parsed.modelPath),
      maxNewTokens: asString(parsed.maxNewTokens),
      temperature: asString(parsed.temperature),
      topK: asString(parsed.topK),
      maxTokenLength: asString(parsed.maxTokenLength),
      positionEmbeddingType: asString(parsed.positionEmbeddingType),
      samplingPreset: asSamplingPreset(parsed.samplingPreset),
      draftMessage: asString(parsed.draftMessage),
    };
  } catch {
    return {};
  }
}

function isChatMessage(value: unknown): value is ChatMessage {
  if (!isRecord(value)) return false;
  return (
    (value.role === "user" || value.role === "assistant")
    && typeof value.content === "string"
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function asSamplingPreset(value: unknown): SamplingPreset | undefined {
  if (
    value === "deterministic"
    || value === "balanced"
    || value === "creative"
    || value === "custom"
  ) {
    return value;
  }
  return undefined;
}
