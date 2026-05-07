export interface StoredAgentMessage {
  role: "user" | "assistant";
  content: string;
  toolsUsed?: string[];
  scriptUpdated?: boolean;
  navigatedTo?: string;
}

const AGENT_MESSAGES_KEY = "crucible_agent_chat_messages_v1";

export function loadAgentMessages(dataRoot: string): StoredAgentMessage[] {
  const rawValue = readLocalStorageValue(storageKey(dataRoot));
  return normalizeAgentMessages(parseJson(rawValue));
}

export function saveAgentMessages(dataRoot: string, messages: StoredAgentMessage[]): void {
  if (messages.length === 0) {
    removeLocalStorageValue(storageKey(dataRoot));
    return;
  }
  writeLocalStorageValue(storageKey(dataRoot), JSON.stringify(messages));
}

export function clearAgentMessages(dataRoot: string): void {
  removeLocalStorageValue(storageKey(dataRoot));
}

export function normalizeAgentMessages(value: unknown): StoredAgentMessage[] {
  if (!Array.isArray(value)) return [];
  return value.map(normalizeAgentMessage).filter((msg): msg is StoredAgentMessage => msg !== null);
}

function normalizeAgentMessage(value: unknown): StoredAgentMessage | null {
  if (!isRecord(value) || (value.role !== "user" && value.role !== "assistant")) return null;
  if (typeof value.content !== "string") return null;
  const tools = value.toolsUsed ?? value.tools_used;
  return {
    role: value.role,
    content: value.content,
    toolsUsed: Array.isArray(tools) ? tools.filter((tool) => typeof tool === "string") : undefined,
    scriptUpdated: value.scriptUpdated === true || value.script_updated === true,
    navigatedTo: typeof value.navigatedTo === "string"
      ? value.navigatedTo
      : typeof value.navigated_to === "string" ? value.navigated_to : undefined,
  };
}

function storageKey(dataRoot: string): string {
  return `${AGENT_MESSAGES_KEY}:${dataRoot}`;
}

function parseJson(rawValue: string | null): unknown {
  if (!rawValue) return null;
  try {
    return JSON.parse(rawValue);
  } catch {
    return null;
  }
}

function readLocalStorageValue(key: string): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(key);
}

function writeLocalStorageValue(key: string, value: string): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(key, value);
}

function removeLocalStorageValue(key: string): void {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(key);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
