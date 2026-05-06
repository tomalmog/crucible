import type { AgentTraceEvent } from "../types/agent";

const EVENT_PREFIX = "CRUCIBLE_AGENT_EVENT:";
const RESULT_PREFIX = "CRUCIBLE_AGENT_RESULT:";

interface ParsedAgentStdout {
  events: AgentTraceEvent[];
  result: Record<string, unknown> | null;
}

export function parseAgentStdout(stdout: string): ParsedAgentStdout {
  const events: AgentTraceEvent[] = [];
  let result: Record<string, unknown> | null = null;
  for (const rawLine of stdout.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) continue;
    if (line.startsWith(EVENT_PREFIX)) {
      const payload = parseJson(line.slice(EVENT_PREFIX.length));
      if (hasAgentTraceEventShape(payload)) {
        events.push(payload);
      }
      continue;
    }
    if (line.startsWith(RESULT_PREFIX)) {
      result = parseJson(line.slice(RESULT_PREFIX.length));
    }
  }
  return { events, result };
}

export function parseAgentResult(stdout: string): Record<string, unknown> {
  const parsed = parseAgentStdout(stdout);
  if (parsed.result) {
    return parsed.result;
  }
  const fallback = parseJson(stdout.trim());
  if (fallback) {
    return fallback;
  }
  throw new Error(`Invalid agent response: ${stdout.trim().slice(0, 200)}`);
}

export function summarizeAgentError(stderr: string): string {
  const lines = stderr
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  return lines.length > 0 ? lines[lines.length - 1] : "Agent command failed";
}

function parseJson(raw: string): Record<string, unknown> | null {
  try {
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
      // Safe after the runtime object/null/array checks above.
      return parsed as Record<string, unknown>;
    }
    return null;
  } catch {
    return null;
  }
}

function hasAgentTraceEventShape(value: unknown): value is AgentTraceEvent {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  // Safe after the runtime object/null/array checks above.
  const record = value as Record<string, unknown>;
  if (typeof record.type !== "string") {
    return false;
  }
  return [
    "status",
    "assistant_note",
    "tool_call",
    "tool_result",
    "navigation",
    "pending_chain",
  ].includes(record.type);
}
