import type { AgentJobPreview, AgentWorkspaceDirective } from "../types/agent";

export function inferPreviewWorkspace(
  preview: AgentJobPreview,
): AgentWorkspaceDirective {
  if (preview.kind === "eval") {
    return { mode: "compare", cards: ["previous_eval", "latest_eval", "context"] };
  }
  if (preview.kind === "interp") {
    return { mode: "focus", cards: ["latest_interp", "live_trace", "context"] };
  }
  return { mode: "focus", cards: ["latest_training", "live_trace", "context"] };
}

export function readWorkspaceDirective(
  result: Record<string, unknown>,
): AgentWorkspaceDirective | undefined {
  const mode = typeof result.workspace_mode === "string" ? result.workspace_mode : null;
  const cards = Array.isArray(result.workspace_cards)
    ? result.workspace_cards.filter((value): value is string => typeof value === "string")
    : [];
  if (!mode && cards.length === 0) {
    return undefined;
  }
  return {
    mode: isWorkspaceMode(mode) ? mode : "auto",
    cards: cards.filter(isWorkspaceCardSelector),
  };
}

function isWorkspaceMode(value: string | null): value is AgentWorkspaceDirective["mode"] {
  return value === "auto"
    || value === "focus"
    || value === "compare"
    || value === "board"
    || value === "plan";
}

function isWorkspaceCardSelector(
  value: string,
): value is AgentWorkspaceDirective["cards"][number] {
  return value === "artifact"
    || value === "context"
    || value === "latest"
    || value === "latest_training"
    || value === "latest_eval"
    || value === "previous_eval"
    || value === "latest_interp"
    || value === "live_trace"
    || value === "pending_chain"
    || value === "trace";
}
