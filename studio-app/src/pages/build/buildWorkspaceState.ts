import type { PendingChain } from "../../hooks/useAgentChat";
import type {
  AgentEvalJobPreview,
  AgentJobPreview,
  AgentMessage,
  AgentTraceEvent,
  AgentWorkspaceCardSelector,
  AgentWorkspaceDirective,
} from "../../types/agent";

export interface AgentArtifactCardModel {
  type: "artifact";
  id: string;
  artifact: AgentJobPreview;
  messageContent: string | null;
}

export interface AgentTraceCardModel {
  type: "trace";
  id: string;
  events: AgentTraceEvent[];
}

export interface AgentPendingCardModel {
  type: "pending_chain";
  id: string;
  pendingChain: PendingChain;
}

export interface AgentContextCardModel {
  type: "context";
  id: string;
}

export type AgentWorkspaceCardModel =
  | AgentArtifactCardModel
  | AgentTraceCardModel
  | AgentPendingCardModel
  | AgentContextCardModel;

export interface AgentWorkspaceState {
  mode: "focus" | "compare" | "board";
  highlightedCards: AgentWorkspaceCardModel[];
  boardCards: AgentWorkspaceCardModel[];
}

export function buildWorkspaceState(
  messages: AgentMessage[],
  currentTrace: AgentTraceEvent[],
  pendingChain: PendingChain | null,
): AgentWorkspaceState {
  const artifactCards = buildArtifactCards(messages);
  const latestTrace = currentTrace.length > 0
    ? currentTrace
    : [...messages].reverse().find((message) => message.trace?.length)?.trace ?? [];
  const traceCard: AgentTraceCardModel | null = latestTrace.length > 0
    ? { type: "trace", id: "trace-live", events: latestTrace }
    : null;
  const pendingCard: AgentPendingCardModel | null = pendingChain
    ? { type: "pending_chain", id: "pending-chain", pendingChain }
    : null;
  const contextCard: AgentContextCardModel = { type: "context", id: "run-context" };
  const latestDirective = [...messages]
    .reverse()
    .find((message) => message.role === "assistant" && message.workspaceDirective)
    ?.workspaceDirective;

  if (latestDirective) {
    const cards = resolveDirectiveCards(
      latestDirective,
      artifactCards,
      traceCard,
      pendingCard,
      contextCard,
    );
    if (cards.length > 0) {
      const mode = latestDirective.mode === "auto"
        ? inferDefaultMode(artifactCards)
        : normalizeWorkspaceMode(latestDirective.mode);
      return splitWorkspaceCards(mode, cards, artifactCards, traceCard, pendingCard, contextCard);
    }
  }

  const defaultMode = inferDefaultMode(artifactCards);
  if (defaultMode === "compare") {
    const latestPair = artifactCards.filter(isEvalArtifactCard).slice(-2);
    return splitWorkspaceCards(defaultMode, latestPair, artifactCards, traceCard, pendingCard, contextCard);
  }
  const latestArtifact = lastItem(artifactCards);
  return splitWorkspaceCards(
    defaultMode,
    latestArtifact ? [latestArtifact] : [contextCard],
    artifactCards,
    traceCard,
    pendingCard,
    contextCard,
  );
}

function buildArtifactCards(messages: AgentMessage[]): AgentArtifactCardModel[] {
  return messages.flatMap((message, index) => {
    if (message.role !== "assistant" || !message.artifact) {
      return [];
    }
    return [{
      type: "artifact",
      id: `artifact-${index}-${message.artifact.jobId}`,
      artifact: message.artifact,
      messageContent: message.content,
    }];
  });
}

function inferDefaultMode(artifactCards: AgentArtifactCardModel[]): "focus" | "compare" | "board" {
  const evalCards = artifactCards.filter(isEvalArtifactCard);
  const latestArtifact = lastItem(artifactCards);
  if (latestArtifact?.artifact.kind === "eval" && evalCards.length >= 2) {
    return "compare";
  }
  return latestArtifact ? "focus" : "board";
}

function resolveDirectiveCards(
  directive: AgentWorkspaceDirective,
  artifactCards: AgentArtifactCardModel[],
  traceCard: AgentTraceCardModel | null,
  pendingCard: AgentPendingCardModel | null,
  contextCard: AgentContextCardModel,
): AgentWorkspaceCardModel[] {
  const cards = directive.cards.flatMap((selector) =>
    resolveSelector(selector, artifactCards, traceCard, pendingCard, contextCard),
  );
  return dedupeCards(cards);
}

function resolveSelector(
  selector: AgentWorkspaceCardSelector,
  artifactCards: AgentArtifactCardModel[],
  traceCard: AgentTraceCardModel | null,
  pendingCard: AgentPendingCardModel | null,
  contextCard: AgentContextCardModel,
): AgentWorkspaceCardModel[] {
  switch (selector) {
    case "artifact":
    case "latest":
      return lastItem(artifactCards) ? [lastItem(artifactCards)!] : [];
    case "context":
      return [contextCard];
    case "latest_training":
      return findLastCard(artifactCards, isTrainingArtifactCard)
        ? [findLastCard(artifactCards, isTrainingArtifactCard)!]
        : [];
    case "latest_eval":
      return findLastCard(artifactCards, isEvalArtifactCard)
        ? [findLastCard(artifactCards, isEvalArtifactCard)!]
        : [];
    case "previous_eval": {
      const evalCards = artifactCards.filter(isEvalArtifactCard);
      return evalCards.length >= 2 ? [evalCards[evalCards.length - 2]] : [];
    }
    case "latest_interp":
      return findLastCard(artifactCards, isInterpArtifactCard)
        ? [findLastCard(artifactCards, isInterpArtifactCard)!]
        : [];
    case "trace":
    case "live_trace":
      return traceCard ? [traceCard] : [];
    case "pending_chain":
      return pendingCard ? [pendingCard] : [];
    default:
      return [contextCard];
  }
}

function normalizeWorkspaceMode(
  mode: AgentWorkspaceDirective["mode"],
): "focus" | "compare" | "board" {
  if (mode === "plan") {
    return "board";
  }
  if (mode === "focus" || mode === "compare" || mode === "board") {
    return mode;
  }
  return "board";
}

function splitWorkspaceCards(
  mode: "focus" | "compare" | "board",
  highlightedCards: AgentWorkspaceCardModel[],
  artifactCards: AgentArtifactCardModel[],
  traceCard: AgentTraceCardModel | null,
  pendingCard: AgentPendingCardModel | null,
  contextCard: AgentContextCardModel,
): AgentWorkspaceState {
  const allCards = dedupeCards([
    ...artifactCards,
    contextCard,
    ...(traceCard ? [traceCard] : []),
    ...(pendingCard ? [pendingCard] : []),
  ]);
  const highlightedIds = new Set(highlightedCards.map((card) => card.id));
  return {
    mode,
    highlightedCards,
    boardCards: allCards.filter((card) => !highlightedIds.has(card.id)),
  };
}

function isEvalArtifactCard(card: AgentArtifactCardModel): card is AgentArtifactCardModel & {
  artifact: AgentEvalJobPreview;
} {
  return card.artifact.kind === "eval";
}

function isInterpArtifactCard(card: AgentArtifactCardModel): boolean {
  return card.artifact.kind === "interp";
}

function isTrainingArtifactCard(card: AgentArtifactCardModel): boolean {
  return card.artifact.kind === "training";
}

function dedupeCards(cards: AgentWorkspaceCardModel[]): AgentWorkspaceCardModel[] {
  const seen = new Set<string>();
  return cards.filter((card) => {
    if (seen.has(card.id)) {
      return false;
    }
    seen.add(card.id);
    return true;
  });
}

function lastItem<T>(items: T[]): T | undefined {
  return items.length > 0 ? items[items.length - 1] : undefined;
}

function findLastCard<T extends AgentArtifactCardModel>(
  cards: AgentArtifactCardModel[],
  predicate: (card: AgentArtifactCardModel) => card is T,
): T | undefined;
function findLastCard(
  cards: AgentArtifactCardModel[],
  predicate: (card: AgentArtifactCardModel) => boolean,
): AgentArtifactCardModel | undefined;
function findLastCard(
  cards: AgentArtifactCardModel[],
  predicate: (card: AgentArtifactCardModel) => boolean,
): AgentArtifactCardModel | undefined {
  for (let index = cards.length - 1; index >= 0; index -= 1) {
    const card = cards[index];
    if (predicate(card)) {
      return card;
    }
  }
  return undefined;
}
