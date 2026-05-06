import { AgentEventTimeline } from "../../components/shared/AgentEventTimeline";
import { AgentJobPreviewCard } from "../../components/shared/AgentJobPreviewCard";
import type { PendingChain } from "../../hooks/useAgentChat";
import type { AgentEvalJobPreview } from "../../types/agent";
import type { EvalBenchmarkDelta, EvalComparisonSummary } from "./buildWorkspaceComparison";
import {
  type AgentArtifactCardModel,
  type AgentPendingCardModel,
  type AgentTraceCardModel,
  type AgentWorkspaceCardModel,
} from "./buildWorkspaceState";

export interface BuildWorkspaceCardsProps {
  cards: AgentWorkspaceCardModel[];
  compact: boolean;
  selectedDatasetName: string | null;
  selectedModelName: string | null;
}

export function BuildWorkspaceCards({
  cards,
  compact,
  selectedDatasetName,
  selectedModelName,
}: BuildWorkspaceCardsProps): React.ReactNode {
  return cards.map((card) => (
    <BuildWorkspaceCard
      key={card.id}
      card={card}
      compact={compact}
      selectedDatasetName={selectedDatasetName}
      selectedModelName={selectedModelName}
    />
  ));
}

function BuildWorkspaceCard({
  card,
  compact,
  selectedDatasetName,
  selectedModelName,
}: {
  card: AgentWorkspaceCardModel;
  compact: boolean;
  selectedDatasetName: string | null;
  selectedModelName: string | null;
}): React.ReactNode {
  const className = compact
    ? "build-workspace-card build-workspace-card-compact"
    : "build-workspace-card";
  if (card.type === "artifact") {
    return <BuildArtifactCard artifactCard={card} className={className} />;
  }
  if (card.type === "trace") {
    return <BuildTraceCard traceCard={card} className={className} />;
  }
  if (card.type === "pending_chain") {
    return <BuildPendingCard pendingCard={card} className={className} />;
  }
  return (
    <BuildContextCard
      className={className}
      selectedDatasetName={selectedDatasetName}
      selectedModelName={selectedModelName}
    />
  );
}

function BuildArtifactCard({
  artifactCard,
  className,
}: {
  artifactCard: AgentArtifactCardModel;
  className: string;
}): React.ReactNode {
  const { artifact, messageContent } = artifactCard;
  return (
    <section className={className}>
      <div className="build-workspace-card-header">
        <div>
          <span className="build-workspace-label">{artifactLabel(artifact.kind)}</span>
          <strong>{artifact.title}</strong>
        </div>
        <span>{artifact.cluster ?? "local"}</span>
      </div>
      {messageContent && <p className="build-workspace-summary">{messageContent}</p>}
      <AgentJobPreviewCard artifact={artifact} displayMode="workspace" />
    </section>
  );
}

function BuildTraceCard({
  traceCard,
  className,
}: {
  traceCard: AgentTraceCardModel;
  className: string;
}): React.ReactNode {
  return (
    <section className={`${className} build-workspace-card-trace`}>
      <div className="build-workspace-card-header">
        <div>
          <span className="build-workspace-label">Reasoning trace</span>
          <strong>Tool use and decisions</strong>
        </div>
      </div>
      <AgentEventTimeline events={traceCard.events} />
    </section>
  );
}

function BuildPendingCard({
  pendingCard,
  className,
}: {
  pendingCard: AgentPendingCardModel;
  className: string;
}): React.ReactNode {
  return (
    <section className={className}>
      <div className="build-workspace-card-header">
        <div>
          <span className="build-workspace-label">Queued follow-up</span>
          <strong>{pendingCard.pendingChain.jobId}</strong>
        </div>
        <span>{pendingStatusLabel(pendingCard.pendingChain)}</span>
      </div>
      <div className="build-workspace-steps">
        {pendingCard.pendingChain.steps.map((step, index) => (
          <div key={index} className="build-workspace-step">
            <span>{index + 1}</span>
            <p>{step}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function BuildContextCard({
  className,
  selectedDatasetName,
  selectedModelName,
}: {
  className: string;
  selectedDatasetName: string | null;
  selectedModelName: string | null;
}): React.ReactNode {
  return (
    <section className={className}>
      <div className="build-workspace-card-header">
        <div>
          <span className="build-workspace-label">Run context</span>
          <strong>What the agent is optimizing</strong>
        </div>
      </div>
      <div className="build-workspace-stats">
        <div className="build-workspace-stat">
          <span>Selected model</span>
          <strong>{selectedModelName ?? "No model selected"}</strong>
        </div>
        <div className="build-workspace-stat">
          <span>Selected dataset</span>
          <strong>{selectedDatasetName ?? "No dataset selected"}</strong>
        </div>
        <div className="build-workspace-stat">
          <span>Workspace behavior</span>
          <strong>Agent-controlled cards and comparisons</strong>
        </div>
      </div>
    </section>
  );
}

export function BuildWorkspaceComparisonCard({
  comparison,
}: {
  comparison: EvalComparisonSummary;
}): React.ReactNode {
  return (
    <section className="build-workspace-comparison">
      <div className="build-workspace-comparison-header">
        <div>
          <span className="build-workspace-label">Eval delta</span>
          <strong>
            {comparison.previous.title} to {comparison.current.title}
          </strong>
        </div>
        <div className="build-workspace-comparison-delta">
          {formatSignedDelta(comparison.averageDelta)} pts
        </div>
      </div>
      <div className="build-workspace-comparison-grid">
        <div className="build-workspace-comparison-list">
          <span className="build-workspace-label">Improved</span>
          {comparison.improved.map((delta) => (
            <BuildComparisonRow key={delta.name} delta={delta} />
          ))}
        </div>
        {comparison.regressed.length > 0 && (
          <div className="build-workspace-comparison-list">
            <span className="build-workspace-label">Watchouts</span>
            {comparison.regressed.map((delta) => (
              <BuildComparisonRow key={delta.name} delta={delta} />
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

export function buildHighlightedEvalComparison(
  cards: AgentWorkspaceCardModel[],
  buildSummary: (
    current: AgentEvalJobPreview,
    previous: AgentEvalJobPreview,
  ) => EvalComparisonSummary | null,
): EvalComparisonSummary | null {
  if (cards.length < 2) {
    return null;
  }
  const evalCards = cards.flatMap((card) => isEvalArtifactCard(card) ? [card.artifact] : []);
  if (evalCards.length < 2) {
    return null;
  }
  return buildSummary(evalCards[evalCards.length - 1], evalCards[evalCards.length - 2]);
}

function BuildComparisonRow({
  delta,
}: {
  delta: EvalBenchmarkDelta;
}): React.ReactNode {
  return (
    <div className="build-workspace-comparison-row">
      <span>{delta.name.toUpperCase()}</span>
      <strong>{formatSignedDelta(delta.delta)} pts</strong>
    </div>
  );
}

function pendingStatusLabel(pendingChain: PendingChain): string {
  return pendingChain.jobComplete ? "Ready to resume" : "Waiting on job";
}

function artifactLabel(kind: AgentArtifactCardModel["artifact"]["kind"]): string {
  if (kind === "training") {
    return "Training artifact";
  }
  if (kind === "eval") {
    return "Eval artifact";
  }
  return "Interpretability artifact";
}

function formatSignedDelta(value: number): string {
  const prefix = value >= 0 ? "+" : "";
  return `${prefix}${value.toFixed(1)}`;
}

function isEvalArtifactCard(
  card: AgentWorkspaceCardModel,
): card is AgentArtifactCardModel & { artifact: AgentEvalJobPreview } {
  return card.type === "artifact" && card.artifact.kind === "eval";
}
