import { useMemo } from "react";
import type { PendingChain } from "../../hooks/useAgentChat";
import type { AgentMessage, AgentTraceEvent } from "../../types/agent";
import { buildEvalComparisonSummary } from "./buildWorkspaceComparison";
import {
  BuildWorkspaceCards,
  BuildWorkspaceComparisonCard,
  buildHighlightedEvalComparison,
} from "./BuildWorkspaceCards";
import { buildWorkspaceState } from "./buildWorkspaceState";

interface BuildWorkspacePanelProps {
  currentTrace: AgentTraceEvent[];
  isLoading: boolean;
  messages: AgentMessage[];
  pendingChain: PendingChain | null;
  selectedDatasetName: string | null;
  selectedModelName: string | null;
}

export function BuildWorkspacePanel({
  currentTrace,
  isLoading,
  messages,
  pendingChain,
  selectedDatasetName,
  selectedModelName,
}: BuildWorkspacePanelProps): React.ReactNode {
  const workspace = useMemo(
    () => buildWorkspaceState(messages, currentTrace, pendingChain),
    [messages, currentTrace, pendingChain],
  );
  const comparison = buildHighlightedEvalComparison(
    workspace.highlightedCards,
    buildEvalComparisonSummary,
  );
  const runState = describeRunState(isLoading, pendingChain, workspace.highlightedCards.length > 0);

  return (
    <aside className="build-workspace">
      <header className="build-workspace-header">
        <div>
          <span className="build-workspace-kicker">Execution workspace</span>
          <h2>Live model iteration</h2>
          <p>
            Training curves, eval deltas, and interpretability artifacts land here while
            the agent works.
          </p>
        </div>
        <div className={`build-workspace-status${isLoading ? " build-workspace-status-live" : ""}`}>
          {runState}
        </div>
      </header>

      {comparison && <BuildWorkspaceComparisonCard comparison={comparison} />}

      <div className={highlightGridClassName(workspace.highlightedCards.length)}>
        <BuildWorkspaceCards
          cards={workspace.highlightedCards}
          compact={workspace.mode === "board"}
          selectedDatasetName={selectedDatasetName}
          selectedModelName={selectedModelName}
        />
      </div>

      {workspace.boardCards.length > 0 && (
        <section className="build-workspace-board">
          <div className="build-workspace-board-header">
            <span className="build-workspace-label">Card board</span>
            <strong>Keep every useful output within reach</strong>
          </div>
          <div className="build-workspace-board-grid">
            <BuildWorkspaceCards
              cards={workspace.boardCards}
              compact
              selectedDatasetName={selectedDatasetName}
              selectedModelName={selectedModelName}
            />
          </div>
        </section>
      )}
    </aside>
  );
}

function describeRunState(
  isLoading: boolean,
  pendingChain: PendingChain | null,
  hasCards: boolean,
): string {
  if (pendingChain) {
    return pendingChain.jobComplete ? "Ready for follow-up" : "Running remotely";
  }
  if (isLoading) {
    return "Planning";
  }
  return hasCards ? "Artifacts ready" : "Waiting for instructions";
}

function highlightGridClassName(cardCount: number): string {
  if (cardCount >= 2) {
    return "build-workspace-highlight-grid build-workspace-highlight-grid-compare";
  }
  return "build-workspace-highlight-grid";
}
