import { useEffect, useRef } from "react";
import { CheckCircle2, Loader2 } from "lucide-react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AgentEventTimeline } from "../../components/shared/AgentEventTimeline";
import { AgentJobPreviewCard } from "../../components/shared/AgentJobPreviewCard";
import type { PendingChain } from "../../hooks/useAgentChat";
import type { AgentEvalJobPreview, AgentMessage, AgentTraceEvent } from "../../types/agent";
import { BuildComposer } from "./BuildComposer";

interface BuildConversationPaneProps {
  currentTrace: AgentTraceEvent[];
  draft: string;
  error: string | null;
  isLoading: boolean;
  messages: AgentMessage[];
  pendingChain: PendingChain | null;
  replyTextareaRef: React.RefObject<HTMLTextAreaElement | null>;
  onCancelChain: () => void;
  onContinueChain: () => void;
  onDraftChange: (value: string) => void;
  onKeyDown: (event: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  onSubmit: (event: React.FormEvent) => void;
}

export function BuildConversationPane({
  currentTrace,
  draft,
  error,
  isLoading,
  messages,
  pendingChain,
  replyTextareaRef,
  onCancelChain,
  onContinueChain,
  onDraftChange,
  onKeyDown,
  onSubmit,
}: BuildConversationPaneProps): React.ReactNode {
  const threadRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (threadRef.current) {
      threadRef.current.scrollTop = threadRef.current.scrollHeight;
    }
  }, [currentTrace, isLoading, messages, pendingChain]);

  return (
    <section className="build-conversation-pane">
      <div className="build-thread" ref={threadRef}>
        {messages.map((message, index) => (
          <article key={index} className={`build-message ${message.role}`}>
            <header>{message.role === "user" ? "You" : "Crucible"}</header>
            {message.role === "user" ? (
              <p>{message.content}</p>
            ) : (
              <div className="agent-markdown">
                <Markdown remarkPlugins={[remarkGfm]}>{message.content}</Markdown>
              </div>
            )}
            {message.artifact && (
              <BuildMessageArtifact index={index} message={message} messages={messages} />
            )}
            {message.toolsUsed && message.toolsUsed.length > 0 && (
              <div className="agent-tool-badge">Used: {message.toolsUsed.join(", ")}</div>
            )}
            {message.scriptUpdated && (
              <div className="agent-tool-badge">Updated training script</div>
            )}
            {message.navigatedTo && (
              <div className="agent-tool-badge">Navigated to {message.navigatedTo}</div>
            )}
            {message.trace && message.trace.length > 0 && (
              <AgentEventTimeline events={message.trace} />
            )}
          </article>
        ))}
        {isLoading && (
          <>
            <div className="agent-loading">
              <Loader2 size={14} className="spin" /> Thinking...
            </div>
            <div className="agent-live-trace build-live-trace">
              <AgentEventTimeline events={currentTrace} />
            </div>
          </>
        )}
      </div>

      {error && <div className="agent-error">{error}</div>}
      {pendingChain && (
        <BuildPendingChainCard
          isLoading={isLoading}
          pendingChain={pendingChain}
          onCancelChain={onCancelChain}
          onContinueChain={onContinueChain}
        />
      )}

      <BuildComposer
        draft={draft}
        disabled={isLoading}
        placeholder="Reply to Crucible..."
        textareaRef={replyTextareaRef}
        onDraftChange={onDraftChange}
        onKeyDown={onKeyDown}
        onSubmit={onSubmit}
      />
    </section>
  );
}

function BuildMessageArtifact({
  index,
  message,
  messages,
}: {
  index: number;
  message: AgentMessage;
  messages: AgentMessage[];
}): React.ReactNode {
  const artifact = message.artifact;
  if (!artifact) {
    return null;
  }
  const previousEval = isEvalArtifact(artifact) && message.workspaceDirective?.mode === "compare"
    ? findPreviousEval(messages, index)
    : null;
  if (previousEval && isEvalArtifact(artifact)) {
    return (
      <div className="build-message-artifact build-message-artifact-comparison">
        <BuildEvalComparisonTable before={previousEval} after={artifact} />
      </div>
    );
  }
  return (
    <div className="build-message-artifact">
      <AgentJobPreviewCard artifact={artifact} displayMode="workspace" />
    </div>
  );
}

function findPreviousEval(messages: AgentMessage[], index: number): AgentEvalJobPreview | null {
  for (let itemIndex = index - 1; itemIndex >= 0; itemIndex -= 1) {
    const artifact = messages[itemIndex].artifact;
    if (isEvalArtifact(artifact)) {
      return artifact;
    }
  }
  return null;
}

function isEvalArtifact(
  artifact: AgentMessage["artifact"],
): artifact is AgentEvalJobPreview {
  return artifact?.kind === "eval";
}

function BuildEvalComparisonTable({
  before,
  after,
}: {
  before: AgentEvalJobPreview;
  after: AgentEvalJobPreview;
}): React.ReactNode {
  const rows = buildEvalComparisonRows(before, after);
  const averageDelta = after.averageScore - before.averageScore;
  return (
    <section className="build-eval-table-card">
      <div className="build-eval-table-header">
        <div>
          <span>medical_safety_triage</span>
          <strong>Before and after evaluation</strong>
        </div>
        <p>
          {formatScore(before.averageScore)} to {formatScore(after.averageScore)}
          {" "}({formatSignedDelta(averageDelta)} pts)
        </p>
      </div>
      <div className="build-eval-table-wrap">
        <table className="build-eval-table">
          <thead>
            <tr>
              <th>Benchmark</th>
              <th>{before.title}</th>
              <th>{after.title}</th>
              <th>Change</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.name}>
                <td>{row.name}</td>
                <td>{formatNullableScore(row.beforeScore)}</td>
                <td>{formatNullableScore(row.afterScore)}</td>
                <td>{row.delta == null ? "—" : `${formatSignedDelta(row.delta)} pts`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

interface EvalComparisonRow {
  name: string;
  beforeScore: number | null;
  afterScore: number | null;
  delta: number | null;
}

function buildEvalComparisonRows(
  before: AgentEvalJobPreview,
  after: AgentEvalJobPreview,
): EvalComparisonRow[] {
  const beforeScores = new Map(before.benchmarks.map((benchmark) => [benchmark.name, benchmark.score]));
  const afterScores = new Map(after.benchmarks.map((benchmark) => [benchmark.name, benchmark.score]));
  const names = new Set<string>([...beforeScores.keys(), ...afterScores.keys()]);
  return [...names]
    .map((name) => {
      const beforeScore = beforeScores.get(name) ?? null;
      const afterScore = afterScores.get(name) ?? null;
      return {
        name,
        beforeScore,
        afterScore,
        delta: beforeScore != null && afterScore != null ? afterScore - beforeScore : null,
      };
    })
    .sort((left, right) => getDeltaMagnitude(right) - getDeltaMagnitude(left));
}

function getDeltaMagnitude(row: EvalComparisonRow): number {
  return row.delta == null ? -1 : Math.abs(row.delta);
}

function formatScore(value: number): string {
  return `${value.toFixed(1)}%`;
}

function formatNullableScore(value: number | null): string {
  return value == null ? "—" : formatScore(value);
}

function formatSignedDelta(value: number): string {
  const prefix = value >= 0 ? "+" : "";
  return `${prefix}${value.toFixed(1)}`;
}

function BuildPendingChainCard({
  isLoading,
  pendingChain,
  onCancelChain,
  onContinueChain,
}: {
  isLoading: boolean;
  pendingChain: PendingChain;
  onCancelChain: () => void;
  onContinueChain: () => void;
}): React.ReactNode {
  if (!pendingChain.jobComplete) {
    return (
      <div className="agent-chain-banner build-chain-banner">
        <div className="agent-chain-header">
          <Loader2 size={14} className="spin" />
          <span>Waiting for job to complete...</span>
        </div>
        <div className="agent-chain-steps">
          {pendingChain.steps.map((step, index) => (
            <div key={index} className="agent-chain-step">
              <span className="agent-chain-step-num">{index + 1}</span>
              {step}
            </div>
          ))}
        </div>
        <div className="agent-chain-actions">
          <button className="btn btn-ghost btn-sm" onClick={onCancelChain}>
            Cancel
          </button>
        </div>
      </div>
    );
  }

  if (pendingChain.jobState !== "completed") {
    return null;
  }

  return (
    <div className="agent-chain-banner build-chain-banner">
      <div className="agent-chain-header agent-chain-ready">
        <CheckCircle2 size={14} />
        <span>Job completed — continuing automatically</span>
      </div>
      <div className="agent-chain-steps">
        <div className="agent-chain-step">
          <span className="agent-chain-step-num">→</span>
          {pendingChain.steps[0]}
        </div>
        {pendingChain.steps.length > 1 && (
          <div className="agent-chain-step text-tertiary">
            + {pendingChain.steps.length - 1} more step{pendingChain.steps.length > 2 ? "s" : ""}
          </div>
        )}
      </div>
      <div className="agent-chain-actions">
        <button className="btn btn-primary btn-sm" onClick={onContinueChain} disabled={isLoading}>
          Continue now
        </button>
        <button className="btn btn-ghost btn-sm" onClick={onCancelChain}>
          Cancel
        </button>
      </div>
    </div>
  );
}
