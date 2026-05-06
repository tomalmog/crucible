import type { ReactNode } from "react";
import { AgentWorkspaceCard, type AgentWorkspaceCardVariant } from "./AgentWorkspaceCard";
import {
  type AgentEvalComparisonInput,
  type AgentEvalComparisonRun,
  type ComparisonRow,
  type ComparisonWinner,
  buildVisibleBenchmarkRows,
  formatDeltaLabel,
  formatNullableScore,
  formatOutcomeBody,
  formatOutcomeHeadline,
  formatScore,
  getComparisonWinner,
  normalizeComparisonRun,
} from "./agentEvalComparison";

export type {
  AgentEvalComparisonBenchmark,
  AgentEvalComparisonInput,
  AgentEvalComparisonRun,
} from "./agentEvalComparison";

export interface AgentEvalComparisonCardProps {
  left: AgentEvalComparisonInput;
  right: AgentEvalComparisonInput;
  label?: string;
  title?: string;
  headerAside?: ReactNode;
  summary?: string;
  emptyState?: string;
  maxBenchmarks?: number;
  variant?: AgentWorkspaceCardVariant;
  className?: string;
}

const DEFAULT_EMPTY_STATE = "Pass benchmark scores from two evaluation runs to compare them here.";
const DEFAULT_LABEL = "Eval comparison";
const DEFAULT_TITLE = "Side-by-side benchmark shift";
const DEFAULT_MAX_BENCHMARKS = 6;

export function AgentEvalComparisonCard({
  left,
  right,
  label = DEFAULT_LABEL,
  title = DEFAULT_TITLE,
  headerAside,
  summary,
  emptyState = DEFAULT_EMPTY_STATE,
  maxBenchmarks = DEFAULT_MAX_BENCHMARKS,
  variant = "primary",
  className,
}: AgentEvalComparisonCardProps): ReactNode {
  const leftRun = normalizeComparisonRun(left);
  const rightRun = normalizeComparisonRun(right);
  const overallWinner = getComparisonWinner(leftRun.averageScore, rightRun.averageScore);
  const benchmarkRows = buildVisibleBenchmarkRows({
    left: leftRun,
    right: rightRun,
    maxBenchmarks,
  });

  return (
    <AgentWorkspaceCard
      label={label}
      title={title}
      headerAside={headerAside}
      summary={summary}
      emptyState={emptyState}
      variant={variant}
      className={buildCardClassName(className)}
    >
      <div className="agent-eval-comparison">
        <div className="agent-eval-comparison-overview">
          <EvalSummaryPanel run={leftRun} side="left" isLeader={overallWinner === "left"} />
          <EvalOutcomePanel left={leftRun} right={rightRun} winner={overallWinner} />
          <EvalSummaryPanel run={rightRun} side="right" isLeader={overallWinner === "right"} />
        </div>

        {benchmarkRows.rows.length > 0 ? (
          <div className="agent-eval-comparison-benchmarks">
            {benchmarkRows.rows.map((row) => (
              <BenchmarkComparisonRow key={row.name} row={row} />
            ))}
          </div>
        ) : (
          <p className="build-workspace-empty">{emptyState}</p>
        )}

        {benchmarkRows.hiddenCount > 0 && (
          <div className="agent-job-preview-footnote">
            Showing the {benchmarkRows.rows.length} largest benchmark swings out of {benchmarkRows.totalCount}.
          </div>
        )}
      </div>
    </AgentWorkspaceCard>
  );
}

function EvalSummaryPanel({
  run,
  side,
  isLeader,
}: {
  run: AgentEvalComparisonRun;
  side: "left" | "right";
  isLeader: boolean;
}): ReactNode {
  const panelClassName = buildPanelClassName(side, isLeader);
  const sideLabelClassName = buildMetaLabelClassName(side);

  return (
    <article className={panelClassName}>
      <div className="agent-job-preview-header">
        <div className="agent-eval-comparison-heading">
          <strong>{run.title}</strong>
          {run.meta && <span>{run.meta}</span>}
        </div>
        <span className={sideLabelClassName}>{side === "left" ? "Left run" : "Right run"}</span>
      </div>
      <div className="agent-job-preview-metrics">
        <div className="agent-job-preview-metric">
          <span>Average</span>
          <strong>{formatScore(run.averageScore)}</strong>
        </div>
        <div className="agent-job-preview-metric">
          <span>Benchmarks</span>
          <strong>{String(run.benchmarkCount)}</strong>
        </div>
      </div>
    </article>
  );
}

function EvalOutcomePanel({
  left,
  right,
  winner,
}: {
  left: AgentEvalComparisonRun;
  right: AgentEvalComparisonRun;
  winner: ComparisonWinner;
}): ReactNode {
  const delta = Math.abs(right.averageScore - left.averageScore);

  return (
    <div className={`agent-eval-comparison-outcome agent-eval-comparison-outcome--${winner}`}>
      <span className={buildMetaLabelClassName(winner)}>
        {winner === "tie" ? "Overall tie" : "Average lead"}
      </span>
      <strong>{formatOutcomeHeadline(winner, delta)}</strong>
      <p>{formatOutcomeBody(winner)}</p>
    </div>
  );
}

function BenchmarkComparisonRow({ row }: { row: ComparisonRow }): ReactNode {
  return (
    <div className={buildBenchmarkRowClassName(row)}>
      <span className="agent-eval-comparison-benchmark-name">{row.name}</span>
      <span className={buildScoreClassName("left", row)}>{formatNullableScore(row.leftScore)}</span>
      <span className={buildDeltaClassName(row.winner)}>
        {formatDeltaLabel(row)}
      </span>
      <span className={buildScoreClassName("right", row)}>{formatNullableScore(row.rightScore)}</span>
    </div>
  );
}

function buildCardClassName(className?: string): string {
  return className ? `agent-eval-comparison-card ${className}` : "agent-eval-comparison-card";
}

function buildPanelClassName(side: "left" | "right", isLeader: boolean): string {
  const leaderClassName = isLeader
    ? side === "left"
      ? "agent-eval-comparison-panel--leading-left"
      : "agent-eval-comparison-panel--leading-right"
    : "";
  return `agent-job-preview agent-eval-comparison-panel agent-eval-comparison-panel--${side} ${leaderClassName}`.trim();
}

function buildBenchmarkRowClassName(row: ComparisonRow): string {
  return row.winner === "tie"
    ? "agent-eval-comparison-benchmark"
    : `agent-eval-comparison-benchmark agent-eval-comparison-benchmark--${row.winner}`;
}

function buildScoreClassName(side: "left" | "right", row: ComparisonRow): string {
  if ((side === "left" && row.leftScore == null) || (side === "right" && row.rightScore == null)) {
    return "agent-eval-comparison-score agent-eval-comparison-score--missing";
  }
  if (row.winner === side) {
    return "agent-eval-comparison-score agent-eval-comparison-score--winner";
  }
  return "agent-eval-comparison-score";
}

function buildMetaLabelClassName(winner: ComparisonWinner): string {
  return `agent-eval-comparison-meta agent-eval-comparison-meta--${winner}`;
}

function buildDeltaClassName(winner: ComparisonWinner): string {
  return `agent-eval-comparison-delta agent-eval-comparison-meta agent-eval-comparison-meta--${winner}`;
}
