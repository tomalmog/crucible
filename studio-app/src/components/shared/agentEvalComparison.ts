import type { AgentEvalJobPreview } from "../../types/agent";

export interface AgentEvalComparisonBenchmark {
  name: string;
  score: number;
}

export interface AgentEvalComparisonRun {
  title: string;
  meta?: string | null;
  averageScore: number;
  benchmarkCount: number;
  benchmarks: ReadonlyArray<AgentEvalComparisonBenchmark>;
}

export type AgentEvalComparisonInput = AgentEvalComparisonRun | AgentEvalJobPreview;
export type ComparisonWinner = "left" | "right" | "tie";

export interface ComparisonRow {
  name: string;
  leftScore: number | null;
  rightScore: number | null;
  delta: number | null;
  winner: ComparisonWinner;
}

export interface VisibleBenchmarkRows {
  rows: ReadonlyArray<ComparisonRow>;
  hiddenCount: number;
  totalCount: number;
}

const SCORE_TIE_EPSILON = 0.05;

export function normalizeComparisonRun(input: AgentEvalComparisonInput): AgentEvalComparisonRun {
  if (isAgentEvalJobPreview(input)) {
    return {
      title: input.title,
      meta: input.cluster,
      averageScore: input.averageScore,
      benchmarkCount: input.benchmarkCount,
      benchmarks: input.topBenchmarks,
    };
  }

  return {
    title: input.title,
    meta: input.meta ?? null,
    averageScore: input.averageScore,
    benchmarkCount: input.benchmarkCount,
    benchmarks: input.benchmarks,
  };
}

export function buildVisibleBenchmarkRows({
  left,
  right,
  maxBenchmarks,
}: {
  left: AgentEvalComparisonRun;
  right: AgentEvalComparisonRun;
  maxBenchmarks: number;
}): VisibleBenchmarkRows {
  const leftScores = createBenchmarkScoreMap(left.benchmarks);
  const rightScores = createBenchmarkScoreMap(right.benchmarks);
  const names = new Set<string>([...leftScores.keys(), ...rightScores.keys()]);
  const allRows = [...names]
    .map((name) => buildComparisonRow(name, leftScores.get(name) ?? null, rightScores.get(name) ?? null))
    .sort(compareBenchmarkRows);
  const visibleCount = Math.max(1, maxBenchmarks);

  return {
    rows: allRows.slice(0, visibleCount),
    hiddenCount: Math.max(0, allRows.length - visibleCount),
    totalCount: allRows.length,
  };
}

export function getComparisonWinner(
  leftScore: number | null,
  rightScore: number | null,
): ComparisonWinner {
  if (leftScore == null && rightScore == null) {
    return "tie";
  }
  if (leftScore == null) {
    return "right";
  }
  if (rightScore == null) {
    return "left";
  }
  if (Math.abs(leftScore - rightScore) < SCORE_TIE_EPSILON) {
    return "tie";
  }
  return leftScore > rightScore ? "left" : "right";
}

export function formatScore(score: number): string {
  return `${score.toFixed(1)}%`;
}

export function formatNullableScore(score: number | null): string {
  return score == null ? "—" : formatScore(score);
}

export function formatOutcomeHeadline(winner: ComparisonWinner, delta: number): string {
  if (winner === "tie") {
    return "Even overall";
  }
  return `${winner === "left" ? "Left" : "Right"} +${delta.toFixed(1)} pts`;
}

export function formatOutcomeBody(winner: ComparisonWinner): string {
  if (winner === "tie") {
    return "Both runs land on the same average score across the provided benchmarks.";
  }
  return "Average score across the benchmarks provided to this card.";
}

export function formatDeltaLabel(row: ComparisonRow): string {
  if (row.leftScore == null && row.rightScore == null) {
    return "No data";
  }
  if (row.leftScore == null) {
    return "Right only";
  }
  if (row.rightScore == null) {
    return "Left only";
  }
  if (row.winner === "tie" || row.delta == null) {
    return "Tie";
  }
  return `${row.winner === "left" ? "Left" : "Right"} +${Math.abs(row.delta).toFixed(1)}`;
}

function isAgentEvalJobPreview(input: AgentEvalComparisonInput): input is AgentEvalJobPreview {
  return "kind" in input;
}

function createBenchmarkScoreMap(
  benchmarks: ReadonlyArray<AgentEvalComparisonBenchmark>,
): Map<string, number> {
  return new Map(benchmarks.map((benchmark) => [benchmark.name, benchmark.score]));
}

function buildComparisonRow(
  name: string,
  leftScore: number | null,
  rightScore: number | null,
): ComparisonRow {
  return {
    name,
    leftScore,
    rightScore,
    delta: leftScore != null && rightScore != null ? rightScore - leftScore : null,
    winner: getComparisonWinner(leftScore, rightScore),
  };
}

function compareBenchmarkRows(left: ComparisonRow, right: ComparisonRow): number {
  const magnitudeDifference = getRowMagnitude(right) - getRowMagnitude(left);
  if (magnitudeDifference !== 0) {
    return magnitudeDifference;
  }
  return left.name.localeCompare(right.name);
}

function getRowMagnitude(row: ComparisonRow): number {
  if (row.delta != null) {
    return Math.abs(row.delta);
  }
  if (row.leftScore != null || row.rightScore != null) {
    return -0.5;
  }
  return -1;
}
