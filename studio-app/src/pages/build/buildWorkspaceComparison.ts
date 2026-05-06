import type { AgentEvalBenchmarkScore, AgentEvalJobPreview } from "../../types/agent";

export interface EvalBenchmarkDelta {
  name: string;
  previousScore: number;
  currentScore: number;
  delta: number;
}

export interface EvalComparisonSummary {
  current: AgentEvalJobPreview;
  previous: AgentEvalJobPreview;
  averageDelta: number;
  improved: EvalBenchmarkDelta[];
  regressed: EvalBenchmarkDelta[];
}

export function buildEvalComparisonSummary(
  current: AgentEvalJobPreview,
  previous: AgentEvalJobPreview,
): EvalComparisonSummary | null {
  const deltas = buildBenchmarkDeltas(current.benchmarks, previous.benchmarks);
  if (deltas.length === 0) {
    return null;
  }
  const averageDelta = current.averageScore - previous.averageScore;
  return {
    current,
    previous,
    averageDelta,
    improved: deltas.filter((delta) => delta.delta >= 0).slice(0, 4),
    regressed: deltas.filter((delta) => delta.delta < 0).slice(0, 2),
  };
}

function buildBenchmarkDeltas(
  currentBenchmarks: AgentEvalBenchmarkScore[],
  previousBenchmarks: AgentEvalBenchmarkScore[],
): EvalBenchmarkDelta[] {
  const previousByName = new Map(
    previousBenchmarks.map((benchmark) => [benchmark.name, benchmark]),
  );
  return currentBenchmarks
    .flatMap((benchmark) => {
      const previous = previousByName.get(benchmark.name);
      if (!previous) {
        return [];
      }
      return [{
        name: benchmark.name,
        previousScore: previous.score,
        currentScore: benchmark.score,
        delta: benchmark.score - previous.score,
      }];
    })
    .sort((left, right) => right.delta - left.delta);
}
