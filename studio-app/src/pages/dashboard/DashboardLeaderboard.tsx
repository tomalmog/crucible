import React, { useEffect, useMemo, useState } from "react";
import { Trophy } from "lucide-react";
import { getJobResult } from "../../api/jobsApi";
import type { JobRecord } from "../../types/jobs";

interface EvalBenchmark {
  name: string;
  score: number;
  num_examples: number;
  correct: number;
  error?: string;
}

interface EvalScore {
  label: string;
  averageScore: number;
  benchmarkCount: number;
  jobId: string;
}

interface DashboardLeaderboardProps {
  dataRoot: string;
  completedJobs: JobRecord[];
}

/** Parse benchmark lines from local eval stdout. */
function parseEvalBenchmarks(stdout: string): EvalBenchmark[] {
  const results: EvalBenchmark[] = [];
  for (const line of stdout.split("\n")) {
    const m = line.match(/^benchmark=(\S+)\s+score=([\d.]+)\s+examples=(\d+)\s+correct=(\d+)(?:\s+error=(.+))?/);
    if (m) {
      const entry: EvalBenchmark = { name: m[1], score: parseFloat(m[2]), num_examples: parseInt(m[3], 10), correct: parseInt(m[4], 10) };
      if (m[5]) entry.error = m[5];
      results.push(entry);
    }
  }
  return results;
}

function pct(b: EvalBenchmark): number {
  return b.num_examples > 0 ? (b.correct / b.num_examples) * 100 : 0;
}

/** Build an EvalScore from a local eval job's stdout. */
function scoreFromStdout(job: JobRecord): EvalScore | null {
  const benchmarks = parseEvalBenchmarks(job.stdout).filter((b) => !b.error && b.num_examples > 0);
  if (benchmarks.length === 0) return null;
  const avg = benchmarks.reduce((s, b) => s + pct(b), 0) / benchmarks.length;
  return {
    label: job.label || job.modelName || job.jobId.slice(0, 12),
    averageScore: avg,
    benchmarkCount: benchmarks.length,
    jobId: job.jobId,
  };
}

export function DashboardLeaderboard({ dataRoot, completedJobs }: DashboardLeaderboardProps): React.ReactNode {
  const evalJobs = useMemo(
    () => completedJobs.filter((j) => j.jobType === "eval"),
    [completedJobs],
  );

  // Immediately resolve local eval jobs from stdout (no async needed)
  const localScores = useMemo(() => {
    const out: EvalScore[] = [];
    for (const job of evalJobs) {
      if (job.backend === "local" && job.stdout) {
        const score = scoreFromStdout(job);
        if (score) out.push(score);
      }
    }
    return out;
  }, [evalJobs]);

  // Fetch remote eval results via API
  const [remoteScores, setRemoteScores] = useState<EvalScore[]>([]);
  useEffect(() => {
    const remoteEvalJobs = evalJobs.filter((j) => j.backend !== "local");
    if (remoteEvalJobs.length === 0) {
      setRemoteScores([]);
      return;
    }
    let cancelled = false;
    async function load(): Promise<void> {
      const results: EvalScore[] = [];
      for (const job of remoteEvalJobs) {
        try {
          const result = await getJobResult(dataRoot, job.jobId, job.state);
          if (result?.average_score != null) {
            results.push({
              label: job.label || job.modelName || job.jobId.slice(0, 12),
              averageScore: result.average_score as number,
              benchmarkCount: (result.benchmarks as unknown[])?.length ?? 0,
              jobId: job.jobId,
            });
          }
        } catch {
          // Skip jobs with no result
        }
      }
      if (!cancelled) {
        setRemoteScores(results);
      }
    }
    load();
    return () => { cancelled = true; };
  }, [dataRoot, evalJobs]);

  const scores = useMemo(
    () => [...localScores, ...remoteScores].sort((a, b) => b.averageScore - a.averageScore),
    [localScores, remoteScores],
  );

  return (
    <div className="resource-card">
      <div className="resource-card-header">
        <h3 className="resource-card-title">Model Leaderboard</h3>
        <Trophy size={14} style={{ color: "var(--warning)" }} />
      </div>
      {scores.length === 0 ? (
        <p className="text-tertiary" style={{ fontSize: "0.8125rem" }}>
          No eval results yet. Run an eval to see scores here.
        </p>
      ) : (
        <table className="docs-table" style={{ fontSize: "0.8125rem" }}>
          <thead>
            <tr>
              <th>#</th>
              <th>Model</th>
              <th>Avg Score</th>
              <th>Benchmarks</th>
            </tr>
          </thead>
          <tbody>
            {scores.slice(0, 5).map((s, i) => (
              <tr key={s.jobId}>
                <td style={{ fontWeight: 600, color: i === 0 ? "var(--warning)" : "var(--text-tertiary)" }}>
                  {i + 1}
                </td>
                <td style={{ fontWeight: 500 }}>{s.label}</td>
                <td style={{ fontFamily: "var(--font-mono)" }}>{s.averageScore.toFixed(1)}%</td>
                <td>{s.benchmarkCount}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
