import React, { useEffect, useState } from "react";
import { Trophy } from "lucide-react";
import { getJobResult } from "../../api/jobsApi";
import type { JobRecord } from "../../types/jobs";

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

export function DashboardLeaderboard({ dataRoot, completedJobs }: DashboardLeaderboardProps): React.ReactNode {
  const [scores, setScores] = useState<EvalScore[]>([]);

  // Fetch eval results from completed eval jobs
  useEffect(() => {
    const evalJobs = completedJobs.filter((j) => j.jobType === "eval");
    if (evalJobs.length === 0) {
      setScores([]);
      return;
    }
    let cancelled = false;
    async function load(): Promise<void> {
      const results: EvalScore[] = [];
      for (const job of evalJobs) {
        try {
          const result = await getJobResult(dataRoot, job.jobId);
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
        setScores(results.sort((a, b) => b.averageScore - a.averageScore));
      }
    }
    load();
    return () => { cancelled = true; };
  }, [dataRoot, completedJobs]);

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
