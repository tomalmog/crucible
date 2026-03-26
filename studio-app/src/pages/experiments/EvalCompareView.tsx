import { useCallback, useEffect, useMemo, useState } from "react";
import { Loader2 } from "lucide-react";
import { useCrucible } from "../../context/CrucibleContext";
import { useUnifiedJobs } from "../../hooks/useUnifiedJobs";
import { getJobResult } from "../../api/jobsApi";
import type { JobRecord } from "../../types/jobs";

interface EvalBenchmark {
  name: string;
  score: number;
  num_examples: number;
  correct: number;
}

interface EvalResult {
  benchmarks: EvalBenchmark[];
  average_score: number;
}

interface SelectedJob {
  jobId: string;
  label: string;
  date: string;
  avgScore: number;
  benchmarks: EvalBenchmark[];
}

const BAR_COLORS = [
  "#769FCD", "#d97706", "#34d399", "#f87171", "#a78bfa", "#fb923c", "#38bdf8",
];

const MAX_SELECTED = 8;

function pct(b: EvalBenchmark): number {
  return b.num_examples > 0 ? (b.correct / b.num_examples) * 100 : 0;
}

/** Parse benchmark lines from local eval stdout:
 *  benchmark=mmlu  score=0.0  examples=5  correct=0 */
function parseBenchmarksFromStdout(stdout: string): EvalBenchmark[] {
  const results: EvalBenchmark[] = [];
  for (const line of stdout.split("\n")) {
    const m = line.match(/^benchmark=(\S+)\s+score=([\d.]+)\s+examples=(\d+)\s+correct=(\d+)/);
    if (m) {
      results.push({ name: m[1], score: parseFloat(m[2]), num_examples: parseInt(m[3], 10), correct: parseInt(m[4], 10) });
    }
  }
  return results;
}

/** Build EvalResult from either remote API result or local stdout. */
function buildEvalResult(job: JobRecord, apiResult: Record<string, unknown>): EvalResult {
  // Remote jobs have benchmarks in the API result
  let benchmarks = ((apiResult.benchmarks ?? []) as EvalBenchmark[]).filter(
    (b) => b.num_examples > 0,
  );
  // Local jobs: API result has no benchmarks, parse from persisted stdout
  if (benchmarks.length === 0 && job.stdout) {
    benchmarks = parseBenchmarksFromStdout(job.stdout);
  }
  const avgScore = benchmarks.length > 0
    ? benchmarks.reduce((s, b) => s + pct(b), 0) / benchmarks.length
    : 0;
  return { benchmarks, average_score: avgScore };
}

export function EvalCompareView() {
  const { dataRoot } = useCrucible();
  const { jobs, isLoading } = useUnifiedJobs(dataRoot);
  const [checkedIds, setCheckedIds] = useState<Set<string>>(new Set());
  const [results, setResults] = useState<Map<string, EvalResult>>(new Map());
  const [loadingIds, setLoadingIds] = useState<Set<string>>(new Set());

  const evalJobs = useMemo(
    () => jobs
      .filter((j) => j.jobType === "eval" && j.state === "completed")
      .sort((a, b) => b.createdAt.localeCompare(a.createdAt)),
    [jobs],
  );

  // Build a lookup for quick access to full job records
  const jobMap = useMemo(() => {
    const m = new Map<string, JobRecord>();
    for (const j of evalJobs) m.set(j.jobId, j);
    return m;
  }, [evalJobs]);

  const toggleJob = useCallback((jobId: string) => {
    setCheckedIds((prev) => {
      const next = new Set(prev);
      if (next.has(jobId)) {
        next.delete(jobId);
      } else if (next.size < MAX_SELECTED) {
        next.add(jobId);
      }
      return next;
    });
  }, []);

  // Fetch results for checked jobs
  useEffect(() => {
    for (const jobId of checkedIds) {
      if (results.has(jobId) || loadingIds.has(jobId)) continue;
      const job = jobMap.get(jobId);
      if (!job) continue;

      // For local jobs with stdout, parse directly — no API call needed
      if (job.backend === "local" && job.stdout) {
        const benchmarks = parseBenchmarksFromStdout(job.stdout);
        const avgScore = benchmarks.length > 0
          ? benchmarks.reduce((s, b) => s + pct(b), 0) / benchmarks.length : 0;
        setResults((prev) => new Map(prev).set(jobId, { benchmarks, average_score: avgScore }));
        continue;
      }

      setLoadingIds((prev) => new Set(prev).add(jobId));
      getJobResult(dataRoot, jobId, "completed")
        .then((raw) => {
          setResults((prev) => new Map(prev).set(jobId, buildEvalResult(job, raw)));
        })
        .catch(() => {
          // Last resort: try parsing from stdout even if API failed
          const fallback = job.stdout ? parseBenchmarksFromStdout(job.stdout) : [];
          const avgScore = fallback.length > 0
            ? fallback.reduce((s, b) => s + pct(b), 0) / fallback.length : 0;
          setResults((prev) => new Map(prev).set(jobId, { benchmarks: fallback, average_score: avgScore }));
        })
        .finally(() => {
          setLoadingIds((prev) => {
            const next = new Set(prev);
            next.delete(jobId);
            return next;
          });
        });
    }
  }, [checkedIds, dataRoot, results, loadingIds, jobMap]);

  // Build selected jobs with fetched data
  const selected: SelectedJob[] = useMemo(
    () => evalJobs
      .filter((j) => checkedIds.has(j.jobId) && results.has(j.jobId))
      .map((j) => {
        const r = results.get(j.jobId)!;
        return {
          jobId: j.jobId,
          label: j.label || j.jobId.slice(0, 8),
          date: new Date(j.createdAt).toLocaleDateString(),
          avgScore: r.average_score,
          benchmarks: r.benchmarks,
        };
      }),
    [evalJobs, checkedIds, results],
  );

  if (isLoading) {
    return <div className="empty-state"><Loader2 className="spinner" size={20} /> Loading jobs...</div>;
  }

  if (evalJobs.length === 0) {
    return <div className="empty-state">No completed evaluation jobs yet. Run an evaluation from the Evaluate tab first.</div>;
  }

  return (
    <div className="split-grid">
      <JobSelector
        evalJobs={evalJobs}
        checkedIds={checkedIds}
        loadingIds={loadingIds}
        results={results}
        onToggle={toggleJob}
      />
      <div className="stack-lg">
        {selected.length < 2
          ? <div className="empty-state">Select at least 2 completed eval jobs to compare.</div>
          : <>
              <ComparisonTable selected={selected} />
              <ComparisonBarChart selected={selected} />
            </>
        }
      </div>
    </div>
  );
}

// ── Job selector list ───────────────────────────────────────────────────

function JobSelector({ evalJobs, checkedIds, loadingIds, results, onToggle }: {
  evalJobs: { jobId: string; label: string; createdAt: string }[];
  checkedIds: Set<string>;
  loadingIds: Set<string>;
  results: Map<string, EvalResult>;
  onToggle: (id: string) => void;
}) {
  return (
    <div className="panel stack">
      <h3>Eval Jobs</h3>
      <div className="stack-sm" style={{ maxHeight: 480, overflowY: "auto" }}>
        {evalJobs.map((j) => {
          const checked = checkedIds.has(j.jobId);
          const loading = loadingIds.has(j.jobId);
          const r = results.get(j.jobId);
          const disabled = !checked && checkedIds.size >= MAX_SELECTED;
          return (
            <label
              key={j.jobId}
              className="form-field"
              style={{ display: "flex", alignItems: "center", gap: 8, opacity: disabled ? 0.5 : 1, cursor: disabled ? "not-allowed" : "pointer" }}
            >
              <input
                type="checkbox"
                checked={checked}
                disabled={disabled}
                onChange={() => onToggle(j.jobId)}
              />
              <span style={{ flex: 1 }}>
                <strong>{j.label || j.jobId.slice(0, 8)}</strong>
                <br />
                <span className="text-secondary text-xs">{new Date(j.createdAt).toLocaleDateString()}</span>
              </span>
              {loading && <Loader2 className="spinner" size={14} />}
              {!loading && r && r.average_score != null && (
                <span className="badge badge-success">{r.average_score.toFixed(1)}%</span>
              )}
            </label>
          );
        })}
      </div>
    </div>
  );
}

// ── Comparison table ────────────────────────────────────────────────────

function ComparisonTable({ selected }: { selected: SelectedJob[] }) {
  const allBenchNames = useMemo(() => {
    const names = new Set<string>();
    for (const s of selected) {
      for (const b of s.benchmarks) names.add(b.name);
    }
    return [...names].sort();
  }, [selected]);

  // Build lookup: jobId → benchName → score
  const scoreMap = useMemo(() => {
    const m = new Map<string, Map<string, number>>();
    for (const s of selected) {
      const inner = new Map<string, number>();
      for (const b of s.benchmarks) inner.set(b.name, pct(b));
      m.set(s.jobId, inner);
    }
    return m;
  }, [selected]);

  return (
    <div className="panel">
      <div className="docs-table-wrap">
        <table className="docs-table">
          <thead>
            <tr>
              <th>Benchmark</th>
              {selected.map((s) => (
                <th key={s.jobId}>{s.label}<br /><span className="text-secondary text-xs">{s.date}</span></th>
              ))}
            </tr>
          </thead>
          <tbody>
            {allBenchNames.map((name) => {
              const scores = selected.map((s) => scoreMap.get(s.jobId)?.get(name) ?? null);
              const best = Math.max(...scores.filter((s): s is number => s !== null));
              return (
                <tr key={name}>
                  <td style={{ fontWeight: 500 }}>{name.toUpperCase()}</td>
                  {scores.map((score, i) => (
                    <td key={selected[i].jobId} style={score !== null && score === best ? { fontWeight: 700 } : undefined}>
                      {score !== null ? `${score.toFixed(1)}%` : "—"}
                    </td>
                  ))}
                </tr>
              );
            })}
            <tr style={{ borderTop: "2px solid var(--border)" }}>
              <td style={{ fontWeight: 700 }}>AVERAGE</td>
              {selected.map((s) => {
                const best = Math.max(...selected.map((x) => x.avgScore));
                return (
                  <td key={s.jobId} style={s.avgScore === best ? { fontWeight: 700 } : undefined}>
                    {s.avgScore.toFixed(1)}%
                  </td>
                );
              })}
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Grouped bar chart ───────────────────────────────────────────────────

const CHART_W = 1000;
const CHART_H = 400;
const BOUNDS = { top: 58, right: 28, bottom: 100, left: 86 };

function ComparisonBarChart({ selected }: { selected: SelectedJob[] }) {
  const allBenchNames = useMemo(() => {
    const names = new Set<string>();
    for (const s of selected) {
      for (const b of s.benchmarks) names.add(b.name);
    }
    return [...names].sort();
  }, [selected]);

  if (allBenchNames.length === 0) return null;

  const chartW = CHART_W - BOUNDS.left - BOUNDS.right;
  const chartH = CHART_H - BOUNDS.top - BOUNDS.bottom;
  const nGroups = allBenchNames.length;
  const nModels = selected.length;
  const groupWidth = chartW / nGroups;
  const barWidth = Math.min(groupWidth * 0.8 / nModels, 60);
  const totalBarsWidth = barWidth * nModels;

  const yTicks = [0, 25, 50, 75, 100];
  const mapY = (v: number) => BOUNDS.top + chartH - (v / 100) * chartH;
  const mapX = (gi: number) => BOUNDS.left + groupWidth * gi + groupWidth / 2;

  // Build lookup per selected job
  const scoreMaps = selected.map((s) => {
    const m = new Map<string, number>();
    for (const b of s.benchmarks) m.set(b.name, pct(b));
    return m;
  });

  return (
    <div className="training-chart-card">
      <svg viewBox={`0 0 ${CHART_W} ${CHART_H}`} className="training-chart-svg">
        <text className="training-chart-title" x={CHART_W / 2} y={30}>
          Benchmark Comparison
        </text>
        {yTicks.map((v) => (
          <g key={`y-${v}`}>
            <line className="training-grid-line" x1={BOUNDS.left} x2={CHART_W - BOUNDS.right} y1={mapY(v)} y2={mapY(v)} />
            <text className="training-axis-tick training-axis-tick-y" x={BOUNDS.left - 10} y={mapY(v) + 4}>{v}%</text>
          </g>
        ))}
        <line className="training-axis-line" x1={BOUNDS.left} x2={CHART_W - BOUNDS.right} y1={mapY(0)} y2={mapY(0)} />
        <line className="training-axis-line" x1={BOUNDS.left} x2={BOUNDS.left} y1={BOUNDS.top} y2={mapY(0)} />

        {allBenchNames.map((name, gi) => {
          const cx = mapX(gi);
          const startX = cx - totalBarsWidth / 2;
          return (
            <g key={name}>
              {selected.map((s, mi) => {
                const score = scoreMaps[mi].get(name) ?? 0;
                const barH = (score / 100) * chartH;
                const x = startX + mi * barWidth;
                const color = BAR_COLORS[mi % BAR_COLORS.length];
                return (
                  <g key={s.jobId}>
                    <rect x={x} y={mapY(score)} width={barWidth * 0.85} height={barH} fill={color} rx={3} />
                    {score > 0 && (
                      <text className="training-axis-tick" x={x + barWidth * 0.425} y={mapY(score) - 6} textAnchor="middle" style={{ fontSize: 10 }}>
                        {score.toFixed(1)}%
                      </text>
                    )}
                  </g>
                );
              })}
              <text className="training-axis-tick training-axis-tick-x" x={cx} y={mapY(0) + 22}>
                {name.toUpperCase()}
              </text>
            </g>
          );
        })}

        {/* Legend */}
        <g transform={`translate(${BOUNDS.left}, ${BOUNDS.top - 28})`}>
          {selected.map((s, i) => {
            const xOff = i * 120;
            return (
              <g key={s.jobId} transform={`translate(${xOff}, 0)`}>
                <rect x={0} y={3} width={12} height={12} fill={BAR_COLORS[i % BAR_COLORS.length]} rx={2} />
                <text className="training-legend-label" x={18} y={13} style={{ fontSize: 11 }}>
                  {s.label.length > 12 ? s.label.slice(0, 12) + "…" : s.label}
                </text>
              </g>
            );
          })}
        </g>

        <text className="training-axis-label" x={24} y={CHART_H / 2} transform={`rotate(-90 24 ${CHART_H / 2})`}>Accuracy</text>
      </svg>
    </div>
  );
}
