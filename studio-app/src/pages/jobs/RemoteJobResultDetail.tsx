import { useCallback, useEffect, useState } from "react";
import { ArrowLeft, Loader2 } from "lucide-react";
import { useCrucible } from "../../context/CrucibleContext";
import { getRemoteJobResult, getRemoteJobLogs } from "../../api/remoteApi";
import { TrainingCurvesView } from "../training/TrainingCurvesView";
import { LogitLensResults } from "../interp/LogitLensResults";
import { ActivationPcaResults } from "../interp/ActivationPcaResults";
import { ActivationPatchingResults } from "../interp/ActivationPatchingResults";
import type { TrainingHistory } from "../../types";
import type { RemoteJobRecord } from "../../types/remote";
import type { LogitLensResult, PcaResult, PatchingResult } from "../../types/interp";

interface RemoteJobResultDetailProps {
  job: RemoteJobRecord;
  onBack: () => void;
}

interface EvalBenchmark {
  name: string;
  score: number;
  num_examples: number;
  correct: number;
}

interface TrainingResult {
  status: string;
  model_path?: string;
  history_path?: string;
  epochs_completed?: number;
  run_id?: string;
  result?: string;
  error?: string;
  traceback?: string;
  training_history?: TrainingHistory;
  // eval fields
  job_type?: string;
  average_score?: number;
  benchmarks?: EvalBenchmark[];
  base_benchmarks?: EvalBenchmark[];
}

function BackButton({ onBack }: { onBack: () => void }) {
  return (
    <button className="btn btn-ghost btn-sm" onClick={onBack} style={{ justifySelf: "start" }}>
      <ArrowLeft size={14} /> Back to Jobs
    </button>
  );
}

export function RemoteJobResultDetail({ job, onBack }: RemoteJobResultDetailProps) {
  const { dataRoot } = useCrucible();
  const [result, setResult] = useState<TrainingResult | null>(null);
  const [logs, setLogs] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchResult = useCallback(async () => {
    if (!dataRoot) return;
    setLoading(true);
    setError(null);
    try {
      const data = await getRemoteJobResult(dataRoot, job.jobId, true);
      setResult(data as unknown as TrainingResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [dataRoot, job.jobId]);

  useEffect(() => { fetchResult(); }, [fetchResult]);

  // Also fetch logs for failed jobs
  useEffect(() => {
    if (!dataRoot || job.state !== "failed") return;
    getRemoteJobLogs(dataRoot, job.jobId, job.state, true)
      .then((content) => setLogs(content?.trim() || ""))
      .catch(() => {});
  }, [dataRoot, job.jobId, job.state]);

  if (loading) {
    return (
      <div className="panel stack-lg">
        <BackButton onBack={onBack} />
        <div style={{ display: "flex", justifyContent: "center", padding: 32 }}>
          <Loader2 size={24} className="spin" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="panel stack-lg">
        <BackButton onBack={onBack} />
        <h3>{job.modelName || job.jobId} — Result</h3>
        <div className="error-alert-prominent">{error}</div>
      </div>
    );
  }

  if (!result || Object.keys(result).length === 0) {
    return (
      <div className="panel stack-lg">
        <BackButton onBack={onBack} />
        <h3>{job.modelName || job.jobId} — Result</h3>
        <div className="empty-state">
          <p>No result.json found on remote cluster.</p>
        </div>
      </div>
    );
  }

  if (job.state === "failed" || result.status === "failed") {
    return <FailedResultView job={job} result={result} logs={logs} onBack={onBack} />;
  }

  if (result.job_type === "eval") {
    return <EvalResultView job={job} result={result} onBack={onBack} />;
  }

  if (result.job_type === "logit-lens" || result.job_type === "activation-pca" || result.job_type === "activation-patch") {
    return <InterpResultView job={job} result={result} onBack={onBack} />;
  }

  return <TrainingResultView job={job} result={result} onBack={onBack} />;
}

function FailedResultView({ job, result, logs, onBack }: {
  job: RemoteJobRecord;
  result: TrainingResult;
  logs: string;
  onBack: () => void;
}) {
  return (
    <div className="panel stack-lg">
      <BackButton onBack={onBack} />
      <h3>Job Failed: {job.modelName || job.jobId}</h3>
      {result.error && (
        <div className="error-alert-prominent">{result.error}</div>
      )}
      {result.traceback && (
        <details>
          <summary className="error-text">Full traceback</summary>
          <pre className="console console-short">{result.traceback}</pre>
        </details>
      )}
      {logs && (
        <details>
          <summary>Remote logs</summary>
          <pre className="console console-short">{logs}</pre>
        </details>
      )}
    </div>
  );
}

// ── Eval bar chart constants ──────────────────────────────────────────

const BAR_CHART_WIDTH = 1000;
const BAR_CHART_HEIGHT = 400;
const BAR_BOUNDS = { top: 58, right: 28, bottom: 100, left: 86 };
const BAR_COLORS = [
  "#769FCD", "#d97706", "#34d399", "#f87171", "#a78bfa", "#fb923c", "#38bdf8",
];

/** Compute score as correct/total percentage (0-100). */
function pct(b: EvalBenchmark): number {
  return b.num_examples > 0 ? (b.correct / b.num_examples) * 100 : 0;
}

function BenchmarkBarChart({ benchmarks, baseBenchmarks }: {
  benchmarks: EvalBenchmark[];
  baseBenchmarks: EvalBenchmark[];
}) {
  if (benchmarks.length === 0) return null;
  const hasBase = baseBenchmarks.length > 0;
  const baseMap = new Map(baseBenchmarks.map((b) => [b.name, b]));

  const chartW = BAR_CHART_WIDTH - BAR_BOUNDS.left - BAR_BOUNDS.right;
  const chartH = BAR_CHART_HEIGHT - BAR_BOUNDS.top - BAR_BOUNDS.bottom;
  const n = benchmarks.length;
  const groupWidth = chartW / n;
  const barWidth = hasBase ? groupWidth * 0.35 : groupWidth * 0.6;
  const gap = hasBase ? groupWidth * 0.04 : 0;

  // Y axis: 0% to 100%
  const yTicks = [0, 25, 50, 75, 100];
  const mapY = (v: number) => BAR_BOUNDS.top + chartH - (v / 100) * chartH;
  const mapX = (i: number) => BAR_BOUNDS.left + groupWidth * i + groupWidth / 2;

  return (
    <div className="training-chart-card">
      <svg viewBox={`0 0 ${BAR_CHART_WIDTH} ${BAR_CHART_HEIGHT}`} className="training-chart-svg">
        <text className="training-chart-title" x={BAR_CHART_WIDTH / 2} y={30}>
          Benchmark Scores
        </text>

        {/* Y grid lines + labels */}
        {yTicks.map((v) => (
          <g key={`y-${v}`}>
            <line
              className="training-grid-line"
              x1={BAR_BOUNDS.left} x2={BAR_CHART_WIDTH - BAR_BOUNDS.right}
              y1={mapY(v)} y2={mapY(v)}
            />
            <text className="training-axis-tick training-axis-tick-y" x={BAR_BOUNDS.left - 10} y={mapY(v) + 4}>
              {v}%
            </text>
          </g>
        ))}

        {/* Axis lines */}
        <line
          className="training-axis-line"
          x1={BAR_BOUNDS.left} x2={BAR_CHART_WIDTH - BAR_BOUNDS.right}
          y1={mapY(0)} y2={mapY(0)}
        />
        <line
          className="training-axis-line"
          x1={BAR_BOUNDS.left} x2={BAR_BOUNDS.left}
          y1={BAR_BOUNDS.top} y2={mapY(0)}
        />

        {/* Bars */}
        {benchmarks.map((b, i) => {
          const cx = mapX(i);
          const base = baseMap.get(b.name);
          const color = BAR_COLORS[i % BAR_COLORS.length];
          const score = pct(b);
          const barH = (score / 100) * chartH;

          if (hasBase && base) {
            const baseScore = pct(base);
            const baseH = (baseScore / 100) * chartH;
            const leftX = cx - barWidth - gap / 2;
            const rightX = cx + gap / 2;
            return (
              <g key={b.name}>
                <rect x={leftX} y={mapY(score)} width={barWidth} height={barH} fill={color} rx={3} />
                <text className="training-axis-tick" x={leftX + barWidth / 2} y={mapY(score) - 6}
                  textAnchor="middle" style={{ fontSize: 11 }}>
                  {score.toFixed(1)}%
                </text>
                <rect x={rightX} y={mapY(baseScore)} width={barWidth} height={baseH} fill={color} rx={3} opacity={0.35} />
                <text className="training-axis-tick" x={rightX + barWidth / 2} y={mapY(baseScore) - 6}
                  textAnchor="middle" style={{ fontSize: 11 }}>
                  {baseScore.toFixed(1)}%
                </text>
                <text className="training-axis-tick training-axis-tick-x" x={cx} y={mapY(0) + 22}>
                  {b.name.toUpperCase()}
                </text>
              </g>
            );
          }

          return (
            <g key={b.name}>
              <rect x={cx - barWidth / 2} y={mapY(score)} width={barWidth} height={barH} fill={color} rx={3} />
              <text className="training-axis-tick" x={cx} y={mapY(score) - 6}
                textAnchor="middle" style={{ fontSize: 11 }}>
                {score.toFixed(1)}%
              </text>
              <text className="training-axis-tick training-axis-tick-x" x={cx} y={mapY(0) + 22}>
                {b.name.toUpperCase()}
              </text>
            </g>
          );
        })}

        {/* Legend */}
        {hasBase && (
          <g transform={`translate(${BAR_CHART_WIDTH - BAR_BOUNDS.right - 180}, ${BAR_BOUNDS.top - 28})`}>
            <rect x={0} y={3} width={14} height={14} fill={BAR_COLORS[0]} rx={2} />
            <text className="training-legend-label" x={20} y={14}>Model</text>
            <rect x={80} y={3} width={14} height={14} fill={BAR_COLORS[0]} rx={2} opacity={0.35} />
            <text className="training-legend-label" x={100} y={14}>Base</text>
          </g>
        )}

        {/* Y axis label */}
        <text className="training-axis-label" x={24} y={BAR_CHART_HEIGHT / 2}
          transform={`rotate(-90 24 ${BAR_CHART_HEIGHT / 2})`}>
          Accuracy
        </text>
      </svg>
    </div>
  );
}

function EvalResultView({ job, result, onBack }: {
  job: RemoteJobRecord;
  result: TrainingResult;
  onBack: () => void;
}) {
  const allBenchmarks = result.benchmarks || [];
  const allBaseBenchmarks = result.base_benchmarks || [];
  // Only show benchmarks that were actually evaluated
  const benchmarks = allBenchmarks.filter((b) => b.num_examples > 0);
  const baseBenchmarks = allBaseBenchmarks.filter((b) => b.num_examples > 0);
  const hasBase = baseBenchmarks.length > 0;
  const baseMap = new Map(baseBenchmarks.map((b) => [b.name, b]));

  // Compute average from correct/total
  const avgScore = benchmarks.length > 0
    ? benchmarks.reduce((sum, b) => sum + pct(b), 0) / benchmarks.length
    : 0;

  return (
    <div className="panel stack-lg">
      <BackButton onBack={onBack} />
      <h3>{job.modelName || job.jobId} — Evaluation Results</h3>

      <div className="stats-grid">
        <div className="metric-card">
          <span className="metric-label">Model</span>
          <span className="metric-value text-sm">{job.modelName || result.model_path?.split("/").pop() || job.jobId}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Average Score</span>
          <span className="metric-value">{avgScore.toFixed(1)}%</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Benchmarks</span>
          <span className="metric-value">{benchmarks.length}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Cluster</span>
          <span className="metric-value text-sm">{job.clusterName}</span>
        </div>
      </div>

      <BenchmarkBarChart benchmarks={benchmarks} baseBenchmarks={baseBenchmarks} />

      {benchmarks.length > 0 && (
        <div className="docs-table-wrap">
          <table className="docs-table">
            <thead>
              <tr>
                <th>Benchmark</th>
                <th>Score</th>
                <th>Correct</th>
                <th>Total</th>
                {hasBase && <th>Base Score</th>}
                {hasBase && <th>Delta</th>}
              </tr>
            </thead>
            <tbody>
              {benchmarks.map((b) => {
                const score = pct(b);
                const base = baseMap.get(b.name);
                const baseScore = base ? pct(base) : null;
                const delta = baseScore != null ? score - baseScore : null;
                return (
                  <tr key={b.name}>
                    <td style={{ fontWeight: 500 }}>{b.name.toUpperCase()}</td>
                    <td>{score.toFixed(1)}%</td>
                    <td>{b.correct}</td>
                    <td>{b.num_examples}</td>
                    {hasBase && <td>{baseScore != null ? `${baseScore.toFixed(1)}%` : "-"}</td>}
                    {hasBase && (
                      <td style={{ color: delta && delta > 0 ? "var(--clr-success)" : delta && delta < 0 ? "var(--clr-error)" : undefined }}>
                        {delta != null ? `${delta > 0 ? "+" : ""}${delta.toFixed(1)}%` : "-"}
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {result.model_path && (
        <div className="run-row-path" style={{ opacity: 0.7 }}>
          Model: {result.model_path}
        </div>
      )}
    </div>
  );
}

function TrainingResultView({ job, result, onBack }: {
  job: RemoteJobRecord;
  result: TrainingResult;
  onBack: () => void;
}) {
  const history = result.training_history ?? null;

  // Build key-value pairs from result, excluding internal/complex fields
  const exclude = new Set(["status", "traceback", "error", "result", "training_history"]);
  const entries = Object.entries(result).filter(
    ([k, v]) => !exclude.has(k) && v != null && v !== "" && typeof v !== "object",
  );

  return (
    <div className="panel stack-lg">
      <BackButton onBack={onBack} />
      <h3>{job.modelName || job.jobId} — Training Result</h3>

      <div className="stats-grid">
        {result.epochs_completed != null && (
          <div className="metric-card">
            <span className="metric-label">Epochs</span>
            <span className="metric-value">{result.epochs_completed}</span>
          </div>
        )}
        {history && history.epochs.length > 0 && (
          <>
            <div className="metric-card">
              <span className="metric-label">Final Train Loss</span>
              <span className="metric-value">
                {history.epochs[history.epochs.length - 1].train_loss.toFixed(6)}
              </span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Final Val Loss</span>
              <span className="metric-value">
                {history.epochs[history.epochs.length - 1].validation_loss.toFixed(6)}
              </span>
            </div>
          </>
        )}
        <div className="metric-card">
          <span className="metric-label">Method</span>
          <span className="metric-value text-sm">{job.trainingMethod}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Cluster</span>
          <span className="metric-value text-sm">{job.clusterName}</span>
        </div>
      </div>

      {entries.length > 0 && (
        <div className="docs-table-wrap">
          <table className="docs-table">
            <thead>
              <tr><th>Field</th><th>Value</th></tr>
            </thead>
            <tbody>
              {entries.map(([k, v]) => (
                <tr key={k}>
                  <td>{k.replace(/_/g, " ")}</td>
                  <td className="text-mono text-sm">{String(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {history && <TrainingCurvesView history={history} />}
    </div>
  );
}

const INTERP_LABELS: Record<string, string> = {
  "logit-lens": "Logit Lens",
  "activation-pca": "Activation PCA",
  "activation-patch": "Activation Patching",
};

function InterpResultView({ job, result, onBack }: {
  job: RemoteJobRecord;
  result: TrainingResult;
  onBack: () => void;
}) {
  const jobType = result.job_type ?? "";
  const label = INTERP_LABELS[jobType] ?? jobType;

  return (
    <div className="panel stack-lg">
      <BackButton onBack={onBack} />
      <h3>{label} — Remote Result</h3>
      <div className="stats-grid">
        <div className="metric-card">
          <span className="metric-label">Analysis</span>
          <span className="metric-value text-sm">{label}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Cluster</span>
          <span className="metric-value text-sm">{job.clusterName}</span>
        </div>
      </div>
      {jobType === "logit-lens" && (
        <LogitLensResults result={result as unknown as LogitLensResult} />
      )}
      {jobType === "activation-pca" && (
        <ActivationPcaResults result={result as unknown as PcaResult} />
      )}
      {jobType === "activation-patch" && (
        <ActivationPatchingResults result={result as unknown as PatchingResult} />
      )}
    </div>
  );
}
