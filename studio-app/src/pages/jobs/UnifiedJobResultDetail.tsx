import { useCallback, useEffect, useMemo, useState } from "react";
import { Loader2, Trophy } from "lucide-react";
import { useCrucible } from "../../context/CrucibleContext";
import type { JobRecord } from "../../types/jobs";
import type { CommandTaskStatus, TrainingHistory } from "../../types";
import { getJobResult, getJobLogs, getCachedJobResult } from "../../api/jobsApi";
import { loadTrainingHistory } from "../../api/studioApi";
import { TrainingCurvesView } from "../training/TrainingCurvesView";
import { LogitLensResults } from "../interp/LogitLensResults";
import { ActivationPcaResults } from "../interp/ActivationPcaResults";
import { ActivationPatchingResults } from "../interp/ActivationPatchingResults";
import type { LogitLensResult, PcaResult, PatchingResult } from "../../types/interp";
import { DetailHeader } from "./RetryButton";

// ── Shared types/constants ─────────────────────────────────────────────

interface UnifiedJobResultDetailProps {
  job: JobRecord;
  /** For local jobs, the CommandTaskStatus with stdout/stderr. */
  localTask?: CommandTaskStatus;
  onBack: () => void;
}

interface EvalBenchmark {
  name: string;
  score: number;
  num_examples: number;
  correct: number;
}

interface ResultData {
  status?: string;
  model_path?: string;
  history_path?: string;
  epochs_completed?: number;
  run_id?: string;
  error?: string;
  traceback?: string;
  training_history?: TrainingHistory;
  job_type?: string;
  average_score?: number;
  benchmarks?: EvalBenchmark[];
  base_benchmarks?: EvalBenchmark[];
  [key: string]: unknown;
}

const TRAINING_TYPES = new Set([
  "train", "sft", "dpo-train", "rlhf-train", "lora-train",
  "distill", "domain-adapt", "grpo-train", "qlora-train",
  "kto-train", "orpo-train", "multimodal-train", "rlvr-train",
]);

const INTERP_TYPES = new Set(["logit-lens", "activation-pca", "activation-patch"]);

const INTERP_LABELS: Record<string, string> = {
  "logit-lens": "Logit Lens",
  "activation-pca": "Activation PCA",
  "activation-patch": "Activation Patching",
};

function extractCrucibleError(stderr: string): string | null {
  const lines = stderr.split("\n");
  for (let i = lines.length - 1; i >= 0; i--) {
    const match = lines[i].match(/Crucible\w+Error:\s*(.+)/);
    if (match) return match[1].trim();
  }
  return null;
}

function parseKeyValueOutput(stdout: string): Record<string, string> {
  const result: Record<string, string> = {};
  for (const line of stdout.split("\n")) {
    const eq = line.indexOf("=");
    if (eq > 0) {
      const key = line.slice(0, eq).trim();
      const val = line.slice(eq + 1).trim();
      if (key && val && val !== "-") result[key] = val;
    }
  }
  return result;
}

// ── Shared logs section ─────────────────────────────────────────────────

/** Log viewer shown at the bottom of result views. For local jobs pass `logs`
 *  directly; for remote jobs pass `jobId` and logs are fetched automatically. */
function LogsSection({ logs, jobId, jobState }: {
  logs?: string;
  jobId?: string;
  jobState?: string;
}) {
  const { dataRoot } = useCrucible();
  const [remoteLogs, setRemoteLogs] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

  useEffect(() => {
    if (logs || !jobId || !dataRoot) return;
    setLoading(true);
    getJobLogs(dataRoot, jobId, jobState)
      .then((c) => setRemoteLogs(c?.trim() || ""))
      .catch((err) => setFetchError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false));
  }, [logs, jobId, jobState, dataRoot]);

  const content = logs ?? remoteLogs;

  if (!logs && !jobId) return null;

  return (
    <div>
      <div style={{ textAlign: "center", fontSize: "1rem", fontWeight: 600, marginTop: 24, marginBottom: 12 }}>Logs</div>
      {loading && (
        <div style={{ display: "flex", justifyContent: "center", padding: 16 }}>
          <Loader2 size={16} className="spin" />
        </div>
      )}
      {fetchError && <div className="error-alert">{fetchError}</div>}
      {content && <pre className="console" style={{ fontSize: "0.8125rem", maxHeight: 300 }}>{content}</pre>}
      {!loading && !fetchError && !content && (
        <p className="text-muted">No logs available.</p>
      )}
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────

export function UnifiedJobResultDetail({ job, localTask, onBack }: UnifiedJobResultDetailProps) {
  const isLocal = job.backend === "local";

  if (isLocal && localTask) {
    return <LocalResultRouter job={job} localTask={localTask} onBack={onBack} />;
  }

  if (isLocal) {
    return <LocalResultFallback job={job} onBack={onBack} />;
  }

  return <RemoteResultRouter job={job} onBack={onBack} />;
}

// ── Local result fallback (reconstruct from persisted stdout) ────────────

function LocalResultFallback({ job, onBack }: { job: JobRecord; onBack: () => void }) {
  // Build a synthetic CommandTaskStatus from persisted JobRecord data
  // so we can reuse the exact same views as LocalResultRouter.
  const syntheticTask = useMemo<CommandTaskStatus>(() => ({
    task_id: job.jobId,
    status: job.state === "completed" ? "completed" : "failed",
    command: job.jobType,
    args: [],
    exit_code: job.state === "completed" ? 0 : 1,
    stdout: job.stdout || "",
    stderr: job.stderr || "",
    elapsed_seconds: 0,
    estimated_total_seconds: 0,
    remaining_seconds: 0,
    progress_percent: 100,
    label: job.label || null,
  }), [job]);

  return <LocalResultRouter job={job} localTask={syntheticTask} onBack={onBack} />;
}

// ── Local result router (parse stdout) ─────────────────────────────────

function LocalResultRouter({ job, localTask, onBack }: {
  job: JobRecord;
  localTask: CommandTaskStatus;
  onBack: () => void;
}) {
  const isSweep = job.jobType === "sweep" || job.isSweep;
  const isTraining = TRAINING_TYPES.has(job.jobType);
  const isInterp = INTERP_TYPES.has(job.jobType);
  const isFailed = job.state === "failed" || localTask.status === "failed";
  const config = job.config;

  if (isFailed) return <LocalFailedView job={job} localTask={localTask} onBack={onBack} config={config} />;
  if (isSweep) return <LocalSweepView job={job} localTask={localTask} onBack={onBack} config={config} />;
  if (isInterp) return <LocalInterpView job={job} localTask={localTask} onBack={onBack} config={config} />;
  if (isTraining) return <LocalTrainingView job={job} localTask={localTask} onBack={onBack} config={config} />;
  return <LocalGenericView job={job} localTask={localTask} onBack={onBack} config={config} />;
}

function LocalFailedView({ job, localTask, onBack, config }: { job: JobRecord; localTask: CommandTaskStatus; onBack: () => void; config: Record<string, unknown> }) {
  const error = extractCrucibleError(localTask.stderr);
  return (
    <div className="panel stack-lg">
      <DetailHeader onBack={onBack} config={config} />
      <h3>Job Failed: {job.label || job.jobType}</h3>
      {error && <div className="error-alert-prominent">{error}</div>}
      {localTask.stderr && (
        <details><summary className="error-text">Full traceback</summary><pre className="console console-short">{localTask.stderr}</pre></details>
      )}
      <LogsSection logs={localTask.stdout} />
    </div>
  );
}

function LocalTrainingView({ job, localTask, onBack, config }: { job: JobRecord; localTask: CommandTaskStatus; onBack: () => void; config: Record<string, unknown> }) {
  const result = useMemo(() => parseKeyValueOutput(localTask.stdout), [localTask.stdout]);
  const [history, setHistory] = useState<TrainingHistory | null>(null);

  useEffect(() => {
    const hp = result.history_path;
    if (hp) loadTrainingHistory(hp).then(setHistory).catch(() => setHistory(null));
  }, [result.history_path]);

  return (
    <div className="panel stack-lg">
      <DetailHeader onBack={onBack} config={config} />
      <h3>{job.label || job.jobType} — Result</h3>
      <div className="stats-grid">
        {result.epochs_completed && (
          <div className="metric-card"><span className="metric-label">Epochs</span><span className="metric-value">{result.epochs_completed}</span></div>
        )}
        {history && history.epochs.length > 0 && (
          <>
            <div className="metric-card"><span className="metric-label">Final Train Loss</span><span className="metric-value">{history.epochs[history.epochs.length - 1].train_loss.toFixed(6)}</span></div>
            <div className="metric-card"><span className="metric-label">Final Val Loss</span><span className="metric-value">{history.epochs[history.epochs.length - 1].validation_loss.toFixed(6)}</span></div>
          </>
        )}
      </div>
      {Object.keys(result).length > 0 && (
        <div className="docs-table-wrap">
          <table className="docs-table">
            <thead><tr><th>Field</th><th>Value</th></tr></thead>
            <tbody>{Object.entries(result).map(([k, v]) => (
              <tr key={k}><td>{k.replace(/_/g, " ")}</td><td className="text-mono text-sm">{v}</td></tr>
            ))}</tbody>
          </table>
        </div>
      )}
      {history && <TrainingCurvesView history={history} />}
      <LogsSection logs={localTask.stdout} />
    </div>
  );
}

function LocalInterpView({ job, localTask, onBack, config }: { job: JobRecord; localTask: CommandTaskStatus; onBack: () => void; config: Record<string, unknown> }) {
  const parsed = useMemo(() => { try { return JSON.parse(localTask.stdout); } catch { return null; } }, [localTask.stdout]);
  const label = INTERP_LABELS[job.jobType] ?? job.jobType;

  return (
    <div className="panel stack-lg">
      <DetailHeader onBack={onBack} config={config} />
      <h3>{label} — Result</h3>
      {parsed && job.jobType === "logit-lens" && <LogitLensResults result={parsed as LogitLensResult} />}
      {parsed && job.jobType === "activation-pca" && <ActivationPcaResults result={parsed as PcaResult} />}
      {parsed && job.jobType === "activation-patch" && <ActivationPatchingResults result={parsed as PatchingResult} />}
      {!parsed && localTask.stdout && <pre className="console">{localTask.stdout}</pre>}
    </div>
  );
}

interface SweepTrial { trial_id: number; parameters: Record<string, number>; metric_value: number; model_path: string }
interface SweepData { trials: SweepTrial[]; best_trial_id: number; best_parameters: Record<string, number>; best_metric_value: number }

function LocalSweepView({ localTask, onBack, config }: { job: JobRecord; localTask: CommandTaskStatus; onBack: () => void; config: Record<string, unknown> }) {
  const data = useMemo(() => {
    try {
      const lines = localTask.stdout.split("\n");
      for (let i = lines.length - 1; i >= 0; i--) {
        const line = lines[i].trim();
        if (line.startsWith("{") && line.includes("best_trial_id")) return JSON.parse(line) as SweepData;
      }
      return JSON.parse(localTask.stdout) as SweepData;
    } catch { return null; }
  }, [localTask.stdout]);

  if (!data) {
    return (
      <div className="panel stack"><DetailHeader onBack={onBack} config={config} /><h3>Sweep Results</h3><pre className="console">{localTask.stdout}</pre></div>
    );
  }

  const paramNames = data.trials.length > 0 ? Object.keys(data.trials[0].parameters).sort() : [];

  return (
    <div className="panel stack-lg">
      <DetailHeader onBack={onBack} config={config} />
      <h3>Sweep Results</h3>
      <div className="stats-grid">
        <div className="metric-card"><span className="metric-label">Total Trials</span><span className="metric-value">{data.trials.length}</span></div>
        <div className="metric-card"><span className="metric-label">Best Trial</span><span className="metric-value">#{data.best_trial_id}</span></div>
        <div className="metric-card"><span className="metric-label">Best Metric</span><span className="metric-value">{data.best_metric_value.toFixed(6)}</span></div>
      </div>
      {data.trials.length > 0 && (
        <div className="docs-table-wrap">
          <table className="docs-table">
            <thead><tr><th>Trial</th>{paramNames.map((n) => <th key={n}>{n}</th>)}<th>Metric</th><th>Model</th><th></th></tr></thead>
            <tbody>{data.trials.map((t) => (
              <tr key={t.trial_id} style={t.trial_id === data.best_trial_id ? { background: "var(--bg-active)" } : undefined}>
                <td>#{t.trial_id}</td>
                {paramNames.map((n) => <td key={n}>{t.parameters[n]}</td>)}
                <td>{t.metric_value.toFixed(6)}</td>
                <td className="text-mono text-xs">{t.model_path ? formatPath(t.model_path) : "-"}</td>
                <td>{t.trial_id === data.best_trial_id && <Trophy size={14} />}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function LocalGenericView({ job, localTask, onBack, config }: { job: JobRecord; localTask: CommandTaskStatus; onBack: () => void; config: Record<string, unknown> }) {
  return (
    <div className="panel stack-lg">
      <DetailHeader onBack={onBack} config={config} />
      <h3>{job.label || job.jobType} — Result</h3>
      {localTask.stdout && <pre className="console">{localTask.stdout}</pre>}
      {localTask.stderr && (<details><summary>stderr</summary><pre className="console console-short">{localTask.stderr}</pre></details>)}
    </div>
  );
}

// ── Remote result router (API-fetched) ─────────────────────────────────

function RemoteResultRouter({ job, onBack }: { job: JobRecord; onBack: () => void }) {
  const { dataRoot } = useCrucible();
  const initialResult = useMemo(() => getCachedJobResult(job.jobId) as ResultData | undefined, [job.jobId]);
  const [result, setResult] = useState<ResultData | null>(initialResult ?? null);
  const [loading, setLoading] = useState(!initialResult);
  const [error, setError] = useState<string | null>(null);

  const fetchResult = useCallback(async () => {
    if (!dataRoot) return;
    if (!result) setLoading(true);
    setError(null);
    try {
      const data = await getJobResult(dataRoot, job.jobId, job.state);
      setResult(data as ResultData);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [dataRoot, job.jobId, job.state]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!initialResult) fetchResult();
  }, [fetchResult]); // eslint-disable-line react-hooks/exhaustive-deps

  const config = job.config;

  if (loading) {
    return (
      <div className="panel stack-lg"><DetailHeader onBack={onBack} config={config} />
        <div style={{ display: "flex", justifyContent: "center", padding: 32 }}><Loader2 size={24} className="spin" /></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="panel stack-lg"><DetailHeader onBack={onBack} config={config} />
        <h3>{job.label || job.jobId} — Result</h3>
        <div className="error-alert-prominent">{error}</div>
      </div>
    );
  }

  if (!result || Object.keys(result).length === 0) {
    return (
      <div className="panel stack-lg"><DetailHeader onBack={onBack} config={config} />
        <h3>{job.label || job.jobId} — Result</h3>
        <div className="empty-state"><p>No result.json found on remote cluster.</p></div>
      </div>
    );
  }

  if (job.state === "failed" || result.status === "failed") {
    return <RemoteFailedView job={job} result={result} onBack={onBack} config={config} />;
  }
  if (result.job_type === "eval") return <RemoteEvalView job={job} result={result} onBack={onBack} config={config} />;
  if (INTERP_TYPES.has(result.job_type || "")) return <RemoteInterpView job={job} result={result} onBack={onBack} config={config} />;
  return <RemoteTrainingView job={job} result={result} onBack={onBack} config={config} />;
}

function RemoteFailedView({ job, result, onBack, config }: { job: JobRecord; result: ResultData; onBack: () => void; config: Record<string, unknown> }) {
  return (
    <div className="panel stack-lg">
      <DetailHeader onBack={onBack} config={config} />
      <h3>Job Failed: {job.label || job.jobId}</h3>
      {result.error && <div className="error-alert-prominent">{result.error}</div>}
      {result.traceback && (<details><summary className="error-text">Full traceback</summary><pre className="console console-short">{result.traceback}</pre></details>)}
      <LogsSection jobId={job.jobId} jobState={job.state} />
    </div>
  );
}

function pct(b: EvalBenchmark): number {
  return b.num_examples > 0 ? (b.correct / b.num_examples) * 100 : 0;
}

function RemoteEvalView({ job, result, onBack, config }: { job: JobRecord; result: ResultData; onBack: () => void; config: Record<string, unknown> }) {
  const benchmarks = (result.benchmarks || []).filter((b) => b.num_examples > 0);
  const baseBenchmarks = (result.base_benchmarks || []).filter((b) => b.num_examples > 0);
  const hasBase = baseBenchmarks.length > 0;
  const baseMap = new Map(baseBenchmarks.map((b) => [b.name, b]));
  const avgScore = benchmarks.length > 0
    ? benchmarks.reduce((sum, b) => sum + pct(b), 0) / benchmarks.length : 0;

  return (
    <div className="panel stack-lg">
      <DetailHeader onBack={onBack} config={config} />
      <h3>{job.label || job.jobId} — Evaluation Results</h3>
      <div className="stats-grid">
        <div className="metric-card"><span className="metric-label">Average Score</span><span className="metric-value">{avgScore.toFixed(1)}%</span></div>
        <div className="metric-card"><span className="metric-label">Benchmarks</span><span className="metric-value">{benchmarks.length}</span></div>
        {job.backendCluster && <div className="metric-card"><span className="metric-label">Cluster</span><span className="metric-value text-sm">{job.backendCluster}</span></div>}
      </div>
      {benchmarks.length > 0 && (
        <div className="docs-table-wrap">
          <table className="docs-table">
            <thead><tr><th>Benchmark</th><th>Score</th><th>Correct</th><th>Total</th>{hasBase && <th>Base</th>}{hasBase && <th>Delta</th>}</tr></thead>
            <tbody>{benchmarks.map((b) => {
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
                  {hasBase && <td style={{ color: delta && delta > 0 ? "var(--clr-success)" : delta && delta < 0 ? "var(--clr-error)" : undefined }}>{delta != null ? `${delta > 0 ? "+" : ""}${delta.toFixed(1)}%` : "-"}</td>}
                </tr>
              );
            })}</tbody>
          </table>
        </div>
      )}
      <LogsSection jobId={job.jobId} jobState={job.state} />
    </div>
  );
}

function RemoteTrainingView({ job, result, onBack, config }: { job: JobRecord; result: ResultData; onBack: () => void; config: Record<string, unknown> }) {
  const history = result.training_history ?? null;
  const exclude = new Set(["status", "traceback", "error", "result", "training_history"]);
  const entries = Object.entries(result).filter(([k, v]) => !exclude.has(k) && v != null && v !== "" && typeof v !== "object");

  return (
    <div className="panel stack-lg">
      <DetailHeader onBack={onBack} config={config} />
      <h3>{job.label || job.jobId} — Training Result</h3>
      <div className="stats-grid">
        {result.epochs_completed != null && <div className="metric-card"><span className="metric-label">Epochs</span><span className="metric-value">{result.epochs_completed}</span></div>}
        {history && history.epochs.length > 0 && (
          <>
            <div className="metric-card"><span className="metric-label">Final Train Loss</span><span className="metric-value">{history.epochs[history.epochs.length - 1].train_loss.toFixed(6)}</span></div>
            <div className="metric-card"><span className="metric-label">Final Val Loss</span><span className="metric-value">{history.epochs[history.epochs.length - 1].validation_loss.toFixed(6)}</span></div>
          </>
        )}
        <div className="metric-card"><span className="metric-label">Method</span><span className="metric-value text-sm">{job.jobType}</span></div>
        {job.backendCluster && <div className="metric-card"><span className="metric-label">Cluster</span><span className="metric-value text-sm">{job.backendCluster}</span></div>}
      </div>
      {entries.length > 0 && (
        <div className="docs-table-wrap">
          <table className="docs-table">
            <thead><tr><th>Field</th><th>Value</th></tr></thead>
            <tbody>{entries.map(([k, v]) => (<tr key={k}><td>{k.replace(/_/g, " ")}</td><td className="text-mono text-sm">{String(v)}</td></tr>))}</tbody>
          </table>
        </div>
      )}
      {history && <TrainingCurvesView history={history} />}
      <LogsSection jobId={job.jobId} jobState={job.state} />
    </div>
  );
}

function RemoteInterpView({ job, result, onBack, config }: { job: JobRecord; result: ResultData; onBack: () => void; config: Record<string, unknown> }) {
  const jobType = result.job_type ?? "";
  const label = INTERP_LABELS[jobType] ?? jobType;

  return (
    <div className="panel stack-lg">
      <DetailHeader onBack={onBack} config={config} />
      <h3>{label} — Result</h3>
      <div className="stats-grid">
        <div className="metric-card"><span className="metric-label">Analysis</span><span className="metric-value text-sm">{label}</span></div>
        {job.backendCluster && <div className="metric-card"><span className="metric-label">Cluster</span><span className="metric-value text-sm">{job.backendCluster}</span></div>}
      </div>
      {jobType === "logit-lens" && <LogitLensResults result={result as unknown as LogitLensResult} />}
      {jobType === "activation-pca" && <ActivationPcaResults result={result as unknown as PcaResult} />}
      {jobType === "activation-patch" && <ActivationPatchingResults result={result as unknown as PatchingResult} />}
      <LogsSection jobId={job.jobId} jobState={job.state} />
    </div>
  );
}

function formatPath(p: string): string {
  const parts = p.split("/");
  return parts.length > 3 ? ".../" + parts.slice(-3).join("/") : p;
}
