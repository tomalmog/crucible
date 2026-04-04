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
import { LinearProbeResults } from "../interp/LinearProbeResults";
import { SaeTrainResults, SaeAnalyzeResults } from "../interp/SaeResults";
import { SteerComputeResults, SteerApplyResults } from "../interp/SteerResults";
import { OnnxExportResults } from "../export/OnnxExportResults";
import { SafeTensorsExportResults } from "../export/SafeTensorsExportResults";
import { GgufExportResults } from "../export/GgufExportResults";
import { HfExportResults } from "../export/HfExportResults";
import type { LogitLensResult, PcaResult, PatchingResult, LinearProbeResult, SaeTrainResult, SaeAnalyzeResult, SteerComputeResult, SteerApplyResult } from "../../types/interp";
import type { OnnxExportResult, SafeTensorsExportResult, GgufExportResult, HfExportResult } from "../../types/export";
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
  error?: string;
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

const INTERP_TYPES = new Set([
  "logit-lens", "activation-pca", "activation-patch",
  "linear-probe", "sae-train", "sae-analyze",
  "steer-compute", "steer-apply",
]);

const EXPORT_TYPES = new Set([
  "onnx-export", "safetensors-export", "gguf-export", "hf-export",
]);

const INTERP_LABELS: Record<string, string> = {
  "logit-lens": "Logit Lens",
  "activation-pca": "Activation PCA",
  "activation-patch": "Activation Patching",
  "linear-probe": "Linear Probe",
  "sae-train": "SAE Train",
  "sae-analyze": "SAE Analyze",
  "steer-compute": "Steer Compute",
  "steer-apply": "Steer Apply",
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

  const isExport = job.jobType === "onnx-export" || job.jobType === "safetensors-export" || job.jobType === "gguf-export" || job.jobType === "hf-export";

  const isEval = job.jobType === "eval";

  if (isFailed) return <LocalFailedView job={job} localTask={localTask} onBack={onBack} config={config} />;
  if (isSweep) return <LocalSweepView job={job} localTask={localTask} onBack={onBack} config={config} />;
  if (isEval) return <LocalEvalView job={job} localTask={localTask} onBack={onBack} config={config} />;
  if (isInterp) return <LocalInterpView job={job} localTask={localTask} onBack={onBack} config={config} />;
  if (isExport) return <LocalExportView job={job} localTask={localTask} onBack={onBack} config={config} />;
  if (isTraining) return <LocalTrainingView job={job} localTask={localTask} onBack={onBack} config={config} />;
  return <LocalGenericView job={job} localTask={localTask} onBack={onBack} config={config} />;
}

function LocalFailedView({ job, localTask, onBack, config }: { job: JobRecord; localTask: CommandTaskStatus; onBack: () => void; config: Record<string, unknown> }) {
  const error = extractCrucibleError(localTask.stderr);
  return (
    <div className="panel stack-lg">
      <DetailHeader onBack={onBack} config={config} jobType={job.jobType} />
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
      <DetailHeader onBack={onBack} config={config} jobType={job.jobType} />
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
      <DetailHeader onBack={onBack} config={config} jobType={job.jobType} />
      <h3>{label} — Result</h3>
      {parsed && job.jobType === "logit-lens" && <LogitLensResults result={parsed as LogitLensResult} />}
      {parsed && job.jobType === "activation-pca" && <ActivationPcaResults result={parsed as PcaResult} />}
      {parsed && job.jobType === "activation-patch" && <ActivationPatchingResults result={parsed as PatchingResult} />}
      {parsed && job.jobType === "linear-probe" && <LinearProbeResults result={parsed as LinearProbeResult} />}
      {parsed && job.jobType === "sae-train" && <SaeTrainResults result={parsed as SaeTrainResult} />}
      {parsed && job.jobType === "sae-analyze" && <SaeAnalyzeResults result={parsed as SaeAnalyzeResult} />}
      {parsed && job.jobType === "steer-compute" && <SteerComputeResults result={parsed as SteerComputeResult} />}
      {parsed && job.jobType === "steer-apply" && <SteerApplyResults result={parsed as SteerApplyResult} />}
      {!parsed && localTask.stdout && <pre className="console">{localTask.stdout}</pre>}
    </div>
  );
}

function LocalExportView({ job, localTask, onBack, config }: { job: JobRecord; localTask: CommandTaskStatus; onBack: () => void; config: Record<string, unknown> }) {
  const parsed = useMemo(() => {
    try {
      // Find the last complete JSON object in stdout (export result is always last)
      const lines = localTask.stdout.split("\n");
      for (let i = lines.length - 1; i >= 0; i--) {
        const line = lines[i].trim();
        if (line.startsWith("{") && line.endsWith("}")) {
          return JSON.parse(line);
        }
      }
      // Fallback: try parsing the whole stdout
      return JSON.parse(localTask.stdout);
    } catch { return null; }
  }, [localTask.stdout]);

  const exportLabel = job.jobType === "safetensors-export" ? "SafeTensors Export"
    : job.jobType === "gguf-export" ? "GGUF Export"
    : job.jobType === "hf-export" ? "HuggingFace Export" : "ONNX Export";

  return (
    <div className="panel stack-lg">
      <DetailHeader onBack={onBack} config={config} jobType={job.jobType} />
      <h3>{job.label || exportLabel} — Result</h3>
      {parsed && job.jobType === "onnx-export" && <OnnxExportResults result={parsed as OnnxExportResult} />}
      {parsed && job.jobType === "safetensors-export" && <SafeTensorsExportResults result={parsed as SafeTensorsExportResult} />}
      {parsed && job.jobType === "gguf-export" && <GgufExportResults result={parsed as GgufExportResult} />}
      {parsed && job.jobType === "hf-export" && <HfExportResults result={parsed as HfExportResult} />}
      {!parsed && localTask.stdout && <pre className="console">{localTask.stdout}</pre>}
      <LogsSection logs={localTask.stdout} />
    </div>
  );
}

interface SweepTrial { trial_id: number; parameters: Record<string, number>; metric_value: number; model_path: string }
interface SweepData { trials: SweepTrial[]; best_trial_id: number; best_parameters: Record<string, number>; best_metric_value: number }

function LocalSweepView({ job, localTask, onBack, config }: { job: JobRecord; localTask: CommandTaskStatus; onBack: () => void; config: Record<string, unknown> }) {
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
      <div className="panel stack"><DetailHeader onBack={onBack} config={config} jobType={job.jobType} /><h3>Sweep Results</h3><pre className="console">{localTask.stdout}</pre></div>
    );
  }

  const paramNames = data.trials.length > 0 ? Object.keys(data.trials[0].parameters).sort() : [];

  return (
    <div className="panel stack-lg">
      <DetailHeader onBack={onBack} config={config} jobType={job.jobType} />
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

function LocalEvalView({ job, localTask, onBack, config }: { job: JobRecord; localTask: CommandTaskStatus; onBack: () => void; config: Record<string, unknown> }) {
  const benchmarks = useMemo(() => parseEvalBenchmarks(localTask.stdout), [localTask.stdout]);
  const failedBenchmarks = benchmarks.filter((b) => b.error);
  const passedBenchmarks = benchmarks.filter((b) => !b.error);
  const allFailed = benchmarks.length > 0 && passedBenchmarks.length === 0;
  const avgScore = passedBenchmarks.length > 0
    ? passedBenchmarks.reduce((sum, b) => sum + pct(b), 0) / passedBenchmarks.length : 0;

  // Deduplicate error messages for the summary banner.
  // Strip leading benchmark name prefix (e.g. "MMLU benchmark requires..." →
  // "requires...") so errors with the same root cause collapse into one.
  const normalizeError = (e: string) => e.replace(/^\w[\w-]*\s+benchmark\s+/i, "");
  const uniqueErrors = [...new Set(failedBenchmarks.map((b) => normalizeError(b.error!)))];

  return (
    <div className="panel stack-lg">
      <DetailHeader onBack={onBack} config={config} jobType={job.jobType} />
      <h3>{job.label || job.jobType} — Evaluation Results</h3>
      {allFailed && (
        <div className="error-alert-prominent">
          All {benchmarks.length} benchmarks failed to run.
          {uniqueErrors.length === 1
            ? ` ${uniqueErrors[0].charAt(0).toUpperCase() + uniqueErrors[0].slice(1)}`
            : uniqueErrors.map((e, i) => <div key={i} style={{ marginTop: 4, fontSize: "0.8125rem" }}>• {e}</div>)}
        </div>
      )}
      {!allFailed && failedBenchmarks.length > 0 && (
        <div className="error-alert">
          {failedBenchmarks.length} of {benchmarks.length} benchmarks failed:{" "}
          {failedBenchmarks.map((b) => b.name.toUpperCase()).join(", ")}
        </div>
      )}
      <div className="stats-grid">
        <div className="metric-card"><span className="metric-label">Average Score</span><span className="metric-value">{avgScore.toFixed(1)}%</span></div>
        <div className="metric-card"><span className="metric-label">Benchmarks</span><span className="metric-value">{passedBenchmarks.length}{failedBenchmarks.length > 0 ? ` / ${benchmarks.length}` : ""}</span></div>
        {failedBenchmarks.length > 0 && (
          <div className="metric-card"><span className="metric-label">Errors</span><span className="metric-value" style={{ color: "var(--error)" }}>{failedBenchmarks.length}</span></div>
        )}
      </div>
      {passedBenchmarks.length > 0 && <BenchmarkBarChart benchmarks={passedBenchmarks} baseBenchmarks={[]} />}
      {benchmarks.length > 0 ? (
        <div className="docs-table-wrap">
          <table className="docs-table">
            <thead><tr><th>Benchmark</th><th>Score</th><th>Correct</th><th>Total</th><th>Status</th></tr></thead>
            <tbody>{benchmarks.map((b) => (
              <tr key={b.name} style={b.error ? { opacity: 0.7 } : undefined}>
                <td style={{ fontWeight: 500 }}>{b.name.toUpperCase()}</td>
                <td>{b.error ? "—" : `${pct(b).toFixed(1)}%`}</td>
                <td>{b.error ? "—" : b.correct}</td>
                <td>{b.error ? "—" : b.num_examples}</td>
                <td>{b.error
                  ? <span className="error-text" title={b.error} style={{ fontSize: "0.75rem" }}>Error: {b.error.length > 60 ? b.error.slice(0, 60) + "…" : b.error}</span>
                  : <span style={{ color: "var(--success)", fontSize: "0.75rem" }}>OK</span>
                }</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      ) : localTask.stdout ? (
        <p className="text-muted text-sm">Could not parse benchmark results from output. Check logs below.</p>
      ) : null}
      <LogsSection logs={localTask.stdout} />
    </div>
  );
}

function LocalGenericView({ job, localTask, onBack, config }: { job: JobRecord; localTask: CommandTaskStatus; onBack: () => void; config: Record<string, unknown> }) {
  return (
    <div className="panel stack-lg">
      <DetailHeader onBack={onBack} config={config} jobType={job.jobType} />
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
      <div className="panel stack-lg"><DetailHeader onBack={onBack} config={config} jobType={job.jobType} />
        <div style={{ display: "flex", justifyContent: "center", padding: 32 }}><Loader2 size={24} className="spin" /></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="panel stack-lg"><DetailHeader onBack={onBack} config={config} jobType={job.jobType} />
        <h3>{job.label || job.jobId} — Result</h3>
        <div className="error-alert-prominent">{error}</div>
      </div>
    );
  }

  if (!result || Object.keys(result).length === 0) {
    return (
      <div className="panel stack-lg"><DetailHeader onBack={onBack} config={config} jobType={job.jobType} />
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
  if (result.job_type === "sweep" || result.trials) return <RemoteSweepView job={job} result={result} onBack={onBack} config={config} />;
  if (EXPORT_TYPES.has(result.job_type || "") || EXPORT_TYPES.has(job.jobType)) return <RemoteExportView job={job} result={result} onBack={onBack} config={config} />;
  if (TRAINING_TYPES.has(result.job_type || "") || TRAINING_TYPES.has(job.jobType)) return <RemoteTrainingView job={job} result={result} onBack={onBack} config={config} />;
  return <RemoteGenericView job={job} result={result} onBack={onBack} config={config} />;
}

function RemoteFailedView({ job, result, onBack, config }: { job: JobRecord; result: ResultData; onBack: () => void; config: Record<string, unknown> }) {
  const partialBenchmarks = (result.benchmarks || []).filter((b) => b.num_examples > 0);
  const isPartialEval = partialBenchmarks.length > 0;

  return (
    <div className="panel stack-lg">
      <DetailHeader onBack={onBack} config={config} jobType={job.jobType} />
      <h3>Job Failed: {job.label || job.jobId}</h3>
      {result.error && <div className="error-alert-prominent">{result.error}</div>}
      {isPartialEval && (
        <>
          <div className="info-banner">
            Partial results: {partialBenchmarks.length} of {(result as any).benchmarks_total ?? "?"} benchmarks completed before failure
          </div>
          <BenchmarkBarChart benchmarks={partialBenchmarks} baseBenchmarks={[]} />
        </>
      )}
      {result.traceback && (<details><summary className="error-text">Full traceback</summary><pre className="console console-short">{result.traceback}</pre></details>)}
      <LogsSection jobId={job.jobId} jobState={job.state} />
    </div>
  );
}

function pct(b: EvalBenchmark): number {
  return b.num_examples > 0 ? (b.correct / b.num_examples) * 100 : 0;
}

// ── SVG benchmark bar chart ─────────────────────────────────────────────

const BAR_CHART_WIDTH = 1000;
const BAR_CHART_HEIGHT = 400;
const BAR_BOUNDS = { top: 58, right: 28, bottom: 100, left: 86 };
const BAR_COLORS = [
  "#769FCD", "#d97706", "#34d399", "#f87171", "#a78bfa", "#fb923c", "#38bdf8",
];

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

  const yTicks = [0, 25, 50, 75, 100];
  const mapY = (v: number) => BAR_BOUNDS.top + chartH - (v / 100) * chartH;
  const mapX = (i: number) => BAR_BOUNDS.left + groupWidth * i + groupWidth / 2;

  return (
    <div className="training-chart-card">
      <svg viewBox={`0 0 ${BAR_CHART_WIDTH} ${BAR_CHART_HEIGHT}`} className="training-chart-svg">
        <text className="training-chart-title" x={BAR_CHART_WIDTH / 2} y={30}>
          Benchmark Scores
        </text>
        {yTicks.map((v) => (
          <g key={`y-${v}`}>
            <line className="training-grid-line" x1={BAR_BOUNDS.left} x2={BAR_CHART_WIDTH - BAR_BOUNDS.right} y1={mapY(v)} y2={mapY(v)} />
            <text className="training-axis-tick training-axis-tick-y" x={BAR_BOUNDS.left - 10} y={mapY(v) + 4}>{v}%</text>
          </g>
        ))}
        <line className="training-axis-line" x1={BAR_BOUNDS.left} x2={BAR_CHART_WIDTH - BAR_BOUNDS.right} y1={mapY(0)} y2={mapY(0)} />
        <line className="training-axis-line" x1={BAR_BOUNDS.left} x2={BAR_BOUNDS.left} y1={BAR_BOUNDS.top} y2={mapY(0)} />
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
                <text className="training-axis-tick" x={leftX + barWidth / 2} y={mapY(score) - 6} textAnchor="middle" style={{ fontSize: 11 }}>{score.toFixed(1)}%</text>
                <rect x={rightX} y={mapY(baseScore)} width={barWidth} height={baseH} fill={color} rx={3} opacity={0.35} />
                <text className="training-axis-tick" x={rightX + barWidth / 2} y={mapY(baseScore) - 6} textAnchor="middle" style={{ fontSize: 11 }}>{baseScore.toFixed(1)}%</text>
                <text className="training-axis-tick training-axis-tick-x" x={cx} y={mapY(0) + 22}>{b.name.toUpperCase()}</text>
              </g>
            );
          }

          return (
            <g key={b.name}>
              <rect x={cx - barWidth / 2} y={mapY(score)} width={barWidth} height={barH} fill={color} rx={3} />
              <text className="training-axis-tick" x={cx} y={mapY(score) - 6} textAnchor="middle" style={{ fontSize: 11 }}>{score.toFixed(1)}%</text>
              <text className="training-axis-tick training-axis-tick-x" x={cx} y={mapY(0) + 22}>{b.name.toUpperCase()}</text>
            </g>
          );
        })}
        {hasBase && (
          <g transform={`translate(${BAR_CHART_WIDTH - BAR_BOUNDS.right - 180}, ${BAR_BOUNDS.top - 28})`}>
            <rect x={0} y={3} width={14} height={14} fill={BAR_COLORS[0]} rx={2} />
            <text className="training-legend-label" x={20} y={14}>Model</text>
            <rect x={80} y={3} width={14} height={14} fill={BAR_COLORS[0]} rx={2} opacity={0.35} />
            <text className="training-legend-label" x={100} y={14}>Base</text>
          </g>
        )}
        <text className="training-axis-label" x={24} y={BAR_CHART_HEIGHT / 2} transform={`rotate(-90 24 ${BAR_CHART_HEIGHT / 2})`}>Accuracy</text>
      </svg>
    </div>
  );
}

function RemoteEvalView({ job, result, onBack, config }: { job: JobRecord; result: ResultData; onBack: () => void; config: Record<string, unknown> }) {
  const allBenchmarks = result.benchmarks || [];
  const failedBenchmarks = allBenchmarks.filter((b) => b.error || b.num_examples === 0);
  const benchmarks = allBenchmarks.filter((b) => !b.error && b.num_examples > 0);
  const baseBenchmarks = (result.base_benchmarks || []).filter((b) => b.num_examples > 0);
  const hasBase = baseBenchmarks.length > 0;
  const baseMap = new Map(baseBenchmarks.map((b) => [b.name, b]));
  const allFailed = allBenchmarks.length > 0 && benchmarks.length === 0;
  const avgScore = benchmarks.length > 0
    ? benchmarks.reduce((sum, b) => sum + pct(b), 0) / benchmarks.length : 0;

  const normalizeError = (e: string) => e.replace(/^\w[\w-]*\s+benchmark\s+/i, "");
  const uniqueErrors = [...new Set(
    (failedBenchmarks.map((b) => b.error).filter(Boolean) as string[]).map(normalizeError),
  )];

  return (
    <div className="panel stack-lg">
      <DetailHeader onBack={onBack} config={config} jobType={job.jobType} />
      <h3>{job.label || job.jobId} — Evaluation Results</h3>
      {allFailed && (
        <div className="error-alert-prominent">
          All {allBenchmarks.length} benchmarks failed to run.
          {uniqueErrors.length === 1
            ? ` ${uniqueErrors[0].charAt(0).toUpperCase() + uniqueErrors[0].slice(1)}`
            : uniqueErrors.map((e, i) => <div key={i} style={{ marginTop: 4, fontSize: "0.8125rem" }}>• {e}</div>)}
        </div>
      )}
      {!allFailed && failedBenchmarks.length > 0 && (
        <div className="error-alert">
          {failedBenchmarks.length} of {allBenchmarks.length} benchmarks failed:{" "}
          {failedBenchmarks.map((b) => b.name.toUpperCase()).join(", ")}
        </div>
      )}
      <div className="stats-grid">
        <div className="metric-card"><span className="metric-label">Average Score</span><span className="metric-value">{avgScore.toFixed(1)}%</span></div>
        <div className="metric-card"><span className="metric-label">Benchmarks</span><span className="metric-value">{benchmarks.length}{failedBenchmarks.length > 0 ? ` / ${allBenchmarks.length}` : ""}</span></div>
        {failedBenchmarks.length > 0 && (
          <div className="metric-card"><span className="metric-label">Errors</span><span className="metric-value" style={{ color: "var(--error)" }}>{failedBenchmarks.length}</span></div>
        )}
        {job.backendCluster && <div className="metric-card"><span className="metric-label">Cluster</span><span className="metric-value text-sm">{job.backendCluster}</span></div>}
      </div>
      {benchmarks.length > 0 && <BenchmarkBarChart benchmarks={benchmarks} baseBenchmarks={baseBenchmarks} />}
      {allBenchmarks.length > 0 && (
        <div className="docs-table-wrap">
          <table className="docs-table">
            <thead><tr><th>Benchmark</th><th>Score</th><th>Correct</th><th>Total</th>{hasBase && <th>Base</th>}{hasBase && <th>Delta</th>}<th>Status</th></tr></thead>
            <tbody>{allBenchmarks.map((b) => {
              const hasFailed = !!(b.error || b.num_examples === 0);
              const score = hasFailed ? 0 : pct(b);
              const base = baseMap.get(b.name);
              const baseScore = base ? pct(base) : null;
              const delta = baseScore != null && !hasFailed ? score - baseScore : null;
              return (
                <tr key={b.name} style={hasFailed ? { opacity: 0.7 } : undefined}>
                  <td style={{ fontWeight: 500 }}>{b.name.toUpperCase()}</td>
                  <td>{hasFailed ? "—" : `${score.toFixed(1)}%`}</td>
                  <td>{hasFailed ? "—" : b.correct}</td>
                  <td>{hasFailed ? "—" : b.num_examples}</td>
                  {hasBase && <td>{!hasFailed && baseScore != null ? `${baseScore.toFixed(1)}%` : "—"}</td>}
                  {hasBase && <td style={{ color: delta && delta > 0 ? "var(--clr-success)" : delta && delta < 0 ? "var(--clr-error)" : undefined }}>{delta != null ? `${delta > 0 ? "+" : ""}${delta.toFixed(1)}%` : "—"}</td>}
                  <td>{hasFailed
                    ? <span className="error-text" title={b.error} style={{ fontSize: "0.75rem" }}>Error{b.error ? `: ${b.error.length > 50 ? b.error.slice(0, 50) + "…" : b.error}` : ""}</span>
                    : <span style={{ color: "var(--success)", fontSize: "0.75rem" }}>OK</span>
                  }</td>
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
      <DetailHeader onBack={onBack} config={config} jobType={job.jobType} />
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
      {history ? <TrainingCurvesView history={history} /> : (
        <p className="text-muted text-sm">Training history unavailable for this run.</p>
      )}
      <LogsSection jobId={job.jobId} jobState={job.state} />
    </div>
  );
}

function RemoteInterpView({ job, result, onBack, config }: { job: JobRecord; result: ResultData; onBack: () => void; config: Record<string, unknown> }) {
  const jobType = result.job_type ?? "";
  const label = INTERP_LABELS[jobType] ?? jobType;

  return (
    <div className="panel stack-lg">
      <DetailHeader onBack={onBack} config={config} jobType={job.jobType} />
      <h3>{label} — Result</h3>
      <div className="stats-grid">
        <div className="metric-card"><span className="metric-label">Analysis</span><span className="metric-value text-sm">{label}</span></div>
        {job.backendCluster && <div className="metric-card"><span className="metric-label">Cluster</span><span className="metric-value text-sm">{job.backendCluster}</span></div>}
      </div>
      {jobType === "logit-lens" && <LogitLensResults result={result as unknown as LogitLensResult} />}
      {jobType === "activation-pca" && <ActivationPcaResults result={result as unknown as PcaResult} />}
      {jobType === "activation-patch" && <ActivationPatchingResults result={result as unknown as PatchingResult} />}
      {jobType === "linear-probe" && <LinearProbeResults result={result as unknown as LinearProbeResult} />}
      {jobType === "sae-train" && <SaeTrainResults result={result as unknown as SaeTrainResult} />}
      {jobType === "sae-analyze" && <SaeAnalyzeResults result={result as unknown as SaeAnalyzeResult} />}
      {jobType === "steer-compute" && <SteerComputeResults result={result as unknown as SteerComputeResult} />}
      {jobType === "steer-apply" && <SteerApplyResults result={result as unknown as SteerApplyResult} />}
      <LogsSection jobId={job.jobId} jobState={job.state} />
    </div>
  );
}

function RemoteSweepView({ job, result, onBack, config }: { job: JobRecord; result: ResultData; onBack: () => void; config: Record<string, unknown> }) {
  const trials = Array.isArray(result.trials) ? (result.trials as Record<string, unknown>[]) : [];
  const bestId = String(result.best_trial_id ?? "");
  return (
    <div className="panel stack-lg">
      <DetailHeader onBack={onBack} config={config} jobType={job.jobType} />
      <h3>Sweep Result</h3>
      <div className="stats-grid">
        <div className="metric-card"><span className="metric-label">Trials</span><span className="metric-value">{trials.length}</span></div>
        {result.best_metric_value != null && <div className="metric-card"><span className="metric-label">Best Metric</span><span className="metric-value">{Number(result.best_metric_value).toFixed(4)}</span></div>}
        {job.backendCluster && <div className="metric-card"><span className="metric-label">Cluster</span><span className="metric-value text-sm">{job.backendCluster}</span></div>}
      </div>
      {result.best_parameters ? (
        <div>
          <h4>Best Parameters</h4>
          <pre className="console console-short">{JSON.stringify(result.best_parameters, null, 2)}</pre>
        </div>
      ) : null}
      {trials.length > 0 && (
        <div className="docs-table-wrap">
          <table className="docs-table">
            <thead><tr><th>Trial</th><th>Parameters</th><th>Metric</th><th></th></tr></thead>
            <tbody>{trials.map((t, i) => {
              const tid = String(t.trial_id ?? i);
              const isBest = tid === bestId;
              return (
                <tr key={tid}>
                  <td className="text-mono">{tid}</td>
                  <td className="text-sm">{JSON.stringify(t.parameters ?? {})}</td>
                  <td className="text-mono">{t.metric_value != null ? Number(t.metric_value as number).toFixed(4) : "\u2014"}</td>
                  <td>{isBest && <Trophy size={14} className="text-success" />}</td>
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

function RemoteExportView({ job, result, onBack, config }: { job: JobRecord; result: ResultData; onBack: () => void; config: Record<string, unknown> }) {
  const exportType = String(result.job_type ?? job.jobType);
  const exclude = new Set(["status", "job_type"]);
  const entries = Object.entries(result).filter(
    ([k, v]) => !exclude.has(k) && v != null && v !== "" && typeof v !== "object"
  );
  return (
    <div className="panel stack-lg">
      <DetailHeader onBack={onBack} config={config} jobType={job.jobType} />
      <h3>Export Result — {exportType}</h3>
      <div className="stats-grid">
        {result.output_path ? <div className="metric-card"><span className="metric-label">Output</span><span className="metric-value text-sm text-mono">{formatPath(String(result.output_path))}</span></div> : null}
        {result.file_size_mb != null && <div className="metric-card"><span className="metric-label">Size</span><span className="metric-value">{Number(result.file_size_mb as number).toFixed(1)} MB</span></div>}
        {job.backendCluster && <div className="metric-card"><span className="metric-label">Cluster</span><span className="metric-value text-sm">{job.backendCluster}</span></div>}
      </div>
      {entries.length > 0 && (
        <div className="docs-table-wrap">
          <table className="docs-table">
            <thead><tr><th>Field</th><th>Value</th></tr></thead>
            <tbody>{entries.map(([k, v]) => (
              <tr key={k}><td>{k.replace(/_/g, " ")}</td><td className="text-mono text-sm">{String(v)}</td></tr>
            ))}</tbody>
          </table>
        </div>
      )}
      <LogsSection jobId={job.jobId} jobState={job.state} />
    </div>
  );
}

function RemoteGenericView({ job, result, onBack, config }: { job: JobRecord; result: ResultData; onBack: () => void; config: Record<string, unknown> }) {
  const exclude = new Set(["status"]);
  const entries = Object.entries(result).filter(
    ([k, v]) => !exclude.has(k) && v != null && v !== "" && typeof v !== "object"
  );
  return (
    <div className="panel stack-lg">
      <DetailHeader onBack={onBack} config={config} jobType={job.jobType} />
      <h3>{job.label || job.jobId} — Result</h3>
      {entries.length > 0 && (
        <div className="docs-table-wrap">
          <table className="docs-table">
            <thead><tr><th>Field</th><th>Value</th></tr></thead>
            <tbody>{entries.map(([k, v]) => (
              <tr key={k}><td>{k.replace(/_/g, " ")}</td><td className="text-mono text-sm">{String(v)}</td></tr>
            ))}</tbody>
          </table>
        </div>
      )}
      <details>
        <summary>Raw result</summary>
        <pre className="console">{JSON.stringify(result, null, 2)}</pre>
      </details>
      <LogsSection jobId={job.jobId} jobState={job.state} />
    </div>
  );
}

function formatPath(p: string): string {
  const parts = p.split("/");
  return parts.length > 3 ? ".../" + parts.slice(-3).join("/") : p;
}
