import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, Trophy } from "lucide-react";
import { CommandTaskStatus, TrainingHistory } from "../../types";
import { loadTrainingHistory } from "../../api/studioApi";
import { TrainingCurvesView } from "../training/TrainingCurvesView";

interface JobResultDetailProps {
  job: CommandTaskStatus;
  onBack: () => void;
}

const TRAINING_COMMANDS = new Set([
  "train", "sft", "dpo-train", "rlhf-train", "lora-train",
  "distill", "domain-adapt", "grpo-train", "qlora-train",
  "kto-train", "orpo-train", "multimodal-train", "rlvr-train",
]);

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

function extractForgeError(stderr: string): string | null {
  const lines = stderr.split("\n");
  for (let i = lines.length - 1; i >= 0; i--) {
    const match = lines[i].match(/Forge\w+Error:\s*(.+)/);
    if (match) return match[1].trim();
  }
  return null;
}

function parseSweepJson(stdout: string): SweepData | null {
  try {
    // Find last JSON object in stdout (skip progress lines)
    const lines = stdout.split("\n");
    for (let i = lines.length - 1; i >= 0; i--) {
      const line = lines[i].trim();
      if (line.startsWith("{") && line.includes("best_trial_id")) {
        return JSON.parse(line) as SweepData;
      }
    }
    return JSON.parse(stdout) as SweepData;
  } catch {
    return null;
  }
}

interface SweepTrial {
  trial_id: number;
  parameters: Record<string, number>;
  metric_value: number;
  model_path: string;
}

interface SweepData {
  trials: SweepTrial[];
  best_trial_id: number;
  best_parameters: Record<string, number>;
  best_metric_value: number;
}

function formatPath(p: string): string {
  const parts = p.split("/");
  return parts.length > 3 ? ".../" + parts.slice(-3).join("/") : p;
}

/** Extract a named CLI flag value from args array (e.g. "--metric" → "validation_loss"). */
function extractArgValue(args: string[], flag: string): string | null {
  const idx = args.indexOf(flag);
  return idx >= 0 && idx + 1 < args.length ? args[idx + 1] : null;
}

export function JobResultDetail({ job, onBack }: JobResultDetailProps) {
  const isSweep = job.command === "sweep";
  const isTraining = TRAINING_COMMANDS.has(job.command);
  const isFailed = job.status === "failed";

  if (isFailed) return <FailedJobView job={job} onBack={onBack} />;
  if (isSweep) return <SweepResultView job={job} onBack={onBack} />;
  if (isTraining) return <TrainingResultView job={job} onBack={onBack} />;
  return <GenericResultView job={job} onBack={onBack} />;
}

function BackButton({ onBack }: { onBack: () => void }) {
  return (
    <button className="btn btn-ghost btn-sm" onClick={onBack}>
      <ArrowLeft size={14} /> Back to Jobs
    </button>
  );
}

function FailedJobView({ job, onBack }: JobResultDetailProps) {
  const error = extractForgeError(job.stderr);
  return (
    <div className="panel stack-lg">
      <BackButton onBack={onBack} />
      <h3>Job Failed: {job.label || job.command}</h3>
      {error && <div className="error-alert-prominent">{error}</div>}
      {job.stderr && (
        <details>
          <summary className="error-text">Full traceback</summary>
          <pre className="console console-short">{job.stderr}</pre>
        </details>
      )}
      {job.stdout && (
        <details>
          <summary>stdout</summary>
          <pre className="console console-short">{job.stdout}</pre>
        </details>
      )}
    </div>
  );
}

function SweepResultView({ job, onBack }: JobResultDetailProps) {
  const data = useMemo(() => parseSweepJson(job.stdout), [job.stdout]);
  const metricName = extractArgValue(job.args, "--metric") ?? "metric";
  const direction = job.args.includes("--maximize") ? "maximize" : "minimize";

  if (!data) {
    return (
      <div className="panel stack">
        <BackButton onBack={onBack} />
        <h3>Sweep Results</h3>
        <pre className="console">{job.stdout}</pre>
      </div>
    );
  }

  const paramNames = data.trials.length > 0
    ? Object.keys(data.trials[0].parameters).sort()
    : [];

  return (
    <div className="panel stack-lg">
      <BackButton onBack={onBack} />
      <h3>Sweep Results</h3>

      <div className="stats-grid">
        <div className="metric-card">
          <span className="metric-label">Total Trials</span>
          <span className="metric-value">{data.trials.length}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Best Trial</span>
          <span className="metric-value">#{data.best_trial_id}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Best {metricName.replace(/_/g, " ")}</span>
          <span className="metric-value">{data.best_metric_value.toFixed(6)}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Metric</span>
          <span className="metric-value text-sm">{metricName}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Direction</span>
          <span className="metric-value text-sm">{direction}</span>
        </div>
      </div>

      {data.trials.length > 0 && (
        <div className="docs-table-wrap">
          <table className="docs-table">
            <thead>
              <tr>
                <th>Trial</th>
                {paramNames.map((n) => <th key={n}>{n}</th>)}
                <th>{metricName.replace(/_/g, " ")}</th>
                <th>Model</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {data.trials.map((t) => (
                <tr key={t.trial_id} style={t.trial_id === data.best_trial_id ? { background: "var(--bg-active)" } : undefined}>
                  <td>#{t.trial_id}</td>
                  {paramNames.map((n) => <td key={n}>{t.parameters[n]}</td>)}
                  <td>{t.metric_value.toFixed(6)}</td>
                  <td className="text-mono text-xs">{t.model_path ? formatPath(t.model_path) : "-"}</td>
                  <td>{t.trial_id === data.best_trial_id && <Trophy size={14} />}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <details>
        <summary>Raw output</summary>
        <pre className="console console-short">{job.stdout}</pre>
      </details>
    </div>
  );
}

function TrainingResultView({ job, onBack }: JobResultDetailProps) {
  const result = useMemo(() => parseKeyValueOutput(job.stdout), [job.stdout]);
  const [history, setHistory] = useState<TrainingHistory | null>(null);

  useEffect(() => {
    const hp = result.history_path;
    if (hp) {
      loadTrainingHistory(hp).then(setHistory).catch(() => setHistory(null));
    }
  }, [result.history_path]);

  return (
    <div className="panel stack-lg">
      <BackButton onBack={onBack} />
      <h3>{job.label || job.command} — Result</h3>

      <div className="stats-grid">
        {result.epochs_completed && (
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
      </div>

      {Object.keys(result).length > 0 && (
        <div className="docs-table-wrap">
          <table className="docs-table">
            <thead>
              <tr><th>Field</th><th>Value</th></tr>
            </thead>
            <tbody>
              {Object.entries(result).map(([k, v]) => (
                <tr key={k}>
                  <td>{k.replace(/_/g, " ")}</td>
                  <td className="text-mono text-sm">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {history && <TrainingCurvesView history={history} />}

      <details>
        <summary>Raw output</summary>
        <pre className="console console-short">{job.stdout}</pre>
      </details>
    </div>
  );
}

function GenericResultView({ job, onBack }: JobResultDetailProps) {
  return (
    <div className="panel stack-lg">
      <BackButton onBack={onBack} />
      <h3>{job.label || job.command} — Result</h3>
      {job.stdout && <pre className="console">{job.stdout}</pre>}
      {job.stderr && (
        <details>
          <summary>stderr</summary>
          <pre className="console console-short">{job.stderr}</pre>
        </details>
      )}
    </div>
  );
}
