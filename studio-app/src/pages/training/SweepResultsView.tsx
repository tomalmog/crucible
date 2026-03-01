import { useMemo } from "react";
import { ArrowLeft, Trophy, Check } from "lucide-react";

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

interface SweepResultsViewProps {
  output: string;
  onBack: () => void;
  registeredAs?: string | null;
}

export function SweepResultsView({ output, onBack, registeredAs }: SweepResultsViewProps) {
  const data = useMemo((): SweepData | null => {
    try {
      return JSON.parse(output) as SweepData;
    } catch {
      return null;
    }
  }, [output]);

  if (!data) {
    return (
      <div className="panel stack">
        <button className="btn btn-ghost btn-sm" onClick={onBack}>
          <ArrowLeft size={14} /> Back
        </button>
        <h3>Sweep Results</h3>
        <pre className="console">{output}</pre>
      </div>
    );
  }

  const paramNames = data.trials.length > 0
    ? Object.keys(data.trials[0].parameters).sort()
    : [];

  return (
    <div className="panel stack-lg">
      <button className="btn btn-ghost btn-sm" onClick={onBack}>
        <ArrowLeft size={14} /> Back
      </button>
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
          <span className="metric-label">Best Metric</span>
          <span className="metric-value">{data.best_metric_value.toFixed(6)}</span>
        </div>
      </div>

      {registeredAs && (
        <div className="flex-row" style={{ color: "var(--color-success)" }}>
          <Check size={14} />
          <span>Best model registered as &ldquo;{registeredAs}&rdquo;</span>
        </div>
      )}

      <div className="docs-table-wrap">
        <table className="docs-table">
          <thead>
            <tr>
              <th>Trial</th>
              {paramNames.map((n) => <th key={n}>{n}</th>)}
              <th>Metric</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {data.trials.map((t) => (
              <tr key={t.trial_id} style={t.trial_id === data.best_trial_id ? { background: "var(--bg-active)" } : undefined}>
                <td>#{t.trial_id}</td>
                {paramNames.map((n) => <td key={n}>{t.parameters[n]}</td>)}
                <td>{t.metric_value.toFixed(6)}</td>
                <td>{t.trial_id === data.best_trial_id && <Trophy size={14} />}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
