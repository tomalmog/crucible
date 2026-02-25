import { useState, useEffect } from "react";
import { useForgeCommand } from "../../hooks/useForgeCommand";

interface ExperimentDetailProps {
  runId: string;
  dataRoot: string;
}

export function ExperimentDetail({ runId, dataRoot }: ExperimentDetailProps) {
  const command = useForgeCommand();
  const [detail, setDetail] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      setLoading(true);
      const status = await command.run(dataRoot, ["experiment", "show", runId]);
      if (status.status === "completed" && command.output) {
        try {
          setDetail(JSON.parse(command.output));
        } catch {
          setDetail(null);
        }
      }
      setLoading(false);
    }
    load().catch(console.error);
  }, [runId, dataRoot]);

  if (loading) return <div className="panel"><p>Loading...</p></div>;
  if (!detail) return <div className="panel"><p>No data found for run {runId}</p></div>;

  return (
    <div className="panel stack-lg">
      <h2>Run: {runId}</h2>

      {detail.hyperparameters && Object.keys(detail.hyperparameters).length > 0 && (
        <div>
          <h3>Hyperparameters</h3>
          <div className="stats-grid">
            {Object.entries(detail.hyperparameters).map(([k, v]) => (
              <div key={k} className="metric-card">
                <span className="metric-label">{k}</span>
                <span className="metric-value">{String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {detail.hardware && Object.keys(detail.hardware).length > 0 && (
        <div>
          <h3>Hardware</h3>
          <div className="stats-grid">
            {Object.entries(detail.hardware).map(([k, v]) => (
              <div key={k} className="metric-card">
                <span className="metric-label">{k}</span>
                <span className="metric-value">{String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {detail.metric_names && detail.metric_names.length > 0 && (
        <div>
          <h3>Metrics</h3>
          <table className="data-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Final</th>
                <th>Min</th>
                <th>Max</th>
              </tr>
            </thead>
            <tbody>
              {detail.metric_names.map((name: string) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{detail[`${name}_final`] ?? "-"}</td>
                  <td>{detail[`${name}_min`] ?? "-"}</td>
                  <td>{detail[`${name}_max`] ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
