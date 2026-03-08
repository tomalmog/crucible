import { useTrainingRuns } from "../../hooks/useTrainingRuns";
import { EmptyState } from "../../components/shared/EmptyState";

interface TrainingRunHistoryProps {
  dataRoot: string;
}

export function TrainingRunHistory({ dataRoot }: TrainingRunHistoryProps) {
  const { runs, loading, refresh } = useTrainingRuns(dataRoot);

  if (loading) {
    return <p className="text-tertiary">Loading runs...</p>;
  }

  if (runs.length === 0) {
    return <EmptyState title="No training runs" description="Start a training run to see history here." />;
  }

  return (
    <div>
      <div className="row-end gap-top-sm">
        <button className="btn btn-sm" onClick={() => refresh().catch(console.error)}>Refresh</button>
      </div>
      <div className="stack-sm">
        {runs.map((run) => (
          <div key={run.run_id} className="run-row">
            <div className="run-row-header">
              <span className="run-row-id">
                {run.run_id}
              </span>
              <span className={`badge ${run.state === "completed" ? "badge-success" : run.state === "failed" ? "badge-error" : "badge-warning"}`}>
                {run.state}
              </span>
            </div>
            <div className="run-row-meta">
              <span>{run.dataset_name}</span>
              <span>{run.updated_at}</span>
            </div>
            {run.model_path && (
              <span className="run-row-path">
                {run.model_path}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
