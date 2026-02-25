import { useState } from "react";

interface Run {
  run_id: string;
  loss: string;
}

interface RunTableProps {
  runs: Run[];
  onSelect: (runId: string) => void;
  onCompare: (ids: string[]) => void;
}

export function RunTable({ runs, onSelect, onCompare }: RunTableProps) {
  const [selected, setSelected] = useState<Set<string>>(new Set());

  function toggleSelect(runId: string) {
    const next = new Set(selected);
    if (next.has(runId)) next.delete(runId);
    else next.add(runId);
    setSelected(next);
  }

  return (
    <div className="panel">
      {selected.size >= 2 && (
        <div className="row gap-bottom">
          <button className="btn btn-primary btn-sm" onClick={() => onCompare(Array.from(selected))}>
            Compare {selected.size} Runs
          </button>
        </div>
      )}
      {runs.length === 0 ? (
        <p className="text-muted">No experiment runs found. Train a model to see results here.</p>
      ) : (
        <table className="data-table">
          <thead>
            <tr>
              <th></th>
              <th>Run ID</th>
              <th>Loss</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.run_id}>
                <td>
                  <input
                    type="checkbox"
                    checked={selected.has(run.run_id)}
                    onChange={() => toggleSelect(run.run_id)}
                  />
                </td>
                <td>{run.run_id}</td>
                <td>{run.loss}</td>
                <td>
                  <button className="btn btn-ghost btn-sm" onClick={() => onSelect(run.run_id)}>
                    View
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
