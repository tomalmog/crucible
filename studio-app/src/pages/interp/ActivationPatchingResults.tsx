import type { PatchingResult } from "../../types/interp";

interface Props {
  result: PatchingResult;
}

export function ActivationPatchingResults({ result }: Props) {
  const { layer_results, clean_metric, corrupted_metric, metric } = result;
  const maxRecovery = Math.max(...layer_results.map((r) => Math.abs(r.recovery)), 0.01);

  return (
    <div className="panel stack-sm">
      <h3>Activation Patching Results</h3>
      <p className="text-secondary text-sm">
        Metric: {metric} | Clean: {clean_metric.toFixed(3)} | Corrupted: {corrupted_metric.toFixed(3)}
      </p>
      <div>
        {layer_results.map((layer) => {
          const pct = Math.round((Math.abs(layer.recovery) / maxRecovery) * 100);
          return (
            <div className="bar-row" key={layer.layer_name}>
              <div className="bar-label">
                <span>L{layer.layer_index}</span>
                <strong>{(layer.recovery * 100).toFixed(1)}%</strong>
              </div>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${pct}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
