import { ScatterPlot } from "../../components/shared/ScatterPlot";
import type { PcaResult } from "../../types/interp";

interface Props {
  result: PcaResult;
}

export function ActivationPcaResults({ result }: Props) {
  const { points, layer_name, explained_variance, granularity } = result;
  const ev0 = explained_variance[0] ? (explained_variance[0] * 100).toFixed(1) : "?";
  const ev1 = explained_variance[1] ? (explained_variance[1] * 100).toFixed(1) : "?";

  return (
    <div className="panel stack-sm">
      <h3>Activation PCA</h3>
      <p className="text-secondary text-sm">
        Layer: <strong>{layer_name}</strong> | Granularity: {granularity} |
        Variance: PC1={ev0}%, PC2={ev1}%
      </p>
      <ScatterPlot
        points={points}
        xLabel={`PC1 (${ev0}%)`}
        yLabel={`PC2 (${ev1}%)`}
        width={640}
        height={440}
      />
    </div>
  );
}
