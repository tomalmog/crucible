import type { ReactNode } from "react";
import { ScatterPlot } from "../../components/shared/ScatterPlot";
import type { PcaPoint, PcaResult } from "../../types/interp";
import { EvidenceSummary } from "./EvidenceSummary";
import { formatPercent } from "./interpDisplay";

interface Props {
  result: PcaResult;
}

export function ActivationPcaResults({ result }: Props): ReactNode {
  const { points, layer_name, explained_variance, granularity } = result;
  const pc1 = explained_variance[0] ?? 0;
  const pc2 = explained_variance[1] ?? 0;
  const clusters = summarizeLabels(points);
  const exemplars = points.slice(0, 6);

  return (
    <div className="interp-evidence-card stack-md">
      <div className="interp-evidence-header">
        <div>
          <span className="interp-kicker">Activation PCA</span>
          <h3>2D projection of hidden states</h3>
          <p>
            PCA projects activations into two components so clusters, outliers,
            and label separation can be inspected directly.
          </p>
        </div>
        <span className="interp-evidence-badge">{layer_name}</span>
      </div>
      <EvidenceSummary
        items={[
          { label: "Granularity", value: granularity },
          { label: "Examples", value: String(points.length) },
          { label: "PC1 variance", value: formatPercent(pc1), tone: "positive" },
          { label: "PC2 variance", value: formatPercent(pc2) },
        ]}
      />
      <div className="activation-atlas-grid">
        <div className="interp-chart-frame">
          <VarianceBars pc1={pc1} pc2={pc2} />
          <ScatterPlot
            points={points}
            xLabel={`PC1 (${formatPercent(pc1)})`}
            yLabel={`PC2 (${formatPercent(pc2)})`}
            height={500}
          />
        </div>
        <aside className="activation-atlas-sidebar">
          <div>
            <span className="metric-label">Cluster composition</span>
            <div className="cluster-list">
              {clusters.map((cluster) => (
                <div className="cluster-row" key={cluster.label}>
                  <div>
                    <strong>{cluster.label || "unlabeled"}</strong>
                    <span>{cluster.count} points · {formatPercent(cluster.share, 0)}</span>
                  </div>
                  <progress value={cluster.share} max={1} aria-label={`${cluster.label} cluster share`}>
                    {formatPercent(cluster.share, 0)}
                  </progress>
                </div>
              ))}
            </div>
          </div>
          <div>
            <span className="metric-label">Representative points</span>
            <div className="activation-exemplar-list">
              {exemplars.map((point) => (
                <article key={`${point.label}-${point.text}`}>
                  <strong>{point.label || "sample"}</strong>
                  <p>{truncate(point.text, 92)}</p>
                </article>
              ))}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

interface PcaCluster {
  count: number;
  label: string;
  share: number;
}

function VarianceBars({ pc1, pc2 }: { pc1: number; pc2: number }): ReactNode {
  return (
    <div className="variance-bars" aria-label="Explained variance">
      <span className="metric-label">Explained variance</span>
      <div>
        <strong>PC1</strong>
        <progress value={pc1} max={1}>{formatPercent(pc1)}</progress>
        <span>{formatPercent(pc1)}</span>
      </div>
      <div>
        <strong>PC2</strong>
        <progress value={pc2} max={1}>{formatPercent(pc2)}</progress>
        <span>{formatPercent(pc2)}</span>
      </div>
    </div>
  );
}

function summarizeLabels(points: PcaPoint[]): PcaCluster[] {
  const counts = new Map<string, number>();
  for (const point of points) {
    counts.set(point.label, (counts.get(point.label) ?? 0) + 1);
  }
  return Array.from(counts, ([label, count]) => ({
    count,
    label,
    share: points.length > 0 ? count / points.length : 0,
  }))
    .sort((a, b) => b.count - a.count);
}

function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength - 3)}...`;
}
