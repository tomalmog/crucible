import type { ReactNode } from "react";

export interface EvidenceMetric {
  detail?: string;
  label: string;
  tone?: "critical" | "neutral" | "positive";
  value: string;
}

interface EvidenceSummaryProps {
  items: readonly EvidenceMetric[];
  label?: string;
}

export function EvidenceSummary({
  items,
  label = "Evidence summary",
}: EvidenceSummaryProps): ReactNode {
  return (
    <dl className="evidence-summary-grid" aria-label={label}>
      {items.map((item) => (
        <div className={metricClassName(item)} key={item.label}>
          <dt>{item.label}</dt>
          <dd>{item.value}</dd>
          {item.detail && <span>{item.detail}</span>}
        </div>
      ))}
    </dl>
  );
}

function metricClassName(item: EvidenceMetric): string {
  return item.tone ? `evidence-summary-item ${item.tone}` : "evidence-summary-item";
}
