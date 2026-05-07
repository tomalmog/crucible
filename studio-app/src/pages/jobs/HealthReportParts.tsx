import type { ReactNode } from "react";

export function ReportMetric({
  icon,
  label,
  value,
}: {
  icon: ReactNode;
  label: string;
  value: string;
}): ReactNode {
  return (
    <div className="health-report-metric">
      <span>{icon}</span>
      <span className="metric-label">{label}</span>
      <strong>{value || "-"}</strong>
    </div>
  );
}

export function ReportSection({ title, body }: { title: string; body: string }): ReactNode {
  return (
    <article className="health-report-section">
      <strong>{title}</strong>
      <p>{body}</p>
    </article>
  );
}

export function ReportRow({ label, value }: { label: string; value: string }): ReactNode {
  if (!value) return null;
  return (
    <tr>
      <td>{label}</td>
      <td className="text-mono text-sm">{value}</td>
    </tr>
  );
}

export function configString(config: Record<string, unknown>, key: string): string {
  const value = config[key];
  return typeof value === "string" ? value : "";
}

export function artifactRows(result: Record<string, unknown> | null): [string, string][] {
  if (!result) return [];
  return Object.entries(result)
    .filter(([, value]) => typeof value === "string" || typeof value === "number")
    .map(([key, value]) => [key, String(value)]);
}
