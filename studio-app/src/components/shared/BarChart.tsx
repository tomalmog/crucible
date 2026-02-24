interface BarChartProps {
  rows: Array<{ label: string; value: number }>;
  maxValue?: number;
}

export function BarChart({ rows, maxValue }: BarChartProps) {
  const max = maxValue ?? Math.max(...rows.map((r) => r.value), 1);
  return (
    <div>
      {rows.map((row) => {
        const width = Math.max(4, Math.round((row.value / max) * 100));
        return (
          <div className="bar-row" key={row.label}>
            <div className="bar-label">
              <span>{row.label.length > 40 ? `${row.label.slice(0, 40)}...` : row.label}</span>
              <strong>{row.value}</strong>
            </div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${width}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
