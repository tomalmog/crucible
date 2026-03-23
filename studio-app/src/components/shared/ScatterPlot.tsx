import { useState } from "react";

interface ScatterPoint {
  x: number;
  y: number;
  label: string;
  text: string;
}

interface ScatterPlotProps {
  points: ScatterPoint[];
  xLabel?: string;
  yLabel?: string;
  width?: number;
  height?: number;
}

const PAD = { top: 20, right: 20, bottom: 40, left: 50 };
const COLORS = [
  "var(--accent)", "var(--blue)", "var(--green)", "var(--orange)",
  "var(--purple)", "var(--teal)", "var(--pink)", "var(--yellow)",
];

export function ScatterPlot({
  points, xLabel = "X", yLabel = "Y", width = 600, height = 400,
}: ScatterPlotProps) {
  const [hovered, setHovered] = useState<number | null>(null);

  if (points.length === 0) return <p className="text-secondary">No data points.</p>;

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  const plotW = width - PAD.left - PAD.right;
  const plotH = height - PAD.top - PAD.bottom;

  function sx(v: number) { return PAD.left + ((v - xMin) / xRange) * plotW; }
  function sy(v: number) { return PAD.top + plotH - ((v - yMin) / yRange) * plotH; }

  const labels = [...new Set(points.map((p) => p.label).filter(Boolean))];
  const colorMap = new Map(labels.map((l, i) => [l, COLORS[i % COLORS.length]]));

  function pointColor(label: string) {
    return colorMap.get(label) ?? "var(--accent)";
  }

  return (
    <div style={{ position: "relative" }}>
      <svg width={width} height={height}>
        {/* Axes */}
        <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + plotH}
          stroke="var(--border)" />
        <line x1={PAD.left} y1={PAD.top + plotH} x2={PAD.left + plotW} y2={PAD.top + plotH}
          stroke="var(--border)" />

        {/* Axis labels */}
        <text x={PAD.left + plotW / 2} y={height - 6} textAnchor="middle"
          className="chart-label">{xLabel}</text>
        <text x={14} y={PAD.top + plotH / 2} textAnchor="middle"
          className="chart-label"
          transform={`rotate(-90, 14, ${PAD.top + plotH / 2})`}>{yLabel}</text>

        {/* Points */}
        {points.map((p, i) => (
          <circle
            key={i}
            cx={sx(p.x)}
            cy={sy(p.y)}
            r={hovered === i ? 5 : 3}
            fill={pointColor(p.label)}
            opacity={hovered === i ? 1 : 0.7}
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered(null)}
            style={{ cursor: "pointer" }}
          />
        ))}
      </svg>

      {/* Tooltip */}
      {hovered !== null && (
        <div
          className="scatter-tooltip"
          style={{
            left: sx(points[hovered].x) + 10,
            top: sy(points[hovered].y) - 10,
          }}
        >
          {points[hovered].label && (
            <strong>{points[hovered].label}</strong>
          )}
          <span>{points[hovered].text}</span>
        </div>
      )}

      {/* Legend */}
      {labels.length > 0 && (
        <div className="scatter-legend">
          {labels.map((l) => (
            <span key={l} className="scatter-legend-item">
              <span className="scatter-legend-dot" style={{ background: colorMap.get(l) }} />
              {l}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
