import { useCallback, useEffect, useRef, useState } from "react";

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
  height?: number;
}

const PAD = { top: 20, right: 30, bottom: 44, left: 56 };
const TICK_COUNT = 5;
const COLORS = [
  "var(--accent)", "var(--blue)", "var(--green)", "var(--orange)",
  "var(--purple)", "var(--teal)", "var(--pink)", "var(--yellow)",
];

/** Pick "nice" round tick values for an axis range. */
function niceInterval(range: number, count: number): number {
  const rough = range / count;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const residual = rough / mag;
  if (residual <= 1.5) return mag;
  if (residual <= 3) return 2 * mag;
  if (residual <= 7) return 5 * mag;
  return 10 * mag;
}

function makeTicks(min: number, max: number, count: number): number[] {
  const range = max - min || 1;
  const step = niceInterval(range, count);
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= max + step * 0.01; v += step) {
    ticks.push(parseFloat(v.toPrecision(6)));
  }
  return ticks;
}

export function ScatterPlot({
  points, xLabel = "X", yLabel = "Y", height = 440,
}: ScatterPlotProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(640);
  const [hovered, setHovered] = useState<number | null>(null);
  const [mouse, setMouse] = useState({ x: 0, y: 0 });

  // Responsive: fill parent width
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width;
        if (w > 0) setWidth(w);
      }
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setMouse({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  }, []);

  if (points.length === 0) return <p className="text-secondary">No data points.</p>;

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  // Add 5% padding so dots don't sit on axes
  const dataXMin = Math.min(...xs);
  const dataXMax = Math.max(...xs);
  const dataYMin = Math.min(...ys);
  const dataYMax = Math.max(...ys);
  const xPad = (dataXMax - dataXMin || 1) * 0.05;
  const yPad = (dataYMax - dataYMin || 1) * 0.05;
  const xMin = dataXMin - xPad;
  const xMax = dataXMax + xPad;
  const yMin = dataYMin - yPad;
  const yMax = dataYMax + yPad;
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  const plotW = width - PAD.left - PAD.right;
  const plotH = height - PAD.top - PAD.bottom;

  function sx(v: number) { return PAD.left + ((v - xMin) / xRange) * plotW; }
  function sy(v: number) { return PAD.top + plotH - ((v - yMin) / yRange) * plotH; }

  const xTicks = makeTicks(xMin, xMax, TICK_COUNT);
  const yTicks = makeTicks(yMin, yMax, TICK_COUNT);

  const labels = [...new Set(points.map((p) => p.label).filter(Boolean))];
  const colorMap = new Map(labels.map((l, i) => [l, COLORS[i % COLORS.length]]));

  function pointColor(label: string) {
    return colorMap.get(label) ?? "var(--accent)";
  }

  // Tooltip positioning: keep within bounds
  const ttX = Math.min(mouse.x + 14, width - 260);
  const ttY = Math.max(mouse.y - 10, 8);

  return (
    <div ref={containerRef} style={{ position: "relative", width: "100%" }}
      onMouseMove={handleMouseMove}>
      <svg width={width} height={height}>
        {/* Grid lines */}
        {xTicks.map((v) => (
          <line key={`gx-${v}`} x1={sx(v)} y1={PAD.top} x2={sx(v)} y2={PAD.top + plotH}
            stroke="var(--border)" strokeOpacity={0.4} strokeDasharray="3,3" />
        ))}
        {yTicks.map((v) => (
          <line key={`gy-${v}`} x1={PAD.left} y1={sy(v)} x2={PAD.left + plotW} y2={sy(v)}
            stroke="var(--border)" strokeOpacity={0.4} strokeDasharray="3,3" />
        ))}

        {/* Axes */}
        <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + plotH}
          stroke="var(--border)" />
        <line x1={PAD.left} y1={PAD.top + plotH} x2={PAD.left + plotW} y2={PAD.top + plotH}
          stroke="var(--border)" />

        {/* X tick marks + labels */}
        {xTicks.map((v) => (
          <g key={`xt-${v}`}>
            <line x1={sx(v)} y1={PAD.top + plotH} x2={sx(v)} y2={PAD.top + plotH + 4}
              stroke="var(--border)" />
            <text x={sx(v)} y={PAD.top + plotH + 16} textAnchor="middle"
              className="chart-tick">{v.toFixed(1)}</text>
          </g>
        ))}
        {/* Y tick marks + labels */}
        {yTicks.map((v) => (
          <g key={`yt-${v}`}>
            <line x1={PAD.left - 4} y1={sy(v)} x2={PAD.left} y2={sy(v)}
              stroke="var(--border)" />
            <text x={PAD.left - 8} y={sy(v) + 4} textAnchor="end"
              className="chart-tick">{v.toFixed(1)}</text>
          </g>
        ))}

        {/* Axis labels */}
        <text x={PAD.left + plotW / 2} y={height - 4} textAnchor="middle"
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
            r={hovered === i ? 6 : 3.5}
            fill={pointColor(p.label)}
            opacity={hovered === i ? 1 : 0.7}
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered(null)}
            style={{ cursor: "pointer", transition: "r 0.1s, opacity 0.1s" }}
          />
        ))}
      </svg>

      {/* Tooltip */}
      {hovered !== null && (
        <div
          className="scatter-tooltip"
          style={{ left: ttX, top: ttY }}
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
