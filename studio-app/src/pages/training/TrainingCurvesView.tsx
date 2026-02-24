import { TrainingBatchLoss, TrainingEpoch, TrainingHistory } from "../../types";

interface TrainingCurvesViewProps {
  history: TrainingHistory;
}

interface LineSeries {
  label: string;
  color: string;
  values: number[];
}

interface ChartSpec {
  title: string;
  xLabel: string;
  yLabel: string;
  xValues: number[];
  series: LineSeries[];
}

const CHART_WIDTH = 1000;
const CHART_HEIGHT = 430;
const BOUNDS = { top: 58, right: 28, bottom: 78, left: 86 };

export function TrainingCurvesView({ history }: TrainingCurvesViewProps) {
  if (history.epochs.length === 0) {
    return <p className="text-tertiary">History file has no epoch rows.</p>;
  }
  const charts = buildCharts(history.epochs, history.batch_losses);
  return (
    <div className="stack-lg">
      {charts.map((chart) => (
        <LossChart key={chart.title} chart={chart} />
      ))}
    </div>
  );
}

function buildCharts(epochs: TrainingEpoch[], batchLosses: TrainingBatchLoss[]): ChartSpec[] {
  const charts: ChartSpec[] = [];
  if (batchLosses.length > 1) {
    charts.push({
      title: "Training Loss by Step",
      xLabel: "Global Step",
      yLabel: "Loss",
      xValues: batchLosses.map((r) => r.global_step),
      series: [{ label: "train loss", color: "var(--chart-1)", values: batchLosses.map((r) => r.train_loss) }],
    });
  }
  charts.push({
    title: "Epoch Loss Curves",
    xLabel: "Epoch",
    yLabel: "Loss",
    xValues: epochs.map((r) => r.epoch),
    series: [
      { label: "train loss", color: "var(--chart-1)", values: epochs.map((r) => r.train_loss) },
      { label: "validation loss", color: "var(--chart-2)", values: epochs.map((r) => r.validation_loss) },
    ],
  });
  return charts;
}

function LossChart({ chart }: { chart: ChartSpec }) {
  const allY = chart.series.flatMap((s) => s.values);
  const yRange = yDomain(allY);
  const yTicks = createTicks(yRange.min, yRange.max, 5);
  const xTicks = xTickRows(chart.xValues, 6);

  return (
    <div className="training-chart-card">
      <svg viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} className="training-chart-svg">
        <text className="training-chart-title" x={CHART_WIDTH / 2} y={30}>{chart.title}</text>
        {yTicks.map((v) => (
          <g key={`y-${v.toFixed(6)}`}>
            <line className="training-grid-line" x1={BOUNDS.left} x2={CHART_WIDTH - BOUNDS.right} y1={mapY(v, yRange.min, yRange.max)} y2={mapY(v, yRange.min, yRange.max)} />
            <text className="training-axis-tick training-axis-tick-y" x={BOUNDS.left - 10} y={mapY(v, yRange.min, yRange.max) + 4}>{v.toFixed(3)}</text>
          </g>
        ))}
        {xTicks.map((t) => (
          <g key={`x-${t.index}`}>
            <line className="training-grid-line vertical" x1={mapX(t.index, chart.xValues.length)} x2={mapX(t.index, chart.xValues.length)} y1={BOUNDS.top} y2={CHART_HEIGHT - BOUNDS.bottom} />
            <text className="training-axis-tick training-axis-tick-x" x={mapX(t.index, chart.xValues.length)} y={CHART_HEIGHT - BOUNDS.bottom + 22}>{t.label}</text>
          </g>
        ))}
        <line className="training-axis-line" x1={BOUNDS.left} x2={CHART_WIDTH - BOUNDS.right} y1={CHART_HEIGHT - BOUNDS.bottom} y2={CHART_HEIGHT - BOUNDS.bottom} />
        <line className="training-axis-line" x1={BOUNDS.left} x2={BOUNDS.left} y1={BOUNDS.top} y2={CHART_HEIGHT - BOUNDS.bottom} />
        {chart.series.map((s) => (
          <path key={s.label} d={seriesPath(s.values, yRange.min, yRange.max)} fill="none" stroke={s.color} strokeWidth={3} />
        ))}
        <text className="training-axis-label" x={CHART_WIDTH / 2} y={CHART_HEIGHT - 20}>{chart.xLabel}</text>
        <text className="training-axis-label" x={24} y={CHART_HEIGHT / 2} transform={`rotate(-90 24 ${CHART_HEIGHT / 2})`}>{chart.yLabel}</text>
        <g transform={`translate(${CHART_WIDTH - BOUNDS.right - 220}, ${BOUNDS.top - 28})`}>
          {chart.series.map((s, i) => (
            <g key={`legend-${s.label}`} transform={`translate(0, ${i * 18})`}>
              <line x1={0} y1={9} x2={24} y2={9} stroke={s.color} strokeWidth={3} />
              <text className="training-legend-label" x={30} y={13}>{s.label}</text>
            </g>
          ))}
        </g>
      </svg>
    </div>
  );
}

function seriesPath(values: number[], minY: number, maxY: number): string {
  return values.map((v, i) => `${i === 0 ? "M" : "L"}${mapX(i, values.length).toFixed(2)} ${mapY(v, minY, maxY).toFixed(2)}`).join(" ");
}

function mapX(index: number, total: number): number {
  const w = CHART_WIDTH - BOUNDS.left - BOUNDS.right;
  return total <= 1 ? BOUNDS.left + w / 2 : BOUNDS.left + (index / (total - 1)) * w;
}

function mapY(value: number, min: number, max: number): number {
  const h = CHART_HEIGHT - BOUNDS.top - BOUNDS.bottom;
  const range = Math.max(max - min, 0.000001);
  return CHART_HEIGHT - BOUNDS.bottom - ((value - min) / range) * h;
}

function yDomain(values: number[]) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (Math.abs(max - min) < 0.000001) return { min: min - 0.5, max: max + 0.5 };
  const m = (max - min) * 0.08;
  return { min: min - m, max: max + m };
}

function createTicks(min: number, max: number, count: number): number[] {
  if (count < 2) return [min, max];
  const step = (max - min) / (count - 1);
  return Array.from({ length: count }, (_, i) => min + i * step);
}

function xTickRows(xValues: number[], count: number) {
  if (xValues.length <= count) return xValues.map((v, i) => ({ index: i, label: String(v) }));
  const last = xValues.length - 1;
  return Array.from({ length: count }, (_, i) => {
    const idx = Math.round((i / (count - 1)) * last);
    return { index: idx, label: String(xValues[idx]) };
  });
}
