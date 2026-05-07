import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { AlertTriangle, FlaskConical, Gauge, Server, ShieldCheck } from "lucide-react";
import { getDatasetDashboard } from "../../api/studioApi";
import { useCrucible } from "../../context/CrucibleContext";
import type { DatasetDashboard } from "../../types";
import type { SharedTrainingConfig, TrainingMethod } from "../../types/training";

interface TrainingPreflightPanelProps {
  method: TrainingMethod;
  methodName: string;
  shared: SharedTrainingConfig;
  extra: Record<string, string>;
  modelName: string;
  projectName: string;
  evalObjective: string;
  remoteEnabled: boolean;
}

interface PreflightEstimate {
  recordCountLabel: string;
  tokenLabel: string;
  gpuLabel: string;
  durationLabel: string;
  costLabel: string;
  riskLabel: string;
  riskTone: "success" | "warning" | "neutral";
}

export function TrainingPreflightPanel({
  method,
  methodName,
  shared,
  extra,
  modelName,
  projectName,
  evalObjective,
  remoteEnabled,
}: TrainingPreflightPanelProps): ReactNode {
  const { dataRoot } = useCrucible();
  const datasetName = (extra["--dataset"] ?? "").trim();
  const [dashboard, setDashboard] = useState<DatasetDashboard | null>(null);
  const [isLoadingDataset, setIsLoadingDataset] = useState(false);

  // Reacts to dataset changes so the launch preflight reflects the selected dataset.
  useEffect(() => {
    setDashboard(null);
    if (!dataRoot || !datasetName) return;
    let cancelled = false;
    setIsLoadingDataset(true);
    getDatasetDashboard(dataRoot, datasetName)
      .then((row) => {
        if (!cancelled) setDashboard(row);
      })
      .catch(() => {
        if (!cancelled) setDashboard(null);
      })
      .finally(() => {
        if (!cancelled) setIsLoadingDataset(false);
      });
    return () => {
      cancelled = true;
    };
  }, [dataRoot, datasetName]);

  const estimate = useMemo(
    () => buildPreflightEstimate(method, shared, dashboard, remoteEnabled),
    [method, shared, dashboard, remoteEnabled],
  );

  return (
    <section className="preflight-panel">
      <div className="preflight-header">
        <div>
          <h3>Launch preflight</h3>
          <p>{projectName.trim() || "Default Project"} · {modelName.trim() || methodName}</p>
        </div>
        <span className="badge badge-accent">{methodName}</span>
      </div>

      <div className="preflight-grid">
        <PreflightMetric
          icon={<Gauge size={14} />}
          label="Training data"
          value={isLoadingDataset ? "Loading..." : estimate.recordCountLabel}
          detail={estimate.tokenLabel}
        />
        <PreflightMetric
          icon={<Server size={14} />}
          label="Compute"
          value={estimate.gpuLabel}
          detail={`${estimate.durationLabel} · ${estimate.costLabel}`}
        />
        <PreflightMetric
          icon={<FlaskConical size={14} />}
          label="Eval gate"
          value={evalObjective.trim() ? "Required after run" : "Needs success metric"}
          detail={evalObjective.trim() || "Define what must improve before promotion."}
        />
        <PreflightMetric
          icon={estimate.riskTone === "warning" ? <AlertTriangle size={14} /> : <ShieldCheck size={14} />}
          label="Ship risk"
          value={estimate.riskLabel}
          detail={healthDetail(estimate.riskTone)}
          tone={estimate.riskTone}
        />
      </div>
    </section>
  );
}

function PreflightMetric({
  icon,
  label,
  value,
  detail,
  tone = "neutral",
}: {
  icon: ReactNode;
  label: string;
  value: string;
  detail: string;
  tone?: "success" | "warning" | "neutral";
}): ReactNode {
  return (
    <div className={`preflight-metric preflight-metric-${tone}`}>
      <span className="preflight-metric-icon">{icon}</span>
      <span className="metric-label">{label}</span>
      <strong>{value}</strong>
      <span>{detail}</span>
    </div>
  );
}

function buildPreflightEstimate(
  method: TrainingMethod,
  shared: SharedTrainingConfig,
  dashboard: DatasetDashboard | null,
  remoteEnabled: boolean,
): PreflightEstimate {
  const records = dashboard?.record_count ?? 0;
  const avgTokens = Math.max(1, Math.round(dashboard?.avg_token_length ?? parseInt(shared.maxTokenLength, 10) * 0.35));
  const epochs = Math.max(1, parseInt(shared.epochs, 10) || 1);
  const totalTokens = records > 0 ? records * avgTokens * epochs : 0;
  const hours = estimateHours(totalTokens, method);
  const risk = riskFromDataset(dashboard);
  return {
    recordCountLabel: records > 0 ? `${records.toLocaleString()} examples` : "Select dataset",
    tokenLabel: totalTokens > 0 ? `${formatCount(totalTokens)} train tokens` : "Token estimate appears after dataset selection",
    gpuLabel: gpuRecommendation(method),
    durationLabel: hours > 0 ? durationRange(hours) : "Duration pending",
    costLabel: hours > 0 ? costRange(hours, remoteEnabled) : "Cost pending",
    riskLabel: risk.label,
    riskTone: risk.tone,
  };
}

function estimateHours(totalTokens: number, method: TrainingMethod): number {
  if (totalTokens <= 0) return 0;
  const multiplier = method === "dpo-train" ? 1.7
    : method === "qlora-train" ? 1.35
      : method === "domain-adapt" ? 1.45
        : method === "lora-train" ? 0.85
          : 1.0;
  return Math.max(0.08, (totalTokens / 8_000_000) * multiplier);
}

function durationRange(hours: number): string {
  const lowMinutes = Math.max(5, Math.round(hours * 45));
  const highMinutes = Math.max(lowMinutes + 10, Math.round(hours * 90));
  if (highMinutes < 90) return `${lowMinutes}-${highMinutes} min`;
  return `${Math.round(lowMinutes / 60)}-${Math.round(highMinutes / 60)} hr`;
}

function costRange(hours: number, remoteEnabled: boolean): string {
  if (!remoteEnabled) return "Local/BYOC";
  const low = Math.max(1, Math.round(hours * 2.5));
  const high = Math.max(low + 3, Math.round(hours * 8));
  return `$${low}-$${high}`;
}

function gpuRecommendation(method: TrainingMethod): string {
  if (method === "qlora-train") return "1x 24GB+ GPU";
  if (method === "dpo-train") return "1x A100/L40S";
  if (method === "domain-adapt") return "1x A100 class";
  if (method === "train") return "A100/H100 preferred";
  return "1x L4/A10/L40S";
}

function riskFromDataset(
  dashboard: DatasetDashboard | null,
): { label: string; tone: "success" | "warning" | "neutral" } {
  if (!dashboard) return { label: "Waiting on dataset", tone: "neutral" };
  if (dashboard.record_count < 50) return { label: "Too little data", tone: "warning" };
  if (dashboard.average_quality < 0.45) return { label: "Low data quality", tone: "warning" };
  if (dashboard.record_count < 500) return { label: "Small validation set", tone: "warning" };
  return { label: "Ready for eval-gated run", tone: "success" };
}

function healthDetail(tone: "success" | "warning" | "neutral"): string {
  if (tone === "success") return "Run will still need before/after eval before promotion.";
  if (tone === "warning") return "Review dataset size, quality, or validation coverage.";
  return "Select data and define the eval gate before launch.";
}

function formatCount(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${Math.round(value / 1_000)}K`;
  return value.toLocaleString();
}
