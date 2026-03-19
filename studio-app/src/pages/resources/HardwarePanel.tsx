import { MetricCard } from "../../components/shared/MetricCard";
import type { ClusterConfig } from "../../types/remote";

interface HardwarePanelProps {
  hardware: Record<string, string> | null;
  clusters: ClusterConfig[];
}

const DISPLAY_KEYS: Array<{ key: string; label: string }> = [
  { key: "accelerator", label: "Accelerator" },
  { key: "gpu_count", label: "GPU Count" },
  { key: "recommended_precision_mode", label: "Precision" },
  { key: "gpu_name", label: "GPU" },
  { key: "gpu_memory_mb", label: "GPU Memory" },
  { key: "cpu_count", label: "CPU Count" },
  { key: "ram_gb", label: "RAM (GB)" },
];

export function HardwarePanel({ hardware, clusters }: HardwarePanelProps) {
  const entries = hardware
    ? DISPLAY_KEYS.filter((dk) => hardware[dk.key]).map((dk) => ({
        label: dk.label,
        value: hardware[dk.key],
      }))
    : [];

  return (
    <div className="panel">
      <div className="panel-header">
        <h3>Hardware</h3>
      </div>
      {entries.length > 0 ? (
        <div className="metric-row">
          {entries.map((e) => (
            <MetricCard key={e.label} label={e.label} value={e.value} />
          ))}
        </div>
      ) : (
        <p className="text-sm text-tertiary" style={{ padding: "0 16px 16px" }}>
          No hardware profile available.
        </p>
      )}
      {clusters.length > 0 && (
        <div style={{ padding: "0 16px 16px" }}>
          <h4 className="text-sm" style={{ marginBottom: 8 }}>Remote Clusters</h4>
          {clusters.map((c) => (
            <div key={c.name} className="flex-row" style={{ marginBottom: 4 }}>
              <span className="text-sm">{c.name}</span>
              <span className="text-xs text-tertiary">
                {c.gpuTypes.length > 0 ? c.gpuTypes.join(", ") : "No GPUs detected"}
                {c.partitions.length > 0 && ` | ${c.partitions.join(", ")}`}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
