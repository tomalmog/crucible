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
    <div className="resource-card">
      <div className="resource-card-header">
        <h3 className="resource-card-title">Hardware</h3>
      </div>

      {entries.length > 0 ? (
        <div className="hw-grid">
          {entries.map((e) => (
            <div className="hw-item" key={e.label}>
              <span className="hw-item-label">{e.label}</span>
              <span className="hw-item-value">{e.value}</span>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-tertiary" style={{ margin: 0 }}>
          No hardware profile available.
        </p>
      )}

      {clusters.length > 0 && (
        <div className="resource-section">
          <div className="resource-section-header">
            <span className="resource-section-title">Remote Clusters</span>
          </div>
          {clusters.map((c) => (
            <div key={c.name} className="activity-job">
              <span className="activity-job-name">{c.name}</span>
              <span className="activity-job-meta">
                {c.gpuTypes.length > 0 ? c.gpuTypes.join(", ") : "No GPUs"}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
