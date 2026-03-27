import { useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import type { ClusterRemoteStorage } from "../../hooks/useResourceData";
import type { ClusterConfig, ClusterInfo, PartitionInfo } from "../../types/remote";

interface ClusterInfoPanelProps {
  remoteStorage: ClusterRemoteStorage[];
  clusters: ClusterConfig[];
  loading: boolean;
}

export function ClusterInfoPanel({ remoteStorage, clusters, loading }: ClusterInfoPanelProps) {
  const withInfo = remoteStorage.filter((rs) => rs.clusterInfo !== null);
  const [selected, setSelected] = useState(withInfo[0]?.cluster.name ?? "");

  // No clusters configured at all — hide the card entirely
  if (clusters.length === 0 && !loading) return null;

  // Clusters exist but info hasn't arrived yet — show loading skeleton
  if (withInfo.length === 0) {
    return (
      <div className="resource-card">
        <div className="resource-card-header">
          <h3 className="resource-card-title">Cluster</h3>
        </div>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", padding: 24, gap: 8, color: "var(--text-tertiary)" }}>
          <Loader2 size={16} className="spin" />
          <span style={{ fontSize: "0.8125rem" }}>Loading cluster info…</span>
        </div>
      </div>
    );
  }

  const current = withInfo.find((rs) => rs.cluster.name === selected) ?? withInfo[0];
  const info: ClusterInfo = current.clusterInfo!;

  return (
    <div className="resource-card">
      <div className="resource-card-header">
        <h3 className="resource-card-title">Cluster</h3>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <ClusterSelect
            clusters={withInfo}
            value={current.cluster.name}
            onChange={setSelected}
          />
          <span className={`badge ${info.isConnected ? "badge-success" : "badge-error"}`}>
            {info.isConnected ? "Connected" : "Disconnected"}
          </span>
        </div>
      </div>

      {info.isConnected && (
        <>
          <GpuAvailability info={info} />
          <NodeHealth info={info} />
          <PartitionTable partitions={info.partitions} />
        </>
      )}
    </div>
  );
}

interface ClusterSelectProps {
  clusters: ClusterRemoteStorage[];
  value: string;
  onChange: (name: string) => void;
}

function ClusterSelect({ clusters, value, onChange }: ClusterSelectProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const blurTimeout = useRef<ReturnType<typeof setTimeout>>(undefined);

  const filtered = clusters.filter((rs) =>
    rs.cluster.name.toLowerCase().includes(query.toLowerCase()),
  );

  function handleFocus(): void {
    clearTimeout(blurTimeout.current);
    setQuery("");
    setOpen(true);
  }

  function handleBlur(): void {
    blurTimeout.current = setTimeout(() => setOpen(false), 150);
  }

  function pick(name: string): void {
    onChange(name);
    setQuery("");
    setOpen(false);
  }

  return (
    <div className="dataset-select" onFocus={handleFocus} onBlur={handleBlur}>
      <input
        value={open ? query : value}
        onChange={(e) => setQuery(e.currentTarget.value)}
        placeholder="Search clusters…"
        readOnly={!open}
        style={{ minWidth: 120, fontSize: "0.75rem" }}
      />
      {open && filtered.length > 0 && (
        <ul className="dataset-select-dropdown">
          {filtered.map((rs) => (
            <li key={rs.cluster.name}>
              <button
                type="button"
                className="dataset-select-option"
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => pick(rs.cluster.name)}
              >
                {rs.cluster.name}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function GpuAvailability({ info }: { info: ClusterInfo }) {
  return (
    <div className="stats-grid">
      <div className="metric-card">
        <span className="metric-label">Total GPUs</span>
        <span className="metric-value">{info.totalGpus}</span>
      </div>
      <div className="metric-card">
        <span className="metric-label">Idle GPUs</span>
        <span className="metric-value">{info.idleGpus}</span>
      </div>
      <div className="metric-card">
        <span className="metric-label">Utilization</span>
        <span className="metric-value">{info.gpuUtilizationPct}%</span>
      </div>
    </div>
  );
}

function NodeHealth({ info }: { info: ClusterInfo }) {
  return (
    <div className="resource-section">
      <div className="resource-section-header">
        <span className="resource-section-title">Node Health</span>
      </div>
      <div className="stats-grid">
        <div className="metric-card">
          <span className="metric-label">Healthy</span>
          <span className="metric-value">{info.healthyNodes}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Drained</span>
          <span className="metric-value">{info.drainedNodes}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Down</span>
          <span className="metric-value">{info.downNodes}</span>
        </div>
      </div>
    </div>
  );
}

function parseGpuConfig(gres: string): { type: string; count: number }[] {
  const gpus: { type: string; count: number }[] = [];
  for (const part of gres.split(",")) {
    const match = part.match(/gpu:([^:]+):(\d+)/);
    if (match) gpus.push({ type: match[1], count: parseInt(match[2], 10) });
  }
  return gpus;
}

function formatGpuType(raw: string): string {
  return raw.replace(/gpu$/, "").replace(/([a-z])([A-Z])/g, "$1 $2");
}

function PartitionTable({ partitions }: { partitions: PartitionInfo[] }) {
  if (partitions.length === 0) return null;
  const hasGpus = partitions.some((p) => p.gpuConfig);

  return (
    <div className="resource-section">
      <div className="resource-section-header">
        <span className="resource-section-title">Partitions</span>
      </div>
      <div className="docs-table-wrap">
        <table className="docs-table">
          <thead>
            <tr>
              <th>Partition</th>
              <th>Nodes</th>
              <th>GPUs (idle / total)</th>
              {hasGpus && <th>GPU Types</th>}
              {hasGpus && <th>Memory</th>}
              {hasGpus && <th>CPUs</th>}
            </tr>
          </thead>
          <tbody>
            {partitions.map((p) => {
              const gpus = p.gpuConfig ? parseGpuConfig(p.gpuConfig) : [];
              const gpuSummary = gpus.length > 0
                ? gpus.map((g) => `${g.count}× ${formatGpuType(g.type)}`).join(", ")
                : "";
              return (
                <tr key={p.name}>
                  <td style={{ fontWeight: 500 }}>{p.name}{p.isDefault ? " *" : ""}</td>
                  <td>{p.totalNodes}</td>
                  <td>{p.totalGpus > 0 ? `${p.idleGpus} / ${p.totalGpus}` : "—"}</td>
                  {hasGpus && <td>{gpuSummary || "—"}</td>}
                  {hasGpus && <td>{p.gpuConfig ? formatMemory(p.memoryMb) : "—"}</td>}
                  {hasGpus && <td>{p.gpuConfig ? p.cpusPerNode : "—"}</td>}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function formatMemory(mb: number): string {
  if (mb >= 1024) return `${Math.round(mb / 1024)} GB`;
  return `${mb} MB`;
}
