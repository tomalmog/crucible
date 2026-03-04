import { useCallback, useEffect, useState } from "react";
import { FormField } from "../../components/shared/FormField";
import { FormSection } from "../../components/shared/FormSection";
import { listClusters } from "../../api/remoteApi";
import { useForge } from "../../context/ForgeContext";
import type { ClusterConfig } from "../../types/remote";

export interface ClusterSubmitConfig {
  cluster: string;
  partition: string;
  nodes: string;
  gpusPerNode: string;
  gpuType: string;
  cpusPerTask: string;
  memory: string;
  timeLimit: string;
  dataStrategy: "scp" | "shared" | "s3";
  pullModel: boolean;
  extraMethodArgs: string;
}

export const DEFAULT_CLUSTER_CONFIG: ClusterSubmitConfig = {
  cluster: "",
  partition: "",
  nodes: "1",
  gpusPerNode: "1",
  gpuType: "",
  cpusPerTask: "4",
  memory: "32G",
  timeLimit: "12:00:00",
  dataStrategy: "shared",
  pullModel: false,
  extraMethodArgs: "{}",
};

interface ClusterSubmitSectionProps {
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
  clusterConfig: ClusterSubmitConfig;
  onChange: (c: ClusterSubmitConfig) => void;
}

export function ClusterSubmitSection({
  enabled,
  onToggle,
  clusterConfig,
  onChange,
}: ClusterSubmitSectionProps) {
  const { dataRoot } = useForge();
  const [clusters, setClusters] = useState<ClusterConfig[]>([]);

  const loadClusters = useCallback(() => {
    if (!dataRoot) return;
    listClusters(dataRoot).then(setClusters).catch(() => setClusters([]));
  }, [dataRoot]);

  useEffect(() => {
    if (enabled) loadClusters();
  }, [enabled, loadClusters]);

  const selectedCluster = clusters.find((c) => c.name === clusterConfig.cluster);

  function set<K extends keyof ClusterSubmitConfig>(key: K, value: ClusterSubmitConfig[K]) {
    onChange({ ...clusterConfig, [key]: value });
  }

  return (
    <div className="form-section-divider">
      <label className="flex-row">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => onToggle(e.target.checked)}
          style={{ width: "auto" }}
        />
        <span style={{ fontWeight: 500 }}>Submit to remote cluster</span>
      </label>

      {enabled && (
        <div className="stack-md" style={{ marginTop: "var(--space-md)" }}>
          <FormField label="Cluster" required>
            <select
              className="input"
              value={clusterConfig.cluster}
              onChange={(e) => set("cluster", e.currentTarget.value)}
            >
              <option value="">Select a cluster...</option>
              {clusters.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name} ({c.user}@{c.host})
                </option>
              ))}
            </select>
          </FormField>

          <div className="form-row">
            <FormField label="Partition">
              <select
                className="input"
                value={clusterConfig.partition}
                onChange={(e) => set("partition", e.currentTarget.value)}
              >
                <option value="">Default</option>
                {selectedCluster?.partitions.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </FormField>

            <FormField label="GPU Type">
              <select
                className="input"
                value={clusterConfig.gpuType}
                onChange={(e) => set("gpuType", e.currentTarget.value)}
              >
                <option value="">Any</option>
                {selectedCluster?.gpuTypes.map((g) => (
                  <option key={g} value={g}>{g}</option>
                ))}
              </select>
            </FormField>
          </div>

          <div className="form-row">
            <FormField label="Nodes">
              <input
                className="input"
                type="number"
                min={1}
                value={clusterConfig.nodes}
                onChange={(e) => set("nodes", e.currentTarget.value)}
              />
            </FormField>

            <FormField label="GPUs/Node">
              <input
                className="input"
                type="number"
                min={1}
                value={clusterConfig.gpusPerNode}
                onChange={(e) => set("gpusPerNode", e.currentTarget.value)}
              />
            </FormField>

            <FormField label="CPUs/Task">
              <input
                className="input"
                type="number"
                min={1}
                value={clusterConfig.cpusPerTask}
                onChange={(e) => set("cpusPerTask", e.currentTarget.value)}
              />
            </FormField>
          </div>

          <div className="form-row">
            <FormField label="Memory">
              <input
                className="input"
                value={clusterConfig.memory}
                onChange={(e) => set("memory", e.currentTarget.value)}
              />
            </FormField>

            <FormField label="Time Limit">
              <input
                className="input"
                value={clusterConfig.timeLimit}
                onChange={(e) => set("timeLimit", e.currentTarget.value)}
                placeholder="HH:MM:SS"
              />
            </FormField>
          </div>

          <FormField label="Data Strategy">
            <select
              className="input"
              value={clusterConfig.dataStrategy}
              onChange={(e) => set("dataStrategy", e.currentTarget.value as "scp" | "shared" | "s3")}
            >
              <option value="shared">Shared (NFS/Lustre)</option>
              <option value="scp">SCP Upload</option>
              <option value="s3">S3 Streaming</option>
            </select>
          </FormField>

          <FormField label="Auto-Pull Model">
            <label className="flex-row">
              <input
                type="checkbox"
                checked={clusterConfig.pullModel}
                onChange={(e) => set("pullModel", e.currentTarget.checked)}
              />
              <span>Download trained model to local after completion</span>
            </label>
          </FormField>

          <FormSection title="Extra Overrides (JSON)">
            <FormField label="Additional key-value pairs" hint="Merged into method args; overrides form values">
              <textarea
                className="input"
                rows={3}
                value={clusterConfig.extraMethodArgs}
                onChange={(e) => set("extraMethodArgs", e.currentTarget.value)}
                placeholder='{"custom_param": "value"}'
              />
            </FormField>
          </FormSection>
        </div>
      )}
    </div>
  );
}
