import { useCallback, useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { FormField } from "../../components/shared/FormField";
import { FormSection } from "../../components/shared/FormSection";
import { listClusters } from "../../api/remoteApi";
import { useCrucible } from "../../context/CrucibleContext";
import type { ClusterConfig } from "../../types/remote";

export type ClusterMode = "toggle" | "auto";

export interface ClusterSubmitConfig {
  cluster: string;
  partition: string;
  nodes: string;
  gpusPerNode: string;
  gpuType: string;
  cpusPerTask: string;
  memory: string;
  timeLimit: string;
  remoteDataset: string;
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
  remoteDataset: "",
  pullModel: false,
  extraMethodArgs: "{}",
};

interface ClusterSubmitSectionProps {
  mode: ClusterMode;
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
  clusterConfig: ClusterSubmitConfig;
  onChange: (c: ClusterSubmitConfig) => void;
}

export function ClusterSubmitSection({
  mode,
  enabled,
  onToggle,
  clusterConfig,
  onChange,
}: ClusterSubmitSectionProps) {
  const { dataRoot } = useCrucible();
  const [clusters, setClusters] = useState<ClusterConfig[]>([]);
  const [isLoadingClusters, setIsLoadingClusters] = useState(false);

  const loadClusters = useCallback(() => {
    if (!dataRoot) return;
    setIsLoadingClusters(true);
    listClusters(dataRoot)
      .then(setClusters)
      .catch(() => setClusters([]))
      .finally(() => setIsLoadingClusters(false));
  }, [dataRoot]);

  // Load clusters when section becomes active
  useEffect(() => {
    if (mode === "auto" || enabled) loadClusters();
  }, [mode, enabled, loadClusters]);

  const selectedCluster = clusters.find((c) => c.name === clusterConfig.cluster);

  function set<K extends keyof ClusterSubmitConfig>(key: K, value: ClusterSubmitConfig[K]) {
    onChange({ ...clusterConfig, [key]: value });
  }

  const showFields = mode === "toggle" ? enabled : enabled;

  return (
    <div className="form-section-divider">
      {mode === "toggle" && (
        <label className="flex-row">
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => onToggle(e.target.checked)}
            style={{ width: "auto" }}
          />
          <span style={{ fontWeight: 500 }}>Submit to remote cluster</span>
        </label>
      )}

      {mode === "auto" && !enabled && (
        <div className="text-tertiary" style={{ fontSize: "0.8rem" }}>
          Select a remote model above to train on a cluster.
        </div>
      )}

      {showFields && (
        <div className="stack-md" style={{ marginTop: 8 }}>
            <FormField label="Cluster" required>
              <div style={{ position: "relative" }}>
                <select
                  className="input"
                  value={clusterConfig.cluster}
                  onChange={(e) => set("cluster", e.currentTarget.value)}
                  disabled={isLoadingClusters}
                >
                  <option value="">
                    {isLoadingClusters ? "Loading clusters..." : "Select a cluster..."}
                  </option>
                  {clusters.map((c) => (
                    <option key={c.name} value={c.name}>
                      {c.name} ({c.user}@{c.host})
                    </option>
                  ))}
                </select>
                {isLoadingClusters && (
                  <Loader2
                    size={14}
                    className="spin"
                    style={{ position: "absolute", right: 28, top: "50%", transform: "translateY(-50%)" }}
                  />
                )}
              </div>
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
