import { useCallback, useEffect, useState } from "react";
import { Loader2, PanelRightClose, Server } from "lucide-react";
import { FormField } from "../../components/shared/FormField";
import { FormSection } from "../../components/shared/FormSection";
import { listClusters } from "../../api/remoteApi";
import { useCrucible } from "../../context/CrucibleContext";
import type { ClusterConfig } from "../../types/remote";
import type { BackendKind } from "../../types/jobs";

export interface ClusterSubmitConfig {
  backend: BackendKind;
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
  dockerImage: string;
  gpuDeviceIds: string;
  apiEndpoint: string;
  apiToken: string;
}

export const DEFAULT_CLUSTER_CONFIG: ClusterSubmitConfig = {
  backend: "slurm",
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
  dockerImage: "",
  gpuDeviceIds: "",
  apiEndpoint: "",
  apiToken: "",
};

const BACKEND_OPTIONS: readonly BackendKind[] = ["slurm", "ssh", "http-api"];

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

  // Load clusters on mount
  useEffect(() => {
    loadClusters();
  }, [loadClusters]);

  const selectedCluster = clusters.find((c) => c.name === clusterConfig.cluster);

  // Auto-set backend to match the selected cluster's registered backend
  useEffect(() => {
    if (selectedCluster && selectedCluster.backend !== clusterConfig.backend) {
      onChange({ ...clusterConfig, backend: selectedCluster.backend });
    }
  }, [selectedCluster?.name]); // eslint-disable-line react-hooks/exhaustive-deps

  function set<K extends keyof ClusterSubmitConfig>(key: K, value: ClusterSubmitConfig[K]): void {
    onChange({ ...clusterConfig, [key]: value });
  }

  const isSlurm = clusterConfig.backend === "slurm";
  const isDocker = clusterConfig.backend === "ssh";
  const isHttp = clusterConfig.backend === "http-api";

  return (
    <>
      <button
        className={`remote-sidebar-tab${enabled ? " remote-sidebar-tab-hidden" : ""}`}
        onClick={() => onToggle(true)}
        title="Open remote cluster settings"
      >
        <Server size={14} />
        <span>Remote</span>
      </button>

      <div className={`remote-sidebar${enabled ? " remote-sidebar-open" : ""}`}>
        <div className="remote-sidebar-header">
        <h4>Remote Cluster</h4>
        <button
          className="btn btn-ghost btn-sm btn-icon"
          onClick={() => onToggle(false)}
          title="Close remote panel"
        >
          <PanelRightClose size={16} />
        </button>
      </div>

      <div className="remote-sidebar-body">
        <FormField label="Backend" required>
          <select
            className="input"
            value={clusterConfig.backend}
            onChange={(e) => set("backend", e.currentTarget.value as BackendKind)}
          >
            {BACKEND_OPTIONS.map((b) => (
              <option key={b} value={b}>{b}</option>
            ))}
          </select>
        </FormField>

        <FormField label="Cluster" required>
          <div style={{ position: "relative" }}>
            <select
              className="input"
              value={clusterConfig.cluster}
              onChange={(e) => set("cluster", e.currentTarget.value)}
              disabled={isLoadingClusters}
            >
              <option value="">
                {isLoadingClusters ? "Loading..." : "Select..."}
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

        {isSlurm && (
          <SlurmFields
            clusterConfig={clusterConfig}
            selectedCluster={selectedCluster}
            set={set}
          />
        )}

        {isDocker && (
          <DockerFields clusterConfig={clusterConfig} set={set} />
        )}

        {isHttp && (
          <HttpApiFields clusterConfig={clusterConfig} set={set} />
        )}

        <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "0.8125rem", color: "var(--text-secondary)", cursor: "pointer" }}>
          <input
            type="checkbox"
            checked={clusterConfig.pullModel}
            onChange={(e) => set("pullModel", e.currentTarget.checked)}
            style={{ width: "auto" }}
          />
          Auto-pull model after completion
        </label>

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
    </div>
    </>
  );
}

interface FieldSetterProps {
  clusterConfig: ClusterSubmitConfig;
  set: <K extends keyof ClusterSubmitConfig>(key: K, value: ClusterSubmitConfig[K]) => void;
}

interface SlurmFieldsProps extends FieldSetterProps {
  selectedCluster: ClusterConfig | undefined;
}

function SlurmFields({ clusterConfig, selectedCluster, set }: SlurmFieldsProps) {
  return (
    <>
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

      <div className="form-row">
        <FormField label="Nodes">
          <input className="input" type="number" min={1} value={clusterConfig.nodes} onChange={(e) => set("nodes", e.currentTarget.value)} />
        </FormField>
        <FormField label="GPUs/Node">
          <input className="input" type="number" min={1} value={clusterConfig.gpusPerNode} onChange={(e) => set("gpusPerNode", e.currentTarget.value)} />
        </FormField>
      </div>

      <FormField label="CPUs/Task">
        <input className="input" type="number" min={1} value={clusterConfig.cpusPerTask} onChange={(e) => set("cpusPerTask", e.currentTarget.value)} />
      </FormField>

      <div className="form-row">
        <FormField label="Memory">
          <input className="input" value={clusterConfig.memory} onChange={(e) => set("memory", e.currentTarget.value)} />
        </FormField>
        <FormField label="Time Limit">
          <input className="input" value={clusterConfig.timeLimit} onChange={(e) => set("timeLimit", e.currentTarget.value)} placeholder="HH:MM:SS" />
        </FormField>
      </div>
    </>
  );
}

function DockerFields({ clusterConfig, set }: FieldSetterProps) {
  return (
    <>
      <FormField label="Docker Image" hint="Leave blank for pytorch/pytorch:latest">
        <input
          className="input"
          value={clusterConfig.dockerImage}
          onChange={(e) => set("dockerImage", e.currentTarget.value)}
          placeholder="pytorch/pytorch:latest"
        />
      </FormField>

      <FormField label="GPU Device IDs" hint="Comma-separated (e.g. 0,1). Blank = all GPUs">
        <input
          className="input"
          value={clusterConfig.gpuDeviceIds}
          onChange={(e) => set("gpuDeviceIds", e.currentTarget.value)}
          placeholder="all"
        />
      </FormField>
    </>
  );
}

function HttpApiFields({ clusterConfig, set }: FieldSetterProps) {
  return (
    <>
      <FormField label="API Endpoint" hint="URL of the remote Crucible server">
        <input
          className="input"
          value={clusterConfig.apiEndpoint}
          onChange={(e) => set("apiEndpoint", e.currentTarget.value)}
          placeholder="http://server:8080"
        />
      </FormField>

      <FormField label="API Token" hint="Bearer token for authentication">
        <input
          className="input"
          type="password"
          value={clusterConfig.apiToken}
          onChange={(e) => set("apiToken", e.currentTarget.value)}
          placeholder="Optional"
        />
      </FormField>
    </>
  );
}
