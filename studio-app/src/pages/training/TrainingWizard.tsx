import { useCallback, useMemo, useState } from "react";
import { useNavigate } from "react-router";
import { TrainingMethod, TRAINING_METHODS, REQUIRED_METHOD_FIELDS } from "../../types/training";
import { useCrucibleCommand } from "../../hooks/useCrucibleCommand";
import { useCrucible } from "../../context/CrucibleContext";
import { useTrainingConfig } from "../../hooks/useTrainingConfig";
import { useTrainingLocation } from "../../hooks/useTrainingLocation";
import {
  buildTrainingArgs, buildMethodArgs, buildDispatchSpec,
} from "../../api/commandArgs";
import { startCrucibleCommand } from "../../api/studioApi";
import { SharedTrainingFields } from "./forms/SharedTrainingFields";
import { BasicTrainForm } from "./forms/BasicTrainForm";
import { SftTrainForm } from "./forms/SftTrainForm";
import { DpoTrainForm } from "./forms/DpoTrainForm";
import { RlhfTrainForm } from "./forms/RlhfTrainForm";
import { LoraTrainForm } from "./forms/LoraTrainForm";
import { DistillTrainForm } from "./forms/DistillTrainForm";
import { DomainAdaptForm } from "./forms/DomainAdaptForm";
import { GrpoTrainForm } from "./forms/GrpoTrainForm";
import { QloraTrainForm } from "./forms/QloraTrainForm";
import { KtoTrainForm } from "./forms/KtoTrainForm";
import { OrpoTrainForm } from "./forms/OrpoTrainForm";
import { MultimodalTrainForm } from "./forms/MultimodalTrainForm";
import { RlvrTrainForm } from "./forms/RlvrTrainForm";
import { TrainingRunMonitor } from "./TrainingRunMonitor";
import { ClusterSubmitSection, DEFAULT_CLUSTER_CONFIG } from "./ClusterSubmitSection";
import type { ClusterSubmitConfig } from "./ClusterSubmitSection";
import { TrainingClusterContext } from "../../context/TrainingClusterContext";
import type { TrainingClusterContextValue } from "../../context/TrainingClusterContext";
import { FormField } from "../../components/shared/FormField";
import { PageHeader } from "../../components/shared/PageHeader";
import { ArrowLeft, RotateCcw, Check } from "lucide-react";

function parseModelPath(stdout: string): string | null {
  for (const line of stdout.split("\n")) {
    if (line.startsWith("model_path=")) return line.slice("model_path=".length).trim();
  }
  return null;
}

function getMissingFields(
  method: TrainingMethod,
  extra: Record<string, string>,
): string[] {
  const missing: string[] = [];
  for (const flag of REQUIRED_METHOD_FIELDS[method]) {
    if (!(extra[flag] ?? "").trim()) {
      const label = flag.replace(/^--/, "").replace(/-/g, " ");
      missing.push(label);
    }
  }
  return missing;
}

type Step = "config" | "running" | "done";

interface TrainingWizardProps {
  method: TrainingMethod;
  dataRoot: string;
  onBack: () => void;
}

export function TrainingWizard({ method, dataRoot, onBack }: TrainingWizardProps) {
  const methodInfo = TRAINING_METHODS.find((m) => m.id === method)!;
  const { refreshModels } = useCrucible();
  const navigate = useNavigate();
  const command = useCrucibleCommand();
  const registerCommand = useCrucibleCommand();
  const [step, setStep] = useState<Step>("config");
  const [modelName, setModelName] = useState("My-Model-0");
  const [registered, setRegistered] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const [remoteEnabled, setRemoteEnabled] = useState(false);
  const [clusterConfig, setClusterConfig] = useState<ClusterSubmitConfig>(DEFAULT_CLUSTER_CONFIG);
  const [remoteSubmitting, setRemoteSubmitting] = useState(false);
  const config = useTrainingConfig(method, dataRoot);
  const { shared, setShared, extra, setExtra } = config;
  useTrainingLocation(method, extra, setRemoteEnabled, setClusterConfig);

  // Toggle remote on/off based on dataset selection (local vs remote)
  const handleDatasetLocationChanged = useCallback((isRemote: boolean, cluster: string) => {
    setRemoteEnabled(isRemote);
    if (isRemote) {
      setClusterConfig((prev) => ({ ...prev, cluster }));
    }
  }, []);

  const clusterContextValue = useMemo<TrainingClusterContextValue>(() => ({
    cluster: remoteEnabled ? clusterConfig.cluster : "",
    onDatasetLocationChanged: handleDatasetLocationChanged,
  }), [remoteEnabled, clusterConfig.cluster, handleDatasetLocationChanged]);

  const missing = useMemo(() => {
    const m = getMissingFields(method, extra);
    if (!modelName.trim()) m.push("model name");
    if (remoteEnabled && !clusterConfig.cluster) m.push("cluster");
    return m;
  }, [method, extra, modelName, remoteEnabled, clusterConfig.cluster]);
  const canStart = missing.length === 0;

  async function startTraining() {
    if (!canStart) return;
    setStep("running");
    setRegistered(false);
    try {
      const args = buildTrainingArgs(method, shared, extra);
      const status = await command.run(dataRoot, args);
      setStep("done");
      if (status.status === "completed" && status.exit_code === 0) {
        const modelPath = parseModelPath(status.stdout);
        if (modelPath && modelName.trim()) {
          const regStatus = await registerCommand.run(dataRoot, [
            "model", "register", "--name", modelName.trim(), "--model-path", modelPath,
          ]);
          if (regStatus.status === "completed" && regStatus.exit_code === 0) {
            setRegistered(true);
            refreshModels().catch(console.error);
          }
        }
      }
    } catch (err) {
      setStartError(err instanceof Error ? err.message : String(err));
      setStep("done");
    }
  }

  async function submitToCluster() {
    if (!canStart) return;
    setRemoteSubmitting(true);
    setStartError(null);
    try {
      const methodArgsObj = buildMethodArgs(shared, extra);
      let extraOverrides: Record<string, unknown> = {};
      try {
        extraOverrides = JSON.parse(clusterConfig.extraMethodArgs);
      } catch { /* ignore invalid JSON */ }
      const merged = { ...methodArgsObj, ...extraOverrides };
      const args = buildDispatchSpec(method, merged, "slurm", {
        label: modelName.trim() || undefined,
        clusterName: clusterConfig.cluster,
        resources: {
          partition: clusterConfig.partition,
          nodes: parseInt(clusterConfig.nodes, 10) || 1,
          gpus_per_node: parseInt(clusterConfig.gpusPerNode, 10) || 1,
          cpus_per_task: parseInt(clusterConfig.cpusPerTask, 10) || 4,
          memory: clusterConfig.memory || "32G",
          time_limit: clusterConfig.timeLimit || "04:00:00",
          gpu_type: clusterConfig.gpuType || "",
        },
      });
      await startCrucibleCommand(dataRoot, args);
      navigate("/jobs");
    } catch (err) {
      setStartError(err instanceof Error ? err.message : String(err));
    } finally {
      setRemoteSubmitting(false);
    }
  }

  if (!config.isLoaded) return null;

  return (
    <>
      <PageHeader title={methodInfo.name} />
      <button className="detail-back" onClick={onBack}>
        <ArrowLeft size={14} /> Back to Training
      </button>

      {step === "config" && (
        <TrainingClusterContext.Provider value={clusterContextValue}>
        <div className="panel stack-lg has-remote-tab">
            {method === "train" && <BasicTrainForm extra={extra} setExtra={setExtra} />}
            {method === "sft" && <SftTrainForm extra={extra} setExtra={setExtra} />}
            {method === "dpo-train" && <DpoTrainForm extra={extra} setExtra={setExtra} />}
            {method === "rlhf-train" && <RlhfTrainForm extra={extra} setExtra={setExtra} />}
            {method === "lora-train" && <LoraTrainForm extra={extra} setExtra={setExtra} />}
            {method === "distill" && <DistillTrainForm extra={extra} setExtra={setExtra} />}
            {method === "domain-adapt" && <DomainAdaptForm extra={extra} setExtra={setExtra} />}
            {method === "grpo-train" && <GrpoTrainForm extra={extra} setExtra={setExtra} />}
            {method === "qlora-train" && <QloraTrainForm extra={extra} setExtra={setExtra} />}
            {method === "kto-train" && <KtoTrainForm extra={extra} setExtra={setExtra} />}
            {method === "orpo-train" && <OrpoTrainForm extra={extra} setExtra={setExtra} />}
            {method === "multimodal-train" && <MultimodalTrainForm extra={extra} setExtra={setExtra} />}
            {method === "rlvr-train" && <RlvrTrainForm extra={extra} setExtra={setExtra} />}

            <SharedTrainingFields config={shared} onChange={setShared} />

            <FormField label="Model Name" required>
              <input
                value={modelName}
                onChange={(e) => setModelName(e.currentTarget.value)}
                placeholder="My-Model-0"
              />
            </FormField>

            {!canStart && (
              <div className="error-alert">
                Missing required fields: {missing.join(", ")}
              </div>
            )}
            {startError && (
              <div className="error-alert">{startError}</div>
            )}
            <div className="flex-row">
              {remoteEnabled ? (
                <button
                  className="btn btn-primary btn-lg"
                  onClick={() => submitToCluster().catch(console.error)}
                  disabled={!canStart || remoteSubmitting}
                >
                  {remoteSubmitting ? "Submitting..." : "Submit to Cluster"}
                </button>
              ) : (
                <button
                  className="btn btn-primary btn-lg"
                  onClick={() => startTraining().catch(console.error)}
                  disabled={!canStart}
                >
                  Start Training
                </button>
              )}
              <button
                className="btn btn-ghost btn-sm"
                onClick={config.resetToDefaults}
                title="Reset to defaults"
              >
                <RotateCcw size={12} /> Reset
              </button>
            </div>
          </div>

          <ClusterSubmitSection
            enabled={remoteEnabled}
            onToggle={setRemoteEnabled}
            clusterConfig={clusterConfig}
            onChange={setClusterConfig}
          />
        </TrainingClusterContext.Provider>
      )}

      {step === "running" && (
        <TrainingRunMonitor command={command} />
      )}

      {step === "done" && (
        <div className="panel stack-lg">
          {command.status?.exit_code === 0 ? (
            <h3 className="text-success">Training Complete</h3>
          ) : (
            <h3 className="error-text">Training Failed</h3>
          )}
          {command.status && (
            <div className="stats-grid">
              <div className="metric-card">
                <span className="metric-label">Status</span>
                <span className={`metric-value ${command.status.exit_code === 0 ? "text-success" : "error-text"}`}>
                  {command.status.status}
                </span>
              </div>
              <div className="metric-card">
                <span className="metric-label">Duration</span>
                <span className="metric-value">{command.status.elapsed_seconds}s</span>
              </div>
            </div>
          )}
          {(command.error || startError) && (
            <div className="error-alert">{command.error || startError}</div>
          )}
          {registered && (
            <div className="flex-row" style={{ color: "var(--success)" }}>
              <Check size={14} />
              <span>Model registered as &ldquo;{modelName}&rdquo;</span>
            </div>
          )}
          {registerCommand.error && (
            <div className="error-alert">
              Failed to register model: {registerCommand.error}
            </div>
          )}
          <pre className="console">{command.output}</pre>
          <div className="row gap-top">
            <button className="btn" onClick={onBack}>New Training</button>
          </div>
        </div>
      )}
    </>
  );
}
