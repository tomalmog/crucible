import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router";
import { TrainingMethod, TRAINING_METHODS, REQUIRED_METHOD_FIELDS } from "../../types/training";
import { useTrainingConfig } from "../../hooks/useTrainingConfig";
import { useTrainingLocation } from "../../hooks/useTrainingLocation";
import {
  buildTrainingArgs, buildMethodArgs, buildDispatchSpec,
} from "../../api/commandArgs";
import { startCrucibleCommand } from "../../api/studioApi";
import { trainingLabel } from "../../utils/jobLabels";
import { generateScript, parseScriptConfig, configToFormState } from "../../utils/scriptGenerator";
import { CodeEditor } from "../../components/shared/CodeEditor";
import { TabBar } from "../../components/shared/TabBar";
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
import { ClusterSubmitSection, DEFAULT_CLUSTER_CONFIG } from "./ClusterSubmitSection";
import type { ClusterSubmitConfig } from "./ClusterSubmitSection";
import { TrainingClusterContext } from "../../context/TrainingClusterContext";
import type { TrainingClusterContextValue } from "../../context/TrainingClusterContext";
import { FormField } from "../../components/shared/FormField";
import { PageHeader } from "../../components/shared/PageHeader";
import { useScript } from "../../context/ScriptContext";
import { ArrowLeft, RotateCcw } from "lucide-react";

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

interface TrainingWizardProps {
  method: TrainingMethod;
  dataRoot: string;
  onBack: () => void;
  prefill?: Record<string, unknown>;
}

export function TrainingWizard({ method, dataRoot, onBack, prefill }: TrainingWizardProps) {
  const methodInfo = TRAINING_METHODS.find((m) => m.id === method)!;
  const navigate = useNavigate();
  const [modelName, setModelName] = useState(
    typeof prefill?.modelName === "string" ? prefill.modelName : "My-Model-0",
  );
  const [startError, setStartError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  type ViewTab = "configure" | "code";
  const [viewTab, setViewTab] = useState<ViewTab>("configure");
  const [scriptContent, setScriptContent] = useState("");
  const [remoteEnabled, setRemoteEnabled] = useState(
    typeof prefill?.remoteEnabled === "boolean" ? prefill.remoteEnabled : false,
  );
  const [clusterConfig, setClusterConfig] = useState<ClusterSubmitConfig>(
    prefill?.clusterConfig
      ? { ...DEFAULT_CLUSTER_CONFIG, ...(prefill.clusterConfig as Partial<ClusterSubmitConfig>) }
      : DEFAULT_CLUSTER_CONFIG,
  );
  const config = useTrainingConfig(method, dataRoot);
  const { shared, setShared, extra, setExtra } = config;
  useTrainingLocation(method, extra, setRemoteEnabled, setClusterConfig);

  // Expose script state to AI agent via ScriptContext
  const { register: registerScript, unregister: unregisterScript } = useScript();
  const scriptContentRef = useRef(scriptContent);
  scriptContentRef.current = scriptContent;
  const viewTabRef = useRef(viewTab);
  viewTabRef.current = viewTab;
  useEffect(() => {
    registerScript({
      contentRef: scriptContentRef,
      setContent: setScriptContentTracked,
      viewTabRef,
      method,
    });
    return () => unregisterScript();
  }, [method]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!prefill || !config.isLoaded) return;
    if (prefill.shared && typeof prefill.shared === "object") {
      setShared({ ...shared, ...(prefill.shared as Partial<typeof shared>) });
    }
    if (prefill.extra && typeof prefill.extra === "object") {
      setExtra({ ...extra, ...(prefill.extra as Record<string, string>) });
    }
  }, [config.isLoaded]); // eslint-disable-line react-hooks/exhaustive-deps

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

  function snapshotConfig(): Record<string, unknown> {
    return {
      page: "training",
      method,
      modelName,
      shared: { ...shared },
      extra: { ...extra },
      remoteEnabled,
      clusterConfig: { ...clusterConfig },
    };
  }

  // Track whether the agent has pushed a script update that should be
  // preserved when switching from Configure → Code tab.
  const agentScriptRef = useRef<string | null>(null);
  const originalSetScriptContent = setScriptContent;
  // Wrap setContent so we can detect agent-driven updates
  const setScriptContentTracked = useCallback((content: string) => {
    agentScriptRef.current = content;
    originalSetScriptContent(content);
  }, [originalSetScriptContent]);

  function handleTabSwitch(tab: ViewTab) {
    if (tab === "code" && viewTab === "configure") {
      // Preserve agent edits: if the agent set a script while on Configure tab,
      // don't overwrite it with a fresh generate
      if (agentScriptRef.current) {
        setScriptContent(agentScriptRef.current);
        agentScriptRef.current = null;
      } else {
        setScriptContent(generateScript({ method, shared, extra, modelName }));
      }
    } else if (tab === "configure" && viewTab === "code") {
      // Parse script config back into form state
      const parsed = parseScriptConfig(scriptContent);
      const { shared: newShared, extra: newExtra } = configToFormState(parsed, method);
      setShared({ ...shared, ...newShared });
      setExtra({ ...extra, ...newExtra });
      if (parsed.model_name) setModelName(parsed.model_name);
    }
    setViewTab(tab);
  }

  const missing = useMemo(() => {
    const m = getMissingFields(method, extra);
    if (!modelName.trim()) m.push("model name");
    if (remoteEnabled && !clusterConfig.cluster) m.push("cluster");
    return m;
  }, [method, extra, modelName, remoteEnabled, clusterConfig.cluster]);
  const canStart = missing.length === 0;

  async function submitScript() {
    setSubmitting(true);
    setStartError(null);
    try {
      // Parse model_name from the script content (user may have edited it)
      const parsed = parseScriptConfig(scriptContent);
      const scriptModelName = parsed.model_name || modelName.trim() || method;
      const cfg = snapshotConfig();
      const { writeTextFile } = await import("../../api/studioApi");
      const timestamp = Date.now();
      const scriptPath = `${dataRoot}/scripts/_training_${timestamp}.py`;
      await writeTextFile(scriptPath, scriptContent);
      const args = ["run-script", scriptPath, "--model-name", scriptModelName];
      await startCrucibleCommand(dataRoot, args, trainingLabel(method, scriptModelName), cfg);
      navigate("/jobs", { state: { statusFilter: "running" } });
    } catch (err) {
      setStartError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  async function startTraining() {
    if (viewTab === "code") return submitScript();
    if (!canStart) return;
    setSubmitting(true);
    setStartError(null);
    try {
      const cfg = snapshotConfig();
      // Derive a unique output directory from the model name so each
      // training run saves to its own folder instead of overwriting.
      const safeName = modelName.trim().replace(/[^a-zA-Z0-9_-]/g, "_") || "model";
      const uniqueShared = { ...shared, outputDir: `./outputs/${safeName}` };
      if (remoteEnabled) {
        const methodArgsObj = buildMethodArgs(uniqueShared, extra);
        let extraOverrides: Record<string, unknown> = {};
        try {
          extraOverrides = JSON.parse(clusterConfig.extraMethodArgs);
        } catch { /* ignore invalid JSON */ }
        const merged = { ...methodArgsObj, ...extraOverrides };
        const args = buildDispatchSpec(method, merged, clusterConfig.backend, {
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
          config: cfg,
        });
        await startCrucibleCommand(dataRoot, args, trainingLabel(method, modelName), cfg);
      } else {
        const args = [...buildTrainingArgs(method, uniqueShared, extra), "--model-name", modelName.trim()];
        await startCrucibleCommand(dataRoot, args, trainingLabel(method, modelName), cfg);
      }
      navigate("/jobs", { state: { statusFilter: "running" } });
    } catch (err) {
      setStartError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  if (!config.isLoaded) return null;

  return (
    <>
      <PageHeader title={methodInfo.name} />
      <button className="detail-back" onClick={onBack}>
        <ArrowLeft size={14} /> Back to Training
      </button>

      <TabBar
        tabs={["configure", "code"] as const}
        active={viewTab}
        onChange={handleTabSwitch}
        format={(t) => t === "configure" ? "Configure" : "Code"}
      />

      <TrainingClusterContext.Provider value={clusterContextValue}>
      {viewTab === "code" ? (
        <div className="panel stack-lg">
          <CodeEditor value={scriptContent} onChange={setScriptContent} maxHeight="600px" />
          {startError && <div className="error-alert">{startError}</div>}
          <div className="flex-row">
            <button
              className="btn btn-primary btn-lg"
              onClick={() => startTraining().catch(console.error)}
              disabled={submitting}
            >
              {submitting ? "Running..." : "Run Script"}
            </button>
          </div>
        </div>
      ) : (
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
            <button
              className="btn btn-primary btn-lg"
              onClick={() => startTraining().catch(console.error)}
              disabled={!canStart || submitting}
            >
              {submitting ? "Submitting..." : remoteEnabled ? "Submit to Cluster" : "Start Training"}
            </button>
            <button
              className="btn btn-ghost btn-sm"
              onClick={config.resetToDefaults}
              title="Reset to defaults"
            >
              <RotateCcw size={12} /> Reset
            </button>
          </div>
        </div>
      )}

        <ClusterSubmitSection
          enabled={remoteEnabled}
          onToggle={setRemoteEnabled}
          clusterConfig={clusterConfig}
          onChange={setClusterConfig}
        />
      </TrainingClusterContext.Provider>
    </>
  );
}
