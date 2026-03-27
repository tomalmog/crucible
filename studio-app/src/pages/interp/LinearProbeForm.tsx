import { useState, useMemo } from "react";
import { useNavigate } from "react-router";
import { useCrucible } from "../../context/CrucibleContext";
import { CommandFormPanel } from "../../components/shared/CommandFormPanel";
import { FormField } from "../../components/shared/FormField";
import { ModelSelect } from "../../components/shared/ModelSelect";
import { DatasetSelect } from "../../components/shared/DatasetSelect";
import { startCrucibleCommand } from "../../api/studioApi";
import { buildRemoteInterpArgs, buildDispatchSpec } from "../../api/commandArgs";
import { useInterpLocation } from "../../hooks/useInterpLocation";
import { jobLabel } from "../../utils/jobLabels";

interface LinearProbeFormProps {
  prefill?: Record<string, unknown>;
}

export function LinearProbeForm({ prefill }: LinearProbeFormProps) {
  const { dataRoot } = useCrucible();
  const navigate = useNavigate();
  const [modelPath, setModelPath] = useState(
    typeof prefill?.modelPath === "string" ? prefill.modelPath : "",
  );
  const [dataset, setDataset] = useState(
    typeof prefill?.dataset === "string" ? prefill.dataset : "",
  );
  const [labelField, setLabelField] = useState(
    typeof prefill?.labelField === "string" ? prefill.labelField : "",
  );
  const [layerIndex, setLayerIndex] = useState(
    typeof prefill?.layerIndex === "string" ? prefill.layerIndex : "-1",
  );
  const [maxSamples, setMaxSamples] = useState(
    typeof prefill?.maxSamples === "string" ? prefill.maxSamples : "500",
  );
  const [epochs, setEpochs] = useState(
    typeof prefill?.epochs === "string" ? prefill.epochs : "10",
  );
  const [learningRate, setLearningRate] = useState(
    typeof prefill?.learningRate === "string" ? prefill.learningRate : "0.001",
  );
  const [baseModel, setBaseModel] = useState(
    typeof prefill?.baseModel === "string" ? prefill.baseModel : "",
  );
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { isRemote, clusterName, clusterBackend } = useInterpLocation(modelPath);
  const isSlurm = clusterBackend === "slurm";

  const missing = useMemo(() => {
    const m: string[] = [];
    if (!modelPath.trim()) m.push("model");
    if (!dataset.trim()) m.push("dataset");
    if (!labelField.trim()) m.push("label field");
    if (isRemote && !clusterName) m.push("cluster");
    return m;
  }, [modelPath, dataset, labelField, isRemote, clusterName]);

  function snapshotConfig(): Record<string, unknown> {
    return {
      page: "interpretability",
      tab: "linear-probe",
      modelPath, dataset, labelField, layerIndex, maxSamples, epochs, learningRate, baseModel,
    };
  }

  async function submit() {
    if (!dataRoot || missing.length > 0) return;
    setSubmitting(true);
    setError(null);
    try {
      const cfg = snapshotConfig();
      const label = jobLabel("linear-probe", modelPath);
      if (isRemote && clusterName) {
        const methodArgs: Record<string, unknown> = {
          model_path: modelPath,
          dataset_name: dataset,
          label_field: labelField,
          output_dir: "./outputs/interp",
          layer_index: parseInt(layerIndex || "-1", 10),
          max_samples: parseInt(maxSamples || "500", 10),
          epochs: parseInt(epochs || "10", 10),
          learning_rate: parseFloat(learningRate || "0.001"),
        };
        if (baseModel.trim()) methodArgs.base_model = baseModel;
        const args = isSlurm
          ? buildRemoteInterpArgs(clusterName, "linear-probe", JSON.stringify(methodArgs))
          : buildDispatchSpec("linear-probe", methodArgs, clusterBackend as "ssh", {
              label,
              clusterName,
            });
        await startCrucibleCommand(dataRoot, args, label, cfg);
      } else {
        const args = [
          "linear-probe",
          "--model-path", modelPath,
          "--dataset", dataset,
          "--label-field", labelField,
          "--output-dir", "./outputs/interp",
          "--layer-index", layerIndex || "-1",
          "--max-samples", maxSamples || "500",
          "--epochs", epochs || "10",
          "--learning-rate", learningRate || "0.001",
        ];
        if (baseModel.trim()) args.push("--base-model", baseModel);
        await startCrucibleCommand(dataRoot, args, label, cfg);
      }
      navigate("/jobs", { state: { statusFilter: "running" } });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <CommandFormPanel
      title="Linear Probe"
      missing={missing}
      isRunning={submitting}
      submitLabel={isRemote ? "Submit to Cluster" : "Run Analysis"}
      runningLabel="Submitting..."
      onSubmit={() => submit().catch(console.error)}
      error={error}
    >
      <div className="grid-2">
        <FormField label="Model" required>
          <ModelSelect value={modelPath} onChange={setModelPath} />
        </FormField>
        <FormField label="Dataset" required>
          <DatasetSelect value={dataset} onChange={setDataset} />
        </FormField>
        <FormField label="Label Field" required hint="metadata field for classification">
          <input
            value={labelField}
            onChange={(e) => setLabelField(e.currentTarget.value)}
            placeholder="e.g. label, category, sentiment"
          />
        </FormField>
        <FormField label="Layer Index" hint="-1=last, -2=all layers">
          <input
            type="number"
            value={layerIndex}
            onChange={(e) => setLayerIndex(e.currentTarget.value)}
          />
        </FormField>
        <FormField label="Max Samples">
          <input type="number" min={1} value={maxSamples} onChange={(e) => setMaxSamples(e.currentTarget.value)} />
        </FormField>
        <FormField label="Epochs">
          <input type="number" min={1} value={epochs} onChange={(e) => setEpochs(e.currentTarget.value)} />
        </FormField>
        <FormField label="Learning Rate">
          <input type="number" step="0.0001" value={learningRate} onChange={(e) => setLearningRate(e.currentTarget.value)} />
        </FormField>
        <FormField label="Base Model" hint="for LoRA/QLoRA models">
          <input
            value={baseModel}
            onChange={(e) => setBaseModel(e.currentTarget.value)}
            placeholder="optional — HuggingFace ID or path"
          />
        </FormField>
      </div>
      {isRemote && (
        <div className="info-banner">
          Remote model selected — job will run on cluster <strong>{clusterName}</strong>
        </div>
      )}
    </CommandFormPanel>
  );
}
