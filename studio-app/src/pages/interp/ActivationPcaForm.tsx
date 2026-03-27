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
import { activationPcaLabel } from "../../utils/jobLabels";

interface ActivationPcaFormProps {
  prefill?: Record<string, unknown>;
}

export function ActivationPcaForm({ prefill }: ActivationPcaFormProps) {
  const { dataRoot } = useCrucible();
  const navigate = useNavigate();
  const [modelPath, setModelPath] = useState(
    typeof prefill?.modelPath === "string" ? prefill.modelPath : "",
  );
  const [dataset, setDataset] = useState(
    typeof prefill?.dataset === "string" ? prefill.dataset : "",
  );
  const [layerIndex, setLayerIndex] = useState(
    typeof prefill?.layerIndex === "string" ? prefill.layerIndex : "-1",
  );
  const [granularity, setGranularity] = useState(
    typeof prefill?.granularity === "string" ? prefill.granularity : "sample",
  );
  const [colorField, setColorField] = useState(
    typeof prefill?.colorField === "string" ? prefill.colorField : "",
  );
  const [maxSamples, setMaxSamples] = useState(
    typeof prefill?.maxSamples === "string" ? prefill.maxSamples : "500",
  );
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { isRemote, clusterName, clusterBackend } = useInterpLocation(modelPath);
  const isSlurm = clusterBackend === "slurm";

  const missing = useMemo(() => {
    const m: string[] = [];
    if (!modelPath.trim()) m.push("model");
    if (!dataset.trim()) m.push("dataset");
    if (isRemote && !clusterName) m.push("cluster");
    return m;
  }, [modelPath, dataset, isRemote, clusterName]);

  function snapshotConfig(): Record<string, unknown> {
    return {
      page: "interpretability",
      tab: "activation-pca",
      modelPath,
      dataset,
      layerIndex,
      granularity,
      colorField,
      maxSamples,
    };
  }

  async function submit() {
    if (!dataRoot || missing.length > 0) return;
    setSubmitting(true);
    setError(null);
    try {
      const cfg = snapshotConfig();
      if (isRemote && clusterName) {
        const methodArgs: Record<string, unknown> = {
          model_path: modelPath,
          dataset_name: dataset,
          output_dir: "./outputs/interp",
          layer_index: parseInt(layerIndex || "-1", 10),
          max_samples: parseInt(maxSamples || "500", 10),
          granularity,
        };
        if (colorField.trim()) methodArgs.color_field = colorField;
        const args = isSlurm
          ? buildRemoteInterpArgs(clusterName, "activation-pca", JSON.stringify(methodArgs))
          : buildDispatchSpec("activation-pca", methodArgs, clusterBackend as "ssh", {
              label: activationPcaLabel(modelPath),
              clusterName,
            });
        await startCrucibleCommand(dataRoot, args, activationPcaLabel(modelPath), cfg);
      } else {
        const args = [
          "activation-pca",
          "--model-path", modelPath,
          "--dataset", dataset,
          "--output-dir", "./outputs/interp",
          "--layer-index", layerIndex || "-1",
          "--max-samples", maxSamples || "500",
          "--granularity", granularity,
        ];
        if (colorField.trim()) args.push("--color-field", colorField);
        await startCrucibleCommand(dataRoot, args, activationPcaLabel(modelPath), cfg);
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
      title="Activation PCA"
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
        <FormField label="Layer Index" hint="-1 = last layer">
          <input
            type="number"
            value={layerIndex}
            onChange={(e) => setLayerIndex(e.currentTarget.value)}
          />
        </FormField>
        <FormField label="Max Samples">
          <input
            type="number"
            min={1}
            value={maxSamples}
            onChange={(e) => setMaxSamples(e.currentTarget.value)}
          />
        </FormField>
        <FormField label="Granularity">
          <select value={granularity} onChange={(e) => setGranularity(e.currentTarget.value)}>
            <option value="sample">Per Sample</option>
            <option value="token">Per Token</option>
          </select>
        </FormField>
        <FormField label="Color Field" hint="metadata field">
          <input
            value={colorField}
            onChange={(e) => setColorField(e.currentTarget.value)}
            placeholder="e.g. label, category"
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
