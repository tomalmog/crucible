import { useState, useMemo } from "react";
import { useNavigate } from "react-router";
import { useCrucible } from "../../context/CrucibleContext";
import {
  CompactFormCard,
  CompactInfoBanner,
  CompactInlineField,
} from "../../components/shared/CompactForm";
import { ModelSelect } from "../../components/shared/ModelSelect";
import { DatasetSelect } from "../../components/shared/DatasetSelect";
import { startCrucibleCommand } from "../../api/studioApi";
import { buildDispatchSpec } from "../../api/commandArgs";
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
  const [baseModel, setBaseModel] = useState(
    typeof prefill?.baseModel === "string" ? prefill.baseModel : "",
  );
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { isRemote, clusterName, clusterBackend } = useInterpLocation(modelPath);

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
      baseModel,
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
        if (baseModel.trim()) methodArgs.base_model = baseModel;
        const args = buildDispatchSpec("activation-pca", methodArgs, clusterBackend as "slurm", {
          label: activationPcaLabel(modelPath),
          clusterName,
          config: cfg,
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
        if (baseModel.trim()) args.push("--base-model", baseModel);
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
    <CompactFormCard
      title="Activation PCA"
      description="Sample activations, project them into a low-dimensional map, and keep the tuning controls tucked into one row."
      missing={missing}
      actionLabel={isRemote ? "Submit to Cluster" : "Run PCA"}
      runningLabel="Submitting..."
      isRunning={submitting}
      onSubmit={() => submit().catch(console.error)}
      error={error}
    >
      <div className="platform-form-grid platform-form-grid-3">
        <CompactInlineField label="Model" required>
          <ModelSelect value={modelPath} onChange={setModelPath} />
        </CompactInlineField>
        <CompactInlineField label="Dataset" required>
          <DatasetSelect value={dataset} onChange={setDataset} />
        </CompactInlineField>
        <CompactInlineField hint="for LoRA and QLoRA" label="Base model">
          <input
            value={baseModel}
            onChange={(e) => setBaseModel(e.currentTarget.value)}
            placeholder="optional"
          />
        </CompactInlineField>
      </div>
      <div className="platform-form-grid platform-form-grid-4">
        <CompactInlineField hint="-1 = last" label="Layer">
          <input
            type="number"
            value={layerIndex}
            onChange={(e) => setLayerIndex(e.currentTarget.value)}
          />
        </CompactInlineField>
        <CompactInlineField label="Samples">
          <input
            type="number"
            min={1}
            value={maxSamples}
            onChange={(e) => setMaxSamples(e.currentTarget.value)}
          />
        </CompactInlineField>
        <CompactInlineField label="Granularity">
          <select value={granularity} onChange={(e) => setGranularity(e.currentTarget.value)}>
            <option value="sample">Per Sample</option>
            <option value="token">Per Token</option>
          </select>
        </CompactInlineField>
        <CompactInlineField hint="metadata" label="Color field">
          <input
            value={colorField}
            onChange={(e) => setColorField(e.currentTarget.value)}
            placeholder="label"
          />
        </CompactInlineField>
      </div>
      {isRemote && (
        <CompactInfoBanner>
          Remote model selected: job will run on cluster <strong>{clusterName}</strong>.
        </CompactInfoBanner>
      )}
    </CompactFormCard>
  );
}
