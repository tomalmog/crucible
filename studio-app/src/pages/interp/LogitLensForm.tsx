import { useState, useMemo } from "react";
import { useNavigate } from "react-router";
import { useCrucible } from "../../context/CrucibleContext";
import {
  CompactField,
  CompactFormCard,
  CompactInfoBanner,
  CompactInlineField,
} from "../../components/shared/CompactForm";
import { ModelSelect } from "../../components/shared/ModelSelect";
import { startCrucibleCommand } from "../../api/studioApi";
import { buildDispatchSpec } from "../../api/commandArgs";
import { useInterpLocation } from "../../hooks/useInterpLocation";
import { logitLensLabel } from "../../utils/jobLabels";

interface LogitLensFormProps {
  prefill?: Record<string, unknown>;
}

export function LogitLensForm({ prefill }: LogitLensFormProps) {
  const { dataRoot } = useCrucible();
  const navigate = useNavigate();
  const [modelPath, setModelPath] = useState(
    typeof prefill?.modelPath === "string" ? prefill.modelPath : "",
  );
  const [inputText, setInputText] = useState(
    typeof prefill?.inputText === "string" ? prefill.inputText : "",
  );
  const [topK, setTopK] = useState(
    typeof prefill?.topK === "string" ? prefill.topK : "5",
  );
  const [layerIndices, setLayerIndices] = useState(
    typeof prefill?.layerIndices === "string" ? prefill.layerIndices : "",
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
    if (!inputText.trim()) m.push("input text");
    if (isRemote && !clusterName) m.push("cluster");
    return m;
  }, [modelPath, inputText, isRemote, clusterName]);

  function snapshotConfig(): Record<string, unknown> {
    return {
      page: "interpretability",
      tab: "logit-lens",
      modelPath,
      inputText,
      topK,
      layerIndices,
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
          input_text: inputText,
          output_dir: "./outputs/interp",
          top_k: parseInt(topK || "5", 10),
          layer_indices: layerIndices.trim(),
        };
        if (baseModel.trim()) methodArgs.base_model = baseModel;
        const args = buildDispatchSpec("logit-lens", methodArgs, clusterBackend as "slurm", {
          label: logitLensLabel(modelPath),
          clusterName,
          config: cfg,
        });
        await startCrucibleCommand(dataRoot, args, logitLensLabel(modelPath), cfg);
      } else {
        const args = [
          "logit-lens",
          "--model-path", modelPath,
          "--input-text", inputText,
          "--output-dir", "./outputs/interp",
          "--top-k", topK || "5",
        ];
        if (layerIndices.trim()) args.push("--layer-indices", layerIndices);
        if (baseModel.trim()) args.push("--base-model", baseModel);
        await startCrucibleCommand(dataRoot, args, logitLensLabel(modelPath), cfg);
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
      title="Logit Lens"
      description="Project hidden states through the unembedding and inspect decoded predictions by layer."
      missing={missing}
      actionLabel={isRemote ? "Submit to Cluster" : "Run Logit Lens"}
      runningLabel="Submitting..."
      isRunning={submitting}
      onSubmit={() => submit().catch(console.error)}
      error={error}
    >
      <CompactField label="Model" required>
        <ModelSelect value={modelPath} onChange={setModelPath} />
      </CompactField>
      <CompactField label="Input text" required>
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.currentTarget.value)}
          placeholder="Enter text to analyze..."
          rows={3}
        />
      </CompactField>
      <div className="platform-form-grid platform-form-grid-3">
        <CompactInlineField label="Top K">
          <input
            type="number"
            min={1}
            value={topK}
            onChange={(e) => setTopK(e.currentTarget.value)}
          />
        </CompactInlineField>
        <CompactInlineField hint="empty = all" label="Layers">
          <input
            value={layerIndices}
            onChange={(e) => setLayerIndices(e.currentTarget.value)}
            placeholder="0, 3, 5, 11"
          />
        </CompactInlineField>
        <CompactInlineField hint="adapters only" label="Base model">
          <input
            value={baseModel}
            onChange={(e) => setBaseModel(e.currentTarget.value)}
            placeholder="optional"
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
