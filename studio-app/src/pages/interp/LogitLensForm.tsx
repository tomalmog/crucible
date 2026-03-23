import { useState, useMemo } from "react";
import { useNavigate } from "react-router";
import { useCrucible } from "../../context/CrucibleContext";
import { CommandFormPanel } from "../../components/shared/CommandFormPanel";
import { FormField } from "../../components/shared/FormField";
import { ModelSelect } from "../../components/shared/ModelSelect";
import { startCrucibleCommand } from "../../api/studioApi";
import { buildRemoteInterpArgs } from "../../api/commandArgs";
import { useInterpLocation } from "../../hooks/useInterpLocation";
import { logitLensLabel } from "../../utils/jobLabels";

export function LogitLensForm() {
  const { dataRoot } = useCrucible();
  const navigate = useNavigate();
  const [modelPath, setModelPath] = useState("");
  const [inputText, setInputText] = useState("");
  const [topK, setTopK] = useState("5");
  const [layerIndices, setLayerIndices] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { isRemote, clusterName } = useInterpLocation(modelPath);

  const missing = useMemo(() => {
    const m: string[] = [];
    if (!modelPath.trim()) m.push("model");
    if (!inputText.trim()) m.push("input text");
    if (isRemote && !clusterName) m.push("cluster");
    return m;
  }, [modelPath, inputText, isRemote, clusterName]);

  async function submit() {
    if (!dataRoot || missing.length > 0) return;
    setSubmitting(true);
    setError(null);
    try {
      if (isRemote && clusterName) {
        const methodArgs = {
          model_path: modelPath,
          input_text: inputText,
          output_dir: "./outputs/interp",
          top_k: parseInt(topK || "5", 10),
          layer_indices: layerIndices.trim(),
        };
        const args = buildRemoteInterpArgs(
          clusterName, "logit-lens", JSON.stringify(methodArgs),
        );
        await startCrucibleCommand(dataRoot, args, logitLensLabel(modelPath));
      } else {
        const args = [
          "logit-lens",
          "--model-path", modelPath,
          "--input-text", inputText,
          "--output-dir", "./outputs/interp",
          "--top-k", topK || "5",
        ];
        if (layerIndices.trim()) args.push("--layer-indices", layerIndices);
        await startCrucibleCommand(dataRoot, args, logitLensLabel(modelPath));
      }
      navigate("/jobs");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <CommandFormPanel
      title="Logit Lens"
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
        <FormField label="Top K">
          <input
            type="number"
            min={1}
            value={topK}
            onChange={(e) => setTopK(e.currentTarget.value)}
          />
        </FormField>
      </div>
      {isRemote && (
        <div className="info-banner">
          Remote model selected — job will run on cluster <strong>{clusterName}</strong>
        </div>
      )}
      <FormField label="Input Text" required>
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.currentTarget.value)}
          placeholder="Enter text to analyze..."
          rows={3}
        />
      </FormField>
      <FormField label="Layer Indices" hint="comma-separated, leave empty for all">
        <input
          value={layerIndices}
          onChange={(e) => setLayerIndices(e.currentTarget.value)}
          placeholder="e.g. 0,3,5,11"
        />
      </FormField>
    </CommandFormPanel>
  );
}
