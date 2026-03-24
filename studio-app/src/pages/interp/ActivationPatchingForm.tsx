import { useState, useMemo } from "react";
import { useNavigate } from "react-router";
import { useCrucible } from "../../context/CrucibleContext";
import { CommandFormPanel } from "../../components/shared/CommandFormPanel";
import { FormField } from "../../components/shared/FormField";
import { ModelSelect } from "../../components/shared/ModelSelect";
import { startCrucibleCommand } from "../../api/studioApi";
import { buildRemoteInterpArgs } from "../../api/commandArgs";
import { useInterpLocation } from "../../hooks/useInterpLocation";
import { activationPatchingLabel } from "../../utils/jobLabels";

interface ActivationPatchingFormProps {
  prefill?: Record<string, unknown>;
}

export function ActivationPatchingForm({ prefill }: ActivationPatchingFormProps) {
  const { dataRoot } = useCrucible();
  const navigate = useNavigate();
  const [modelPath, setModelPath] = useState(
    typeof prefill?.modelPath === "string" ? prefill.modelPath : "",
  );
  const [cleanText, setCleanText] = useState(
    typeof prefill?.cleanText === "string" ? prefill.cleanText : "",
  );
  const [corruptedText, setCorruptedText] = useState(
    typeof prefill?.corruptedText === "string" ? prefill.corruptedText : "",
  );
  const [metric, setMetric] = useState(
    typeof prefill?.metric === "string" ? prefill.metric : "logit_diff",
  );
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { isRemote, clusterName } = useInterpLocation(modelPath);

  const missing = useMemo(() => {
    const m: string[] = [];
    if (!modelPath.trim()) m.push("model");
    if (!cleanText.trim()) m.push("clean text");
    if (!corruptedText.trim()) m.push("corrupted text");
    if (isRemote && !clusterName) m.push("cluster");
    return m;
  }, [modelPath, cleanText, corruptedText, isRemote, clusterName]);

  function snapshotConfig(): Record<string, unknown> {
    return {
      page: "interpretability",
      tab: "activation-patching",
      modelPath,
      cleanText,
      corruptedText,
      metric,
    };
  }

  async function submit() {
    if (!dataRoot || missing.length > 0) return;
    setSubmitting(true);
    setError(null);
    try {
      const cfg = snapshotConfig();
      if (isRemote && clusterName) {
        const methodArgs = {
          model_path: modelPath,
          clean_text: cleanText,
          corrupted_text: corruptedText,
          output_dir: "./outputs/interp",
          metric,
        };
        const args = buildRemoteInterpArgs(
          clusterName, "activation-patch", JSON.stringify(methodArgs),
        );
        await startCrucibleCommand(dataRoot, args, activationPatchingLabel(modelPath), cfg);
      } else {
        const args = [
          "activation-patch",
          "--model-path", modelPath,
          "--clean-text", cleanText,
          "--corrupted-text", corruptedText,
          "--output-dir", "./outputs/interp",
          "--metric", metric,
        ];
        await startCrucibleCommand(dataRoot, args, activationPatchingLabel(modelPath), cfg);
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
      title="Activation Patching"
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
        <FormField label="Metric">
          <select value={metric} onChange={(e) => setMetric(e.currentTarget.value)}>
            <option value="logit_diff">Logit Difference</option>
            <option value="prob">Max Probability</option>
          </select>
        </FormField>
      </div>
      {isRemote && (
        <div className="info-banner">
          Remote model selected — job will run on cluster <strong>{clusterName}</strong>
        </div>
      )}
      <FormField label="Clean Text" required>
        <textarea
          value={cleanText}
          onChange={(e) => setCleanText(e.currentTarget.value)}
          placeholder="The capital of France is"
          rows={2}
        />
      </FormField>
      <FormField label="Corrupted Text" required>
        <textarea
          value={corruptedText}
          onChange={(e) => setCorruptedText(e.currentTarget.value)}
          placeholder="The capital of Germany is"
          rows={2}
        />
      </FormField>
    </CommandFormPanel>
  );
}
