import { useState, useMemo } from "react";
import { useNavigate } from "react-router";
import { useCrucible } from "../../context/CrucibleContext";
import { CommandFormPanel } from "../../components/shared/CommandFormPanel";
import { FormField } from "../../components/shared/FormField";
import { ModelSelect } from "../../components/shared/ModelSelect";
import { startCrucibleCommand } from "../../api/studioApi";
import { jobLabel } from "../../utils/jobLabels";

const QUANT_OPTIONS = ["F32", "F16", "Q8_0", "Q4_0", "Q4_K_M", "Q5_K_M"] as const;

function defaultExportDir(path: string, format: string): string {
  if (!path) return "./outputs/export";
  const base = path.split("/").pop()?.replace(/\.(pt|pth|bin|safetensors)$/, "") ?? "model";
  return `./outputs/export/${base}-${format}`;
}

export function GgufExportForm() {
  const { dataRoot } = useCrucible();
  const navigate = useNavigate();
  const [modelPath, setModelPath] = useState("");
  const [outputDir, setOutputDir] = useState("./outputs/export");
  const [quantType, setQuantType] = useState("F16");
  const [submitting, setSubmitting] = useState(false);

  function onModelChange(path: string) {
    setModelPath(path);
    setOutputDir(defaultExportDir(path, "gguf"));
  }
  const [error, setError] = useState<string | null>(null);

  const missing = useMemo(() => {
    const m: string[] = [];
    if (!modelPath.trim()) m.push("model");
    if (!outputDir.trim()) m.push("output directory");
    return m;
  }, [modelPath, outputDir]);

  function snapshotConfig(): Record<string, unknown> {
    return {
      page: "export",
      tab: "gguf-export",
      modelPath,
      outputDir,
      quantType,
    };
  }

  async function submit() {
    if (!dataRoot || missing.length > 0) return;
    setSubmitting(true);
    setError(null);
    try {
      const args = [
        "gguf-export",
        "--model-path", modelPath,
        "--output-dir", outputDir,
        "--quant-type", quantType,
      ];
      const label = jobLabel("gguf-export", modelPath);
      const cfg = snapshotConfig();
      await startCrucibleCommand(dataRoot, args, label, cfg);
      navigate("/jobs");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="page-body">
      <CommandFormPanel
        title="GGUF Export"
        missing={missing}
        isRunning={submitting}
        submitLabel="Export Model"
        runningLabel="Exporting..."
        onSubmit={() => submit().catch(console.error)}
        error={error}
      >
        <div className="grid-2">
          <FormField label="Model" required>
            <ModelSelect value={modelPath} onChange={onModelChange} />
          </FormField>
          <FormField label="Quantization Type">
            <select value={quantType} onChange={(e) => setQuantType(e.currentTarget.value)}>
              {QUANT_OPTIONS.map((q) => (
                <option key={q} value={q}>{q}</option>
              ))}
            </select>
          </FormField>
        </div>
        <FormField label="Output Directory">
          <input
            value={outputDir}
            onChange={(e) => setOutputDir(e.currentTarget.value)}
            placeholder="./outputs/export"
          />
        </FormField>
      </CommandFormPanel>
    </div>
  );
}
