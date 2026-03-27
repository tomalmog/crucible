import { useState, useMemo } from "react";
import { useNavigate } from "react-router";
import { useCrucible } from "../../context/CrucibleContext";
import { CommandFormPanel } from "../../components/shared/CommandFormPanel";
import { FormField } from "../../components/shared/FormField";
import { ModelSelect } from "../../components/shared/ModelSelect";
import { startCrucibleCommand } from "../../api/studioApi";
import { jobLabel } from "../../utils/jobLabels";

function defaultExportDir(path: string, format: string): string {
  if (!path) return "./outputs/export";
  const base = path.split("/").pop()?.replace(/\.(pt|pth|bin|safetensors)$/, "") ?? "model";
  return `./outputs/export/${base}-${format}`;
}

export function SafeTensorsExportForm() {
  const { dataRoot } = useCrucible();
  const navigate = useNavigate();
  const [modelPath, setModelPath] = useState("");
  const [outputDir, setOutputDir] = useState("./outputs/export");

  function onModelChange(path: string) {
    setModelPath(path);
    setOutputDir(defaultExportDir(path, "safetensors"));
  }
  const [submitting, setSubmitting] = useState(false);
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
      tab: "safetensors-export",
      modelPath,
      outputDir,
    };
  }

  async function submit() {
    if (!dataRoot || missing.length > 0) return;
    setSubmitting(true);
    setError(null);
    try {
      const args = [
        "safetensors-export",
        "--model-path", modelPath,
        "--output-dir", outputDir,
      ];
      const label = jobLabel("safetensors-export", modelPath);
      const cfg = snapshotConfig();
      await startCrucibleCommand(dataRoot, args, label, cfg);
      navigate("/jobs", { state: { statusFilter: "running" } });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="page-body">
      <CommandFormPanel
        title="SafeTensors Export"
        missing={missing}
        isRunning={submitting}
        submitLabel="Export Model"
        runningLabel="Exporting..."
        onSubmit={() => submit().catch(console.error)}
        error={error}
      >
        <FormField label="Model" required>
          <ModelSelect value={modelPath} onChange={onModelChange} />
        </FormField>
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
