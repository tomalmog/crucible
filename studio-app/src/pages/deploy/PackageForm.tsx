import { useState } from "react";
import { useCrucibleCommand } from "../../hooks/useCrucibleCommand";
import { FormField } from "../../components/shared/FormField";
import { StatusConsole } from "../../components/shared/StatusConsole";
import { CommandProgress } from "../../components/shared/CommandProgress";

interface PackageFormProps {
  dataRoot: string;
}

export function PackageForm({ dataRoot }: PackageFormProps) {
  const command = useCrucibleCommand();
  const [modelPath, setModelPath] = useState("");
  const [outputPath, setOutputPath] = useState("");
  const [format, setFormat] = useState("onnx");

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    const args = ["deploy", "package"];
    if (modelPath.trim()) args.push("--model-path", modelPath.trim());
    if (outputPath.trim()) args.push("--output", outputPath.trim());
    if (format.trim()) args.push("--format", format.trim());
    await command.run(dataRoot, args);
  }

  return (
    <div className="panel">
      <h3 className="panel-title">Package Model</h3>
      <form onSubmit={(e) => onSubmit(e).catch(console.error)} className="stack">
        <div className="grid-3">
          <FormField label="Model Path">
            <input value={modelPath} onChange={(e) => setModelPath(e.currentTarget.value)} placeholder="/path/to/model.pt" />
          </FormField>
          <FormField label="Output Path">
            <input value={outputPath} onChange={(e) => setOutputPath(e.currentTarget.value)} placeholder="./packaged" />
          </FormField>
          <FormField label="Format">
            <select value={format} onChange={(e) => setFormat(e.currentTarget.value)}>
              <option value="onnx">ONNX</option>
              <option value="torchscript">TorchScript</option>
              <option value="safetensors">SafeTensors</option>
            </select>
          </FormField>
        </div>
        <button className="btn btn-primary btn-lg" type="submit" disabled={command.isRunning}>
          {command.isRunning ? "Packaging..." : "Package"}
        </button>
      </form>
      {command.isRunning && command.status && (
        <div className="gap-top">
          <CommandProgress label="Packaging..." percent={command.status.progress_percent} />
        </div>
      )}
      {command.output && <div className="gap-top"><StatusConsole output={command.output} /></div>}
    </div>
  );
}
