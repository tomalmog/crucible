import { useState } from "react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { FormField } from "../../components/shared/FormField";
import { StatusConsole } from "../../components/shared/StatusConsole";
import { CommandProgress } from "../../components/shared/CommandProgress";

interface QuantizeFormProps {
  dataRoot: string;
}

export function QuantizeForm({ dataRoot }: QuantizeFormProps) {
  const command = useForgeCommand();
  const [modelPath, setModelPath] = useState("");
  const [quantType, setQuantType] = useState("int8");
  const [outputPath, setOutputPath] = useState("");

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    const args = ["deploy", "quantize"];
    if (modelPath.trim()) args.push("--model-path", modelPath.trim());
    if (quantType.trim()) args.push("--quantization-type", quantType.trim());
    if (outputPath.trim()) args.push("--output", outputPath.trim());
    await command.run(dataRoot, args);
  }

  return (
    <div className="panel">
      <h3 className="panel-title">Quantize Model</h3>
      <form onSubmit={(e) => onSubmit(e).catch(console.error)} className="stack">
        <div className="grid-3">
          <FormField label="Model Path">
            <input value={modelPath} onChange={(e) => setModelPath(e.currentTarget.value)} placeholder="/path/to/model.pt" />
          </FormField>
          <FormField label="Quantization Type">
            <select value={quantType} onChange={(e) => setQuantType(e.currentTarget.value)}>
              <option value="int8">INT8</option>
              <option value="int4">INT4</option>
              <option value="fp16">FP16</option>
            </select>
          </FormField>
          <FormField label="Output Path">
            <input value={outputPath} onChange={(e) => setOutputPath(e.currentTarget.value)} placeholder="./quantized" />
          </FormField>
        </div>
        <button className="btn btn-primary btn-lg" type="submit" disabled={command.isRunning}>
          {command.isRunning ? "Quantizing..." : "Quantize"}
        </button>
      </form>
      {command.isRunning && command.status && (
        <div className="gap-top">
          <CommandProgress label="Quantizing..." percent={command.status.progress_percent} />
        </div>
      )}
      {command.output && <div className="gap-top"><StatusConsole output={command.output} /></div>}
    </div>
  );
}
