import { useState } from "react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { useForge } from "../../context/ForgeContext";
import { FormField } from "../../components/shared/FormField";
import { PathInput } from "../../components/shared/PathInput";

export function ModelMergeForm() {
  const { dataRoot } = useForge();
  const command = useForgeCommand();
  const [model1, setModel1] = useState("");
  const [model2, setModel2] = useState("");
  const [method, setMethod] = useState("average");
  const [weight, setWeight] = useState("0.5");
  const [outputPath, setOutputPath] = useState("./merged_model.pt");
  const [result, setResult] = useState("");

  async function startMerge() {
    if (!dataRoot || !model1.trim() || !model2.trim()) return;
    const w = parseFloat(weight);
    const args = [
      "merge",
      "--models", model1, model2,
      "--method", method,
      "--weights", `${1 - w},${w}`,
      "--output", outputPath,
    ];
    const status = await command.run(dataRoot, args);
    if (status.status === "completed" && command.output) {
      setResult(command.output);
    }
  }

  return (
    <div className="panel stack-md">
      <h3 className="panel-title">Model Merging</h3>
      <div className="grid-2">
        <FormField label="Model A Path">
          <PathInput value={model1} onChange={setModel1} placeholder="/path/to/model_a.pt" />
        </FormField>
        <FormField label="Model B Path">
          <PathInput value={model2} onChange={setModel2} placeholder="/path/to/model_b.pt" />
        </FormField>
        <FormField label="Merge Method">
          <select value={method} onChange={(e) => setMethod(e.currentTarget.value)}>
            <option value="average">Weighted Average</option>
            <option value="slerp">SLERP</option>
            <option value="ties">TIES</option>
            <option value="dare">DARE</option>
          </select>
        </FormField>
        <FormField label="Weight (for Model B)">
          <input value={weight} onChange={(e) => setWeight(e.currentTarget.value)} placeholder="0.5" />
        </FormField>
        <FormField label="Output Path">
          <PathInput value={outputPath} onChange={setOutputPath} placeholder="./merged_model.pt" kind="folder" />
        </FormField>
      </div>
      <button className="btn btn-primary" onClick={() => startMerge().catch(console.error)} disabled={command.isRunning || !model1.trim() || !model2.trim()}>
        {command.isRunning ? "Merging..." : "Merge Models"}
      </button>
      {result && <pre className="console">{result}</pre>}
    </div>
  );
}
