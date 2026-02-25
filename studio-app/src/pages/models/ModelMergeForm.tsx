import { useState } from "react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { useForge } from "../../context/ForgeContext";
import { FormField } from "../../components/shared/FormField";

export function ModelMergeForm() {
  const { dataRoot } = useForge();
  const command = useForgeCommand();
  const [model1, setModel1] = useState("");
  const [model2, setModel2] = useState("");
  const [method, setMethod] = useState("average");
  const [weight, setWeight] = useState("0.5");
  const [outputPath, setOutputPath] = useState("./merged_model.pt");
  const [merging, setMerging] = useState(false);
  const [result, setResult] = useState("");

  async function startMerge() {
    if (!dataRoot || !model1.trim() || !model2.trim()) return;
    setMerging(true);
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
    setMerging(false);
  }

  return (
    <div className="panel stack-md">
      <h3>Model Merging</h3>
      <div className="grid-2">
        <FormField label="Model A Path">
          <input value={model1} onChange={(e) => setModel1(e.currentTarget.value)} placeholder="/path/to/model_a.pt" />
        </FormField>
        <FormField label="Model B Path">
          <input value={model2} onChange={(e) => setModel2(e.currentTarget.value)} placeholder="/path/to/model_b.pt" />
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
          <input value={outputPath} onChange={(e) => setOutputPath(e.currentTarget.value)} />
        </FormField>
      </div>
      <button className="btn btn-primary" onClick={() => startMerge().catch(console.error)} disabled={merging || !model1.trim() || !model2.trim()}>
        {merging ? "Merging..." : "Merge Models"}
      </button>
      {result && <pre className="console">{result}</pre>}
    </div>
  );
}
