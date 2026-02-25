import { useState } from "react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { useForge } from "../../context/ForgeContext";
import { FormField } from "../../components/shared/FormField";

export function SyntheticDataForm() {
  const { dataRoot } = useForge();
  const command = useForgeCommand();
  const [modelPath, setModelPath] = useState("");
  const [seedPrompts, setSeedPrompts] = useState("");
  const [count, setCount] = useState("1000");
  const [minQuality, setMinQuality] = useState("0.5");
  const [output, setOutput] = useState("./synthetic_data.jsonl");
  const [generating, setGenerating] = useState(false);
  const [result, setResult] = useState("");

  async function generate() {
    if (!dataRoot || !seedPrompts.trim()) return;
    setGenerating(true);
    const args = ["synthetic", "--seed-prompts", seedPrompts, "--count", count, "--min-quality", minQuality, "--output", output];
    if (modelPath.trim()) args.push("--model-path", modelPath);
    const status = await command.run(dataRoot, args);
    if (status.status === "completed" && command.output) setResult(command.output);
    setGenerating(false);
  }

  return (
    <div className="panel stack-md">
      <h3>Synthetic Data Generation</h3>
      <div className="grid-2">
        <FormField label="Seed Prompts File">
          <input value={seedPrompts} onChange={(e) => setSeedPrompts(e.currentTarget.value)} placeholder="/path/to/seed_prompts.jsonl" />
        </FormField>
        <FormField label="Model Path (optional)">
          <input value={modelPath} onChange={(e) => setModelPath(e.currentTarget.value)} placeholder="/path/to/model.pt" />
        </FormField>
        <FormField label="Count">
          <input value={count} onChange={(e) => setCount(e.currentTarget.value)} />
        </FormField>
        <FormField label="Min Quality">
          <input value={minQuality} onChange={(e) => setMinQuality(e.currentTarget.value)} />
        </FormField>
        <FormField label="Output Path">
          <input value={output} onChange={(e) => setOutput(e.currentTarget.value)} />
        </FormField>
      </div>
      <button className="btn btn-primary" onClick={() => generate().catch(console.error)} disabled={generating || !seedPrompts.trim()}>
        {generating ? "Generating..." : "Generate Data"}
      </button>
      {result && <pre className="console">{result}</pre>}
    </div>
  );
}
