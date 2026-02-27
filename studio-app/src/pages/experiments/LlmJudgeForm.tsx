import { useState } from "react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { useForge } from "../../context/ForgeContext";
import { FormField } from "../../components/shared/FormField";

export function LlmJudgeForm() {
  const { dataRoot } = useForge();
  const command = useForgeCommand();
  const [modelPath, setModelPath] = useState("");
  const [judgeApi, setJudgeApi] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [criteria, setCriteria] = useState("helpfulness,accuracy,safety,reasoning");
  const [testPrompts, setTestPrompts] = useState("");
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState("");

  async function runJudge() {
    if (!dataRoot || !modelPath.trim() || !judgeApi.trim()) return;
    setRunning(true);
    const args = ["judge", "--model-path", modelPath, "--judge-api", judgeApi];
    if (apiKey.trim()) args.push("--api-key", apiKey);
    if (criteria.trim()) args.push("--criteria", criteria);
    if (testPrompts.trim()) args.push("--test-prompts", testPrompts);
    const status = await command.run(dataRoot, args);
    if (status.status === "completed" && command.output) {
      setResults(command.output);
    }
    setRunning(false);
  }

  return (
    <div className="panel stack">
      <h3>LLM-as-Judge Evaluation</h3>
      <div className="grid-2">
        <FormField label="Model Path">
          <input value={modelPath} onChange={(e) => setModelPath(e.currentTarget.value)} placeholder="/path/to/model.pt" />
        </FormField>
        <FormField label="Judge API Endpoint">
          <input value={judgeApi} onChange={(e) => setJudgeApi(e.currentTarget.value)} placeholder="https://api.openai.com/v1/chat/completions" />
        </FormField>
        <FormField label="API Key">
          <input type="password" value={apiKey} onChange={(e) => setApiKey(e.currentTarget.value)} placeholder="sk-..." />
        </FormField>
        <FormField label="Criteria (comma-separated)">
          <input value={criteria} onChange={(e) => setCriteria(e.currentTarget.value)} />
        </FormField>
        <FormField label="Test Prompts File (optional)">
          <input value={testPrompts} onChange={(e) => setTestPrompts(e.currentTarget.value)} placeholder="/path/to/prompts.jsonl" />
        </FormField>
      </div>
      <button className="btn btn-primary" onClick={() => runJudge().catch(console.error)} disabled={running || !modelPath.trim() || !judgeApi.trim()}>
        {running ? "Evaluating..." : "Run Judge"}
      </button>
      {results && <pre className="console">{results}</pre>}
    </div>
  );
}
