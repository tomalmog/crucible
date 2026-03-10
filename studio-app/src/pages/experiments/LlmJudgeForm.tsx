import { useState, useMemo } from "react";
import { useCrucibleCommand } from "../../hooks/useCrucibleCommand";
import { useCrucible } from "../../context/CrucibleContext";
import { CommandFormPanel } from "../../components/shared/CommandFormPanel";
import { FormField } from "../../components/shared/FormField";
import { PathInput } from "../../components/shared/PathInput";

export function LlmJudgeForm() {
  const { dataRoot } = useCrucible();
  const command = useCrucibleCommand();
  const [modelPath, setModelPath] = useState("");
  const [judgeApi, setJudgeApi] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [criteria, setCriteria] = useState("helpfulness,accuracy,safety,reasoning");
  const [testPrompts, setTestPrompts] = useState("");
  const [results, setResults] = useState("");

  const missing = useMemo(() => {
    const m: string[] = [];
    if (!modelPath.trim()) m.push("model path");
    if (!judgeApi.trim()) m.push("judge API endpoint");
    return m;
  }, [modelPath, judgeApi]);

  async function runJudge() {
    if (!dataRoot) return;
    const args = ["judge", "--model-path", modelPath, "--judge-api", judgeApi];
    if (apiKey.trim()) args.push("--api-key", apiKey);
    if (criteria.trim()) args.push("--criteria", criteria);
    if (testPrompts.trim()) args.push("--test-prompts", testPrompts);
    const status = await command.run(dataRoot, args);
    if (status.status === "completed" && command.output) {
      setResults(command.output);
    }
  }

  return (
    <CommandFormPanel
      title="LLM-as-Judge Evaluation"
      missing={missing}
      isRunning={command.isRunning}
      submitLabel="Run Judge"
      runningLabel="Evaluating..."
      onSubmit={() => runJudge().catch(console.error)}
      error={command.error}
      output={results}
    >
      <div className="grid-2">
        <FormField label="Model Path" required>
          <PathInput value={modelPath} onChange={setModelPath} placeholder="/path/to/model.pt" filters={[{ name: "Model", extensions: ["pt"] }]} />
        </FormField>
        <FormField label="Judge API Endpoint" required>
          <input value={judgeApi} onChange={(e) => setJudgeApi(e.currentTarget.value)} placeholder="https://api.openai.com/v1/chat/completions" />
        </FormField>
        <FormField label="API Key" hint="optional">
          <input type="password" value={apiKey} onChange={(e) => setApiKey(e.currentTarget.value)} placeholder="sk-..." />
        </FormField>
        <FormField label="Criteria" hint="comma-separated">
          <input value={criteria} onChange={(e) => setCriteria(e.currentTarget.value)} />
        </FormField>
        <FormField label="Test Prompts File" hint="optional">
          <PathInput value={testPrompts} onChange={setTestPrompts} placeholder="/path/to/prompts.jsonl" filters={[{ name: "JSONL", extensions: ["jsonl"] }]} />
        </FormField>
      </div>
    </CommandFormPanel>
  );
}
