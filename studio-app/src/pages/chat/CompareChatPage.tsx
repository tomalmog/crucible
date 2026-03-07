import { useState } from "react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { useForge } from "../../context/ForgeContext";
import { FormField } from "../../components/shared/FormField";
import { ModelSelect } from "../../components/shared/ModelSelect";

interface Comparison {
  prompt: string;
  responseA: string;
  responseB: string;
  preference: string;
}

export function CompareChatPage() {
  const { dataRoot } = useForge();
  const command = useForgeCommand();
  const [modelA, setModelA] = useState("");
  const [modelB, setModelB] = useState("");
  const [prompt, setPrompt] = useState("");
  const [comparisons, setComparisons] = useState<Comparison[]>([]);
  const [loading, setLoading] = useState(false);

  async function sendPrompt() {
    if (!dataRoot || !modelA.trim() || !modelB.trim() || !prompt.trim()) return;
    setLoading(true);
    await command.run(dataRoot, ["ab-chat", "--model-a", modelA, "--model-b", modelB]);
    const newComparison: Comparison = {
      prompt,
      responseA: `[Model A: ${prompt}]`,
      responseB: `[Model B: ${prompt}]`,
      preference: "",
    };
    setComparisons([...comparisons, newComparison]);
    setPrompt("");
    setLoading(false);
  }

  function setPreference(index: number, pref: string) {
    const updated = [...comparisons];
    updated[index] = { ...updated[index], preference: pref };
    setComparisons(updated);
  }

  return (
    <div>
      <div className="page-header">
        <h1>A/B Model Comparison</h1>
      </div>
      <div className="panel stack-md">
        <div className="grid-2">
          <FormField label="Model A">
            <ModelSelect value={modelA} onChange={setModelA} />
          </FormField>
          <FormField label="Model B">
            <ModelSelect value={modelB} onChange={setModelB} />
          </FormField>
        </div>
      </div>

      <div className="panel stack-md">
        {comparisons.map((c, i) => (
          <div key={i} className="stack-sm section-divider">
            <p><strong>Prompt:</strong> {c.prompt}</p>
            <div className="grid-2">
              <div className="panel">
                <h4>Model A</h4>
                <p>{c.responseA}</p>
              </div>
              <div className="panel">
                <h4>Model B</h4>
                <p>{c.responseB}</p>
              </div>
            </div>
            <div className="row">
              <button className={`btn btn-sm${c.preference === "a" ? " btn-primary" : ""}`} onClick={() => setPreference(i, "a")}>
                A is Better
              </button>
              <button className={`btn btn-sm${c.preference === "tie" ? " btn-primary" : ""}`} onClick={() => setPreference(i, "tie")}>
                Tie
              </button>
              <button className={`btn btn-sm${c.preference === "b" ? " btn-primary" : ""}`} onClick={() => setPreference(i, "b")}>
                B is Better
              </button>
            </div>
          </div>
        ))}

        <div className="row">
          <input
            className="input-grow"
            value={prompt}
            onChange={(e) => setPrompt(e.currentTarget.value)}
            placeholder="Enter a prompt to compare..."
            onKeyDown={(e) => e.key === "Enter" && sendPrompt()}
          />
          <button className="btn btn-primary" onClick={() => sendPrompt().catch(console.error)} disabled={loading || !prompt.trim()}>
            {loading ? "Generating..." : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
