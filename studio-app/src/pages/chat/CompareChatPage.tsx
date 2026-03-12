import { useMemo, useState } from "react";
import { useCrucible } from "../../context/CrucibleContext";
import { FormField } from "../../components/shared/FormField";
import { ModelSelect } from "../../components/shared/ModelSelect";
import { PageHeader } from "../../components/shared/PageHeader";
import {
  startCrucibleCommand,
  getCrucibleCommandStatus,
  killCrucibleTask,
  writeTextFile,
} from "../../api/studioApi";
import { listClusters } from "../../api/remoteApi";
import { save } from "@tauri-apps/plugin-dialog";

interface Comparison {
  prompt: string;
  responseA: string;
  responseB: string;
  preference: "" | "a" | "b" | "tie";
}

const POLL_MS = 100;

export function CompareChatPage() {
  const { dataRoot, models } = useCrucible();
  const [modelA, setModelA] = useState("");
  const [modelB, setModelB] = useState("");
  const [prompt, setPrompt] = useState("");
  const [comparisons, setComparisons] = useState<Comparison[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const remoteHostA = useMemo(() => {
    if (!modelA) return "";
    const match = models.find((m) => m.remotePath === modelA);
    return match ? match.remoteHost : "";
  }, [modelA, models]);

  const remoteHostB = useMemo(() => {
    if (!modelB) return "";
    const match = models.find((m) => m.remotePath === modelB);
    return match ? match.remoteHost : "";
  }, [modelB, models]);

  async function buildChatArgs(modelPath: string, remoteHost: string, promptText: string): Promise<string[]> {
    if (remoteHost) {
      const clusters = await listClusters(dataRoot);
      const cluster = clusters.find((c) => c.host === remoteHost);
      if (!cluster) throw new Error(`No registered cluster found for host "${remoteHost}".`);
      return [
        "remote", "chat",
        "--cluster", cluster.name,
        "--model-path", modelPath.trim(),
        "--prompt", promptText,
        "--max-new-tokens", "80",
        "--temperature", "0.7",
        "--top-k", "40",
      ];
    }
    return ["chat", "--model-path", modelPath.trim(), "--prompt", promptText];
  }

  async function sendPrompt() {
    if (!dataRoot || !modelA.trim() || !modelB.trim() || !prompt.trim()) return;
    setLoading(true);
    setError(null);

    const currentPrompt = prompt.trim();
    setPrompt("");

    const idx = comparisons.length;
    setComparisons((c) => [...c, { prompt: currentPrompt, responseA: "", responseB: "", preference: "" }]);

    let taskIdA: string | null = null;
    let taskIdB: string | null = null;

    try {
      const [argsA, argsB] = await Promise.all([
        buildChatArgs(modelA, remoteHostA, currentPrompt),
        buildChatArgs(modelB, remoteHostB, currentPrompt),
      ]);

      const [startA, startB] = await Promise.all([
        startCrucibleCommand(dataRoot, argsA),
        startCrucibleCommand(dataRoot, argsB),
      ]);
      taskIdA = startA.task_id;
      taskIdB = startB.task_id;

      let doneA = false;
      let doneB = false;

      while (!doneA || !doneB) {
        await new Promise((r) => setTimeout(r, POLL_MS));

        const [statusA, statusB] = await Promise.all([
          doneA ? null : getCrucibleCommandStatus(taskIdA),
          doneB ? null : getCrucibleCommandStatus(taskIdB),
        ]);

        setComparisons((c) => {
          const updated = [...c];
          const entry = { ...updated[idx] };
          if (statusA) {
            const partial = statusA.stdout.trim();
            if (partial) entry.responseA = partial;
          }
          if (statusB) {
            const partial = statusB.stdout.trim();
            if (partial) entry.responseB = partial;
          }
          updated[idx] = entry;
          return updated;
        });

        if (statusA && statusA.status !== "running") {
          doneA = true;
          if (statusA.status !== "completed" || statusA.exit_code !== 0) {
            throw new Error(`Model A failed: ${statusA.stderr || "unknown error"}`);
          }
          setComparisons((c) => {
            const updated = [...c];
            updated[idx] = { ...updated[idx], responseA: statusA.stdout.trim() || "(no response)" };
            return updated;
          });
        }

        if (statusB && statusB.status !== "running") {
          doneB = true;
          if (statusB.status !== "completed" || statusB.exit_code !== 0) {
            throw new Error(`Model B failed: ${statusB.stderr || "unknown error"}`);
          }
          setComparisons((c) => {
            const updated = [...c];
            updated[idx] = { ...updated[idx], responseB: statusB.stdout.trim() || "(no response)" };
            return updated;
          });
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      // Remove the incomplete comparison entry
      setComparisons((c) => {
        const updated = [...c];
        const entry = updated[idx];
        if (entry && !entry.responseA && !entry.responseB) {
          updated.splice(idx, 1);
        }
        return updated;
      });
      if (taskIdA) killCrucibleTask(taskIdA).catch(() => {});
      if (taskIdB) killCrucibleTask(taskIdB).catch(() => {});
    } finally {
      setLoading(false);
    }
  }

  function setPreference(index: number, pref: "a" | "b" | "tie") {
    setComparisons((c) => {
      const updated = [...c];
      updated[index] = { ...updated[index], preference: pref };
      return updated;
    });
  }

  async function exportDpo() {
    const pairs = comparisons
      .filter((c) => c.preference === "a" || c.preference === "b")
      .map((c) => ({
        prompt: c.prompt,
        chosen: c.preference === "a" ? c.responseA : c.responseB,
        rejected: c.preference === "a" ? c.responseB : c.responseA,
      }));

    if (pairs.length === 0) return;

    const filePath = await save({
      defaultPath: "dpo_preferences.jsonl",
      filters: [{ name: "JSONL", extensions: ["jsonl"] }],
    });
    if (!filePath) return;

    const jsonl = pairs.map((p) => JSON.stringify(p)).join("\n") + "\n";
    await writeTextFile(filePath, jsonl);
  }

  const ratedCount = comparisons.filter((c) => c.preference === "a" || c.preference === "b").length;

  return (
    <>
      <PageHeader title="A/B Compare">
        {ratedCount > 0 && (
          <button className="btn btn-sm" onClick={exportDpo}>
            Export DPO ({ratedCount})
          </button>
        )}
      </PageHeader>

      <div className="stack-lg">
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

        {error && <p className="chat-error">{error}</p>}

        <div className="stack-lg">
          {comparisons.map((c, i) => (
            <div key={i} className="panel stack-md">
              <p><strong>Prompt:</strong> {c.prompt}</p>
              <div className="grid-2">
                <div className="panel" style={{ minWidth: 0 }}>
                  <h4>Model A</h4>
                  <p style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{c.responseA || (loading && i === comparisons.length - 1 ? "Generating..." : "")}</p>
                </div>
                <div className="panel" style={{ minWidth: 0 }}>
                  <h4>Model B</h4>
                  <p style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{c.responseB || (loading && i === comparisons.length - 1 ? "Generating..." : "")}</p>
                </div>
              </div>
              <div className="row">
                <button
                  className={`btn btn-sm${c.preference === "a" ? " btn-primary" : ""}`}
                  onClick={() => setPreference(i, "a")}
                >
                  A is Better
                </button>
                <button
                  className={`btn btn-sm${c.preference === "tie" ? " btn-primary" : ""}`}
                  onClick={() => setPreference(i, "tie")}
                >
                  Tie
                </button>
                <button
                  className={`btn btn-sm${c.preference === "b" ? " btn-primary" : ""}`}
                  onClick={() => setPreference(i, "b")}
                >
                  B is Better
                </button>
              </div>
            </div>
          ))}
        </div>

        <div className="row">
          <input
            className="input-grow"
            value={prompt}
            onChange={(e) => setPrompt(e.currentTarget.value)}
            placeholder="Enter a prompt to compare..."
            onKeyDown={(e) => e.key === "Enter" && sendPrompt()}
          />
          <button
            className="btn btn-primary"
            onClick={() => sendPrompt().catch(console.error)}
            disabled={loading || !prompt.trim() || !modelA.trim() || !modelB.trim()}
          >
            {loading ? "Generating..." : "Send"}
          </button>
        </div>
      </div>
    </>
  );
}
