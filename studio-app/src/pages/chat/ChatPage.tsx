import { FormEvent, useEffect, useMemo, useState, Dispatch, SetStateAction } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { useForge } from "../../context/ForgeContext";
import { getForgeCommandStatus, startForgeCommand } from "../../api/studioApi";
import { FormField } from "../../components/shared/FormField";
import { FormSection } from "../../components/shared/FormSection";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

const CHAT_POLL_MS = 100;
const SAMPLING_PRESETS = {
  deterministic: { maxNewTokens: "80", temperature: "0", topK: "0" },
  balanced: { maxNewTokens: "120", temperature: "0.7", topK: "40" },
  creative: { maxNewTokens: "160", temperature: "1.0", topK: "80" },
} as const;
type SamplingPreset = keyof typeof SAMPLING_PRESETS | "custom";

export function ChatPage() {
  const { dataRoot, selectedDataset } = useForge();
  const [datasetName, setDatasetName] = useState(selectedDataset ?? "");
  const [tokenizerPath, setTokenizerPath] = useState("");
  const [weightsPath, setWeightsPath] = useState("");
  const [modelPath, setModelPath] = useState("gpt2");
  const [versionId] = useState("");
  const [maxNewTokens, setMaxNewTokens] = useState("120");
  const [temperature, setTemperature] = useState("0.7");
  const [topK, setTopK] = useState("40");
  const [maxTokenLength, setMaxTokenLength] = useState("256");
  const [positionEmbeddingType, setPositionEmbeddingType] = useState("learned");
  const [samplingPreset, setSamplingPreset] = useState<SamplingPreset>("balanced");
  const [draftMessage, setDraftMessage] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);

  useEffect(() => {
    if (selectedDataset) setDatasetName(selectedDataset);
  }, [selectedDataset]);

  const canSend = useMemo(
    () => modelPath.trim().length > 0 && draftMessage.trim().length > 0 && !isSending,
    [modelPath, draftMessage, isSending],
  );

  function applyPreset(preset: SamplingPreset) {
    setSamplingPreset(preset);
    if (preset !== "custom") {
      const v = SAMPLING_PRESETS[preset];
      setMaxNewTokens(v.maxNewTokens);
      setTemperature(v.temperature);
      setTopK(v.topK);
    }
  }

  function updateSampling(field: string, value: string) {
    setSamplingPreset("custom");
    if (field === "maxNewTokens") setMaxNewTokens(value);
    else if (field === "temperature") setTemperature(value);
    else setTopK(value);
  }

  async function onSendMessage(event: FormEvent) {
    event.preventDefault();
    if (!canSend) return;
    const userText = draftMessage.trim();
    setDraftMessage("");
    setChatError(null);
    setMessages((c) => [...c, { role: "user", content: userText }]);
    setIsSending(true);

    try {
      const prompt = buildPromptText(messages, userText);
      const args = buildChatArgs(datasetName, tokenizerPath, modelPath, prompt, versionId, maxNewTokens, temperature, topK, maxTokenLength, positionEmbeddingType, weightsPath);
      setMessages((c) => [...c, { role: "assistant", content: "" }]);
      const taskStart = await startForgeCommand(dataRoot, args);
      const taskStatus = await streamChatTask(taskStart.task_id, setMessages);
      if (taskStatus.status !== "completed" || taskStatus.exit_code !== 0) {
        throw new Error(taskStatus.stderr || "Chat command failed.");
      }
      const responseText = taskStatus.stdout.trim();
      setMessages((c) => {
        const updated = [...c];
        updated[updated.length - 1] = { role: "assistant", content: responseText || "(no response generated)" };
        return updated;
      });
    } catch (error) {
      setChatError(error instanceof Error ? error.message : String(error));
    } finally {
      setIsSending(false);
    }
  }

  return (
    <>
      <PageHeader title="Chat">
        <button className="btn btn-sm" onClick={() => setMessages([])} disabled={isSending}>
          Clear
        </button>
      </PageHeader>

      <div className="stack-lg">
        <FormSection title="Model Configuration" defaultOpen>
          <div className="chat-config-grid">
            <FormField label="Model">
              <input value={modelPath} onChange={(e) => setModelPath(e.currentTarget.value)} placeholder="gpt2, meta-llama/Llama-2-7b, or /path/to/model.pt" />
            </FormField>
            <FormField label="Dataset (optional)">
              <input value={datasetName} onChange={(e) => setDatasetName(e.currentTarget.value)} placeholder="optional" />
            </FormField>
            <FormField label="Tokenizer Path">
              <input value={tokenizerPath} onChange={(e) => setTokenizerPath(e.currentTarget.value)} placeholder="auto-detect" />
            </FormField>
            <FormField label="Custom Weights">
              <input value={weightsPath} onChange={(e) => setWeightsPath(e.currentTarget.value)} placeholder="optional .pt or .safetensors path" />
            </FormField>
            <FormField label="Sampling Preset">
              <select value={samplingPreset} onChange={(e) => applyPreset(e.currentTarget.value as SamplingPreset)}>
                <option value="deterministic">deterministic</option>
                <option value="balanced">balanced</option>
                <option value="creative">creative</option>
                <option value="custom">custom</option>
              </select>
            </FormField>
            <FormField label="Max New Tokens">
              <input value={maxNewTokens} onChange={(e) => updateSampling("maxNewTokens", e.currentTarget.value)} />
            </FormField>
            <FormField label="Temperature">
              <input value={temperature} onChange={(e) => updateSampling("temperature", e.currentTarget.value)} />
            </FormField>
            <FormField label="Top K">
              <input value={topK} onChange={(e) => updateSampling("topK", e.currentTarget.value)} />
            </FormField>
            <FormField label="Max Token Length">
              <input value={maxTokenLength} onChange={(e) => setMaxTokenLength(e.currentTarget.value)} />
            </FormField>
            <FormField label="Position Embedding">
              <select value={positionEmbeddingType} onChange={(e) => setPositionEmbeddingType(e.currentTarget.value)}>
                <option value="learned">learned</option>
                <option value="sinusoidal">sinusoidal</option>
              </select>
            </FormField>
          </div>
        </FormSection>

        <div className="chat-thread">
          {messages.length === 0 ? (
            <p className="chat-empty">Send a message to evaluate your trained model.</p>
          ) : (
            messages.map((msg, i) => (
              <article key={`${msg.role}-${i}`} className={`chat-message ${msg.role}`}>
                <header>{msg.role === "user" ? "You" : "Model"}</header>
                <p>{msg.content}</p>
              </article>
            ))
          )}
        </div>

        {chatError && <p className="chat-error">{chatError}</p>}

        <form className="chat-input-row" onSubmit={onSendMessage}>
          <input value={draftMessage} onChange={(e) => setDraftMessage(e.currentTarget.value)} placeholder="Type a prompt..." />
          <button className="btn btn-primary" type="submit" disabled={!canSend}>
            {isSending ? "Sending..." : "Send"}
          </button>
          <button className="btn" type="button" disabled={isSending} onClick={() => setMessages([])}>
            Clear
          </button>
        </form>
      </div>
    </>
  );
}

async function streamChatTask(taskId: string, setMessages: Dispatch<SetStateAction<ChatMessage[]>>) {
  while (true) {
    const status = await getForgeCommandStatus(taskId);
    const partial = status.stdout.trim();
    if (partial.length > 0) {
      setMessages((c) => {
        const updated = [...c];
        updated[updated.length - 1] = { role: "assistant", content: partial };
        return updated;
      });
    }
    if (status.status !== "running") return status;
    await new Promise((r) => setTimeout(r, CHAT_POLL_MS));
  }
}

function buildPromptText(_messages: ChatMessage[], currentText: string): string {
  return currentText;
}

function buildChatArgs(dataset: string, tokenizer: string, model: string, prompt: string, version: string, maxTokens: string, temp: string, topK: string, maxLen: string, posEmb: string, weights: string): string[] {
  const args = ["chat", "--model-path", model.trim(), "--prompt", prompt];
  const optionals: [string, string][] = [
    ["--dataset", dataset], ["--tokenizer-path", tokenizer], ["--version-id", version],
    ["--max-new-tokens", maxTokens], ["--temperature", temp], ["--top-k", topK],
    ["--max-token-length", maxLen], ["--position-embedding-type", posEmb],
    ["--weights-path", weights],
  ];
  for (const [flag, val] of optionals) {
    if (val.trim()) args.push(flag, val.trim());
  }
  return args;
}
