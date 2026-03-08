import { FormEvent, useEffect, useMemo, useRef, useState, Dispatch, SetStateAction } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { useForge } from "../../context/ForgeContext";
import { getForgeCommandStatus, startForgeCommand } from "../../api/studioApi";
import { listClusters } from "../../api/remoteApi";
import { DatasetSelect } from "../../components/shared/DatasetSelect";
import { FormField } from "../../components/shared/FormField";
import { FormSection } from "../../components/shared/FormSection";
import { ModelSelect } from "../../components/shared/ModelSelect";
import { useRemoteChatConfig } from "../../hooks/useRemoteChatConfig";

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
  const { dataRoot, selectedDataset, modelGroups } = useForge();
  const [datasetName, setDatasetName] = useState(selectedDataset ?? "");
  const [tokenizerPath, setTokenizerPath] = useState("");
  const [weightsPath, setWeightsPath] = useState("");
  const [modelPath, setModelPath] = useState("");
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
  const chatThreadRef = useRef<HTMLDivElement>(null);

  // Detect whether the selected model is on a remote cluster
  const remoteHost = useMemo(() => {
    if (!modelPath) return "";
    const group = modelGroups.find((g) => g.activeRemotePath === modelPath);
    return group ? group.activeRemoteHost : "";
  }, [modelPath, modelGroups]);
  const isRemoteModel = remoteHost.length > 0;
  const remote = useRemoteChatConfig(dataRoot, remoteHost, isRemoteModel);

  // Sync dataset name from global context when it changes
  useEffect(() => {
    if (selectedDataset) setDatasetName(selectedDataset);
  }, [selectedDataset]);

  // Auto-scroll chat thread to bottom when new messages arrive
  useEffect(() => {
    if (chatThreadRef.current) {
      chatThreadRef.current.scrollTop = chatThreadRef.current.scrollHeight;
    }
  }, [messages]);

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
      let args: string[];
      if (isRemoteModel) {
        const clusterName = await resolveClusterName(dataRoot, remoteHost);
        args = buildRemoteChatArgs(clusterName, modelPath, prompt, maxNewTokens, temperature, topK, remote.config);
      } else {
        args = buildChatArgs(datasetName, tokenizerPath, modelPath, prompt, maxNewTokens, temperature, topK, maxTokenLength, positionEmbeddingType, weightsPath);
      }
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
              <ModelSelect value={modelPath} onChange={setModelPath} />
            </FormField>
            {!isRemoteModel && <>
              <FormField label="Dataset (optional)">
                <DatasetSelect value={datasetName} onChange={(v) => setDatasetName(v)} placeholder="optional" />
              </FormField>
              <FormField label="Tokenizer Path">
                <input value={tokenizerPath} onChange={(e) => setTokenizerPath(e.currentTarget.value)} placeholder="auto-detect" />
              </FormField>
              <FormField label="Custom Weights">
                <input value={weightsPath} onChange={(e) => setWeightsPath(e.currentTarget.value)} placeholder="optional .pt or .safetensors path" />
              </FormField>
            </>}
            {isRemoteModel && remote.clusterInfo && <>
              <FormField label="Partition">
                <select value={remote.config.partition} onChange={(e) => remote.setPartition(e.currentTarget.value)}>
                  <option value="">Default</option>
                  {remote.clusterInfo.partitions.map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
              </FormField>
              <FormField label="GPU Type">
                <select value={remote.config.gpuType} onChange={(e) => remote.setGpuType(e.currentTarget.value)}>
                  <option value="">Any</option>
                  {remote.clusterInfo.gpuTypes.map((g) => <option key={g} value={g}>{g}</option>)}
                </select>
              </FormField>
              <FormField label="Memory">
                <input value={remote.config.memory} onChange={(e) => remote.setMemory(e.currentTarget.value)} />
              </FormField>
              <FormField label="Time Limit">
                <input value={remote.config.timeLimit} onChange={(e) => remote.setTimeLimit(e.currentTarget.value)} placeholder="HH:MM:SS" />
              </FormField>
            </>}
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
            {!isRemoteModel && <>
              <FormField label="Max Token Length">
                <input value={maxTokenLength} onChange={(e) => setMaxTokenLength(e.currentTarget.value)} />
              </FormField>
              <FormField label="Position Embedding">
                <select value={positionEmbeddingType} onChange={(e) => setPositionEmbeddingType(e.currentTarget.value)}>
                  <option value="learned">learned</option>
                  <option value="sinusoidal">sinusoidal</option>
                </select>
              </FormField>
            </>}
          </div>
        </FormSection>

        <div className="chat-thread" ref={chatThreadRef}>
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

async function resolveClusterName(dataRoot: string, host: string): Promise<string> {
  const clusters = await listClusters(dataRoot);
  const match = clusters.find((c) => c.host === host);
  if (!match) {
    throw new Error(`No registered cluster found for host "${host}".`);
  }
  return match.name;
}

function buildRemoteChatArgs(
  cluster: string, modelPath: string, prompt: string,
  maxTokens: string, temp: string, topK: string,
  res: { partition: string; gpuType: string; memory: string; timeLimit: string },
): string[] {
  const args = [
    "remote", "chat",
    "--cluster", cluster,
    "--model-path", modelPath.trim(),
    "--prompt", prompt,
    "--max-new-tokens", maxTokens.trim() || "80",
    "--temperature", temp.trim() || "0.7",
    "--top-k", topK.trim() || "40",
  ];
  if (res.partition) args.push("--partition", res.partition);
  if (res.gpuType) args.push("--gpu-type", res.gpuType);
  if (res.memory) args.push("--memory", res.memory);
  if (res.timeLimit) args.push("--time-limit", res.timeLimit);
  return args;
}

function buildChatArgs(dataset: string, tokenizer: string, model: string, prompt: string, maxTokens: string, temp: string, topK: string, maxLen: string, posEmb: string, weights: string): string[] {
  const args = ["chat", "--model-path", model.trim(), "--prompt", prompt];
  const optionals: [string, string][] = [
    ["--dataset", dataset], ["--tokenizer-path", tokenizer],
    ["--max-new-tokens", maxTokens], ["--temperature", temp], ["--top-k", topK],
    ["--max-token-length", maxLen], ["--position-embedding-type", posEmb],
    ["--weights-path", weights],
  ];
  for (const [flag, val] of optionals) {
    if (val.trim()) args.push(flag, val.trim());
  }
  return args;
}
