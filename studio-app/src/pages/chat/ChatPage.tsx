import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { useCrucible } from "../../context/CrucibleContext";
import { getCrucibleCommandStatus, killCrucibleTask, startCrucibleCommand } from "../../api/studioApi";
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

const SESSION_CHAT_KEY = "crucible_chat_messages";
const SESSION_MODEL_KEY = "crucible_chat_model";
const SESSION_TASK_KEY = "crucible_chat_task";
const SESSION_SENDING_KEY = "crucible_chat_sending";

function loadSessionMessages(): ChatMessage[] {
  try {
    const raw = sessionStorage.getItem(SESSION_CHAT_KEY);
    return raw ? (JSON.parse(raw) as ChatMessage[]) : [];
  } catch {
    return [];
  }
}

function saveSessionMessages(msgs: ChatMessage[]): void {
  sessionStorage.setItem(SESSION_CHAT_KEY, JSON.stringify(msgs));
}

// Module-level active task — survives component unmount.
let _activeTaskId: string | null = sessionStorage.getItem(SESSION_TASK_KEY) || null;

function setActiveTask(id: string | null) {
  _activeTaskId = id;
  if (id) sessionStorage.setItem(SESSION_TASK_KEY, id);
  else sessionStorage.removeItem(SESSION_TASK_KEY);
}

// Kill running chat task when the app window closes.
window.addEventListener("beforeunload", () => {
  if (_activeTaskId) {
    killCrucibleTask(_activeTaskId).catch(() => {});
  }
});

const CHAT_POLL_MS = 100;

/** Extract the last CRUCIBLE_ status line from stderr for progress display. */
function extractStatusLine(stderr: string): string {
  const lines = stderr.split("\n").filter((l) => l.startsWith("CRUCIBLE_"));
  return lines.length > 0 ? lines[lines.length - 1].replace(/^CRUCIBLE_\w+:\s*/, "") : "";
}
const SAMPLING_PRESETS = {
  deterministic: { maxNewTokens: "80", temperature: "0", topK: "0" },
  balanced: { maxNewTokens: "120", temperature: "0.7", topK: "40" },
  creative: { maxNewTokens: "160", temperature: "1.0", topK: "80" },
} as const;
type SamplingPreset = keyof typeof SAMPLING_PRESETS | "custom";

export function ChatPage() {
  const { dataRoot, selectedDataset, models } = useCrucible();
  const [datasetName, setDatasetName] = useState(selectedDataset ?? "");
  const [tokenizerPath, setTokenizerPath] = useState("");
  const [weightsPath, setWeightsPath] = useState("");
  const [modelPath, setModelPath] = useState(() => sessionStorage.getItem(SESSION_MODEL_KEY) ?? "");
  const [maxNewTokens, setMaxNewTokens] = useState("120");
  const [temperature, setTemperature] = useState("0.7");
  const [topK, setTopK] = useState("40");
  const [maxTokenLength, setMaxTokenLength] = useState("256");
  const [positionEmbeddingType, setPositionEmbeddingType] = useState("learned");
  const [samplingPreset, setSamplingPreset] = useState<SamplingPreset>("balanced");
  const [draftMessage, setDraftMessage] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>(loadSessionMessages);
  const [isSending, setIsSending] = useState(() => sessionStorage.getItem(SESSION_SENDING_KEY) === "1");
  const [chatError, setChatError] = useState<string | null>(null);
  const [statusLine, setStatusLine] = useState("");
  const chatThreadRef = useRef<HTMLDivElement>(null);

  // On mount: if there's an in-flight task from before navigation, resume polling.
  useEffect(() => {
    if (!_activeTaskId) {
      // No task — clear any stale sending flag.
      if (isSending) {
        setIsSending(false);
        sessionStorage.removeItem(SESSION_SENDING_KEY);
      }
      return;
    }
    let cancelled = false;

    async function resumePoll() {
      const taskId = _activeTaskId!;
      try {
        // Initial check — if the task is gone or already done, clear immediately.
        const first = await getCrucibleCommandStatus(taskId);
        if (first.status !== "running") {
          if (first.status === "completed" && first.exit_code === 0) {
            const text = first.stdout.trim();
            setMessages((c) => {
              const updated = [...c];
              updated[updated.length - 1] = { role: "assistant", content: text || "(no response generated)" };
              return updated;
            });
          } else if (first.stderr) {
            setChatError(first.stderr);
          }
          clearActiveChat();
          return;
        }

        // Still running — poll until done.
        while (!cancelled) {
          await new Promise((r) => setTimeout(r, CHAT_POLL_MS));
          const status = await getCrucibleCommandStatus(taskId);
          const partial = status.stdout.trim();
          if (partial.length > 0) {
            setMessages((c) => {
              const updated = [...c];
              updated[updated.length - 1] = { role: "assistant", content: partial };
              return updated;
            });
          }
          setStatusLine(extractStatusLine(status.stderr));
          if (status.status !== "running") {
            if (status.status !== "completed" || status.exit_code !== 0) {
              setChatError(status.stderr || "Chat command failed.");
              killCrucibleTask(taskId).catch(() => {});
            } else {
              const text = status.stdout.trim();
              setMessages((c) => {
                const updated = [...c];
                updated[updated.length - 1] = { role: "assistant", content: text || "(no response generated)" };
                return updated;
              });
            }
            clearActiveChat();
            return;
          }
        }
      } catch {
        // Task doesn't exist or status call failed — clear stale state.
        if (!cancelled) clearActiveChat();
      }
    }

    function clearActiveChat() {
      setActiveTask(null);
      setIsSending(false);
      setStatusLine("");
      sessionStorage.removeItem(SESSION_SENDING_KEY);
    }

    resumePoll();
    return () => { cancelled = true; };
  }, []);

  // Detect whether the selected model is on a remote cluster
  const remoteHost = useMemo(() => {
    if (!modelPath) return "";
    const match = models.find((m) => m.remotePath === modelPath);
    return match ? match.remoteHost : "";
  }, [modelPath, models]);
  const isRemoteModel = remoteHost.length > 0;
  const remote = useRemoteChatConfig(dataRoot, remoteHost, isRemoteModel);

  // Sync dataset name from global context when it changes
  useEffect(() => {
    if (selectedDataset) setDatasetName(selectedDataset);
  }, [selectedDataset]);

  // Persist messages and model to sessionStorage so they survive page navigation.
  useEffect(() => { saveSessionMessages(messages); }, [messages]);
  useEffect(() => { sessionStorage.setItem(SESSION_MODEL_KEY, modelPath); }, [modelPath]);

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
    sessionStorage.setItem(SESSION_SENDING_KEY, "1");

    let taskId: string | null = null;
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
      const taskStart = await startCrucibleCommand(dataRoot, args);
      taskId = taskStart.task_id;
      setActiveTask(taskId);

      // Poll until done
      while (true) {
        const status = await getCrucibleCommandStatus(taskId);
        const partial = status.stdout.trim();
        if (partial.length > 0) {
          setMessages((c) => {
            const updated = [...c];
            updated[updated.length - 1] = { role: "assistant", content: partial };
            return updated;
          });
        }
        setStatusLine(extractStatusLine(status.stderr));
        if (status.status !== "running") {
          if (status.status !== "completed" || status.exit_code !== 0) {
            throw new Error(status.stderr || "Chat command failed.");
          }
          const responseText = status.stdout.trim();
          setMessages((c) => {
            const updated = [...c];
            updated[updated.length - 1] = { role: "assistant", content: responseText || "(no response generated)" };
            return updated;
          });
          break;
        }
        await new Promise((r) => setTimeout(r, CHAT_POLL_MS));
      }
    } catch (error) {
      setChatError(error instanceof Error ? error.message : String(error));
      if (taskId) killCrucibleTask(taskId).catch(() => {});
    } finally {
      setActiveTask(null);
      setIsSending(false);
      setStatusLine("");
      sessionStorage.removeItem(SESSION_SENDING_KEY);
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
            {isRemoteModel && remote.clusterInfo && remote.isSlurm && <>
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

        {isSending && statusLine && (
          <p className="chat-status">{statusLine}</p>
        )}
        {chatError && <p className="chat-error">{chatError}</p>}

        <form className="chat-input-row" onSubmit={onSendMessage}>
          <input value={draftMessage} onChange={(e) => setDraftMessage(e.currentTarget.value)} placeholder="Type a prompt..." />
          <button className="btn btn-primary" type="submit" disabled={!canSend}>
            {isSending ? "Sending..." : "Send"}
          </button>
          <button className="btn" type="button" onClick={() => {
            if (_activeTaskId) killCrucibleTask(_activeTaskId).catch(() => {});
            setActiveTask(null);
            setIsSending(false);
            setStatusLine("");
            sessionStorage.removeItem(SESSION_SENDING_KEY);
            setChatError(null);
            setMessages([]);
          }}>
            Clear
          </button>
        </form>
      </div>
    </>
  );
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
