import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { useCrucible } from "../../context/CrucibleContext";
import { getCrucibleCommandStatus, killCrucibleTask, startCrucibleCommand } from "../../api/studioApi";
import { useRemoteChatConfig } from "../../hooks/useRemoteChatConfig";
import { ChatConfigPanel } from "./ChatConfigPanel";
import { ChatInputRow } from "./ChatInputRow";
import { ChatThread } from "./ChatThread";
import {
  loadChatMessages,
  loadChatScreenState,
  saveChatMessages,
  saveChatScreenState,
  type ChatMessage,
  type SamplingPreset,
} from "./chatPersistence";
import {
  SAMPLING_PRESETS,
  buildChatArgs,
  buildPromptText,
  buildRemoteChatArgs,
  extractStatusLine,
  formatChatFailure,
  resolveClusterName,
} from "./chatRuntime";
import {
  clearChatSending,
  getActiveChatTaskId,
  markChatSending,
  readChatSendingFlag,
  setActiveChatTaskId,
} from "./chatTaskSession";
import { useChatTaskResume } from "./useChatTaskResume";

// Kill running chat task when the app window closes.
if (typeof window !== "undefined") {
  window.addEventListener("beforeunload", () => {
    const taskId = getActiveChatTaskId();
    if (taskId) {
      killCrucibleTask(taskId).catch(() => {});
    }
  });
}

const CHAT_POLL_MS = 100;

export function ChatPage() {
  const { dataRoot, selectedDataset, models } = useCrucible();
  const initialState = useMemo(() => loadChatScreenState(), []);
  const [datasetName, setDatasetName] = useState(initialState.datasetName || selectedDataset || "");
  const [tokenizerPath, setTokenizerPath] = useState(initialState.tokenizerPath);
  const [weightsPath, setWeightsPath] = useState(initialState.weightsPath);
  const [modelPath, setModelPath] = useState(initialState.modelPath);
  const [maxNewTokens, setMaxNewTokens] = useState(initialState.maxNewTokens);
  const [temperature, setTemperature] = useState(initialState.temperature);
  const [topK, setTopK] = useState(initialState.topK);
  const [maxTokenLength, setMaxTokenLength] = useState(initialState.maxTokenLength);
  const [positionEmbeddingType, setPositionEmbeddingType] = useState(initialState.positionEmbeddingType);
  const [samplingPreset, setSamplingPreset] = useState<SamplingPreset>(initialState.samplingPreset);
  const [draftMessage, setDraftMessage] = useState(initialState.draftMessage);
  const [messages, setMessages] = useState<ChatMessage[]>(loadChatMessages);
  const [isSending, setIsSending] = useState(readChatSendingFlag);
  const [chatError, setChatError] = useState<string | null>(null);
  const [statusLine, setStatusLine] = useState("");
  const chatThreadRef = useRef<HTMLDivElement>(null);

  useChatTaskResume({ isSending, setIsSending, setMessages, setChatError, setStatusLine });

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

  // Persist chat messages so the screen survives page navigation and app restarts.
  useEffect(() => { saveChatMessages(messages); }, [messages]);

  // Persist chat configuration so reloads restore the same model and generation setup.
  useEffect(() => {
    saveChatScreenState({
      datasetName,
      tokenizerPath,
      weightsPath,
      modelPath,
      maxNewTokens,
      temperature,
      topK,
      maxTokenLength,
      positionEmbeddingType,
      samplingPreset,
      draftMessage,
    });
  }, [
    datasetName,
    tokenizerPath,
    weightsPath,
    modelPath,
    maxNewTokens,
    temperature,
    topK,
    maxTokenLength,
    positionEmbeddingType,
    samplingPreset,
    draftMessage,
  ]);

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
    markChatSending();

    let taskId: string | null = null;
    try {
      const prompt = buildPromptText(messages, userText);
      let args: string[];
      if (isRemoteModel) {
        const clusterName = await resolveClusterName(dataRoot, remoteHost);
        args = buildRemoteChatArgs({
          cluster: clusterName,
          modelPath,
          prompt,
          maxTokens: maxNewTokens,
          temperature,
          topK,
          resources: remote.config,
        });
      } else {
        args = buildChatArgs({
          dataset: datasetName,
          tokenizer: tokenizerPath,
          model: modelPath,
          prompt,
          maxTokens: maxNewTokens,
          temperature,
          topK,
          maxTokenLength,
          positionEmbeddingType,
          weights: weightsPath,
        });
      }
      setMessages((c) => [...c, { role: "assistant", content: "" }]);
      const taskStart = await startCrucibleCommand(dataRoot, args);
      taskId = taskStart.task_id;
      setActiveChatTaskId(taskId);

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
            throw new Error(formatChatFailure(status.stderr));
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
      setActiveChatTaskId(null);
      setIsSending(false);
      setStatusLine("");
      clearChatSending();
    }
  }

  function clearChat() {
    const taskId = getActiveChatTaskId();
    if (taskId) killCrucibleTask(taskId).catch(() => {});
    setActiveChatTaskId(null);
    setIsSending(false);
    setStatusLine("");
    setChatError(null);
    setMessages([]);
    clearChatSending();
  }

  return (
    <>
      <PageHeader title="Chat">
        <button className="btn btn-sm" onClick={clearChat} disabled={isSending}>
          Clear
        </button>
      </PageHeader>

      <div className="stack-lg">
        <ChatConfigPanel
          modelPath={modelPath}
          onModelPathChange={setModelPath}
          isRemoteModel={isRemoteModel}
          datasetName={datasetName}
          onDatasetNameChange={setDatasetName}
          tokenizerPath={tokenizerPath}
          onTokenizerPathChange={setTokenizerPath}
          weightsPath={weightsPath}
          onWeightsPathChange={setWeightsPath}
          remote={remote}
          samplingPreset={samplingPreset}
          onSamplingPresetChange={applyPreset}
          maxNewTokens={maxNewTokens}
          temperature={temperature}
          topK={topK}
          onSamplingFieldChange={updateSampling}
          maxTokenLength={maxTokenLength}
          onMaxTokenLengthChange={setMaxTokenLength}
          positionEmbeddingType={positionEmbeddingType}
          onPositionEmbeddingTypeChange={setPositionEmbeddingType}
        />

        <ChatThread
          messages={messages}
          isSending={isSending}
          statusLine={statusLine}
          chatError={chatError}
          threadRef={chatThreadRef}
        />

        <ChatInputRow
          draftMessage={draftMessage}
          onDraftMessageChange={setDraftMessage}
          canSend={canSend}
          isSending={isSending}
          onSubmit={onSendMessage}
          onClear={clearChat}
        />
      </div>
    </>
  );
}
