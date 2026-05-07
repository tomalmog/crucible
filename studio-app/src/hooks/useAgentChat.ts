import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router";
import { useCrucible } from "../context/CrucibleContext";
import { loadAgentJobCompletion } from "../api/agentJobPreviews";
import { getJob } from "../api/jobsApi";
import { buildAgentChatContext, buildChainContinuationPrompt } from "./agentChatPayload";
import { runAgentCommand } from "./agentCommandRunner";
import { inferPreviewWorkspace, readWorkspaceDirective } from "./agentWorkspaceDirective";
import type {
  AgentChatSummary,
  AgentJobPreview,
  AgentMessage,
  AgentTraceEvent,
  AgentWorkspaceDirective,
} from "../types/agent";
import { TERMINAL_JOB_STATES } from "../types/jobs";

const ANTHROPIC_API_KEY_STORAGE = "crucible_anthropic_api_key";
const OPENAI_API_KEY_STORAGE = "crucible_openai_api_key";
const PROVIDER_STORAGE = "crucible_agent_provider";
const OPENAI_MODEL_STORAGE = "crucible_agent_openai_model";
const OLLAMA_MODEL_STORAGE = "crucible_agent_ollama_model";
const OLLAMA_URL_STORAGE = "crucible_agent_ollama_url";
const GEMINI_MODEL_STORAGE = "crucible_agent_gemini_model";
const GEMINI_API_KEY_STORAGE = "crucible_gemini_api_key";
const CHAIN_POLL_MS = 3000;
const AUTO_CONTINUE_DELAY_MS = 2200;

export interface PendingChain {
  chatId: string;
  jobId: string;
  steps: string[];
  jobComplete: boolean;
  jobState: string | null;
  jobModelPath: string | null;
  jobModelName: string | null;
  autoContinueAt: number | null;
}

export interface UseAgentChatReturn {
  activeChatId: string | null;
  chats: AgentChatSummary[];
  messages: AgentMessage[];
  currentTrace: AgentTraceEvent[];
  isLoading: boolean;
  error: string | null;
  sendMessage: (text: string) => Promise<void>;
  clearConversation: () => Promise<void>;
  createChat: () => Promise<void>;
  switchChat: (chatId: string) => Promise<void>;
  deleteChat: (chatId: string) => Promise<void>;
  searchChats: (query: string) => Promise<void>;
  pendingChain: PendingChain | null;
  continueChain: () => Promise<void>;
  cancelChain: () => Promise<void>;
}
export function useAgentChat(): UseAgentChatReturn {
  const { dataRoot, models, datasets, selectedModel, selectedDataset } = useCrucible();
  const location = useLocation();
  const navigate = useNavigate();
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [chats, setChats] = useState<AgentChatSummary[]>([]);
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [currentTrace, setCurrentTrace] = useState<AgentTraceEvent[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingChain, setPendingChain] = useState<PendingChain | null>(null);
  const hasLoaded = useRef(false);
  const currentTraceRef = useRef<AgentTraceEvent[]>([]);

  function applyLoadedChatState(res: Record<string, unknown>): void {
    setMessages(readAgentMessages(res.messages));
    setChats(readChatSummaries(res.chats));
    const nextChatId = typeof res.active_chat_id === "string" ? res.active_chat_id : null;
    setActiveChatId(nextChatId);
    setPendingChain(readPendingChain(res.chain, nextChatId));
    currentTraceRef.current = [];
    setCurrentTrace([]);
    setError(null);
  }

  useEffect(() => {
    if (hasLoaded.current || !dataRoot) return;
    hasLoaded.current = true;
    runAgentCommand(dataRoot, { action: "load" })
      .then((res) => {
        applyLoadedChatState(res);
      })
      .catch(() => { /* no history yet */ });
  }, [dataRoot]);

  useEffect(() => {
    if (!pendingChain || pendingChain.jobComplete || !dataRoot) return;
    let isActive = true;
    let isCompleting = false;

    // React to remote-job completion so the agent can surface artifacts
    // immediately and continue the chain without waiting for another prompt.
    const checkPendingJob = async (): Promise<void> => {
      if (isCompleting) return;
      try {
        const job = await getJob(dataRoot, pendingChain.jobId);
        if (!TERMINAL_JOB_STATES.has(job.state)) return;
        if (job.state !== "completed") {
          if (!isActive) return;
          setPendingChain(null);
          runAgentCommand(dataRoot, {
            action: "cancel_chain",
            chat_id: pendingChain.chatId,
          }).catch(() => {});
          const errorDetail = job.errorMessage ? `: ${job.errorMessage}` : "";
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: `The job ${pendingChain.jobId} **${job.state}**${errorDetail}. The remaining chain steps have been cancelled.\n\nYou can fix the issue and ask me to retry.`,
            },
          ]);
          return;
        }

        isCompleting = true;
        const completion = await loadAgentJobCompletion(dataRoot, job);
        if (!isActive) return;
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: completion.content,
            artifact: completion.preview || undefined,
            workspaceDirective: completion.preview ? inferPreviewWorkspace(completion.preview) : undefined,
          },
        ]);
        setPendingChain((prev) => prev ? {
          ...prev,
          jobComplete: true,
          jobState: job.state,
          jobModelPath: completion.modelPath,
          jobModelName: job.modelName || null,
          autoContinueAt: Date.now() + AUTO_CONTINUE_DELAY_MS,
        } : null);
      } catch {
        // The remote record may not be readable yet; the next poll will retry.
      }
    };

    void checkPendingJob();
    const interval = setInterval(async () => {
      await checkPendingJob();
    }, CHAIN_POLL_MS);

    return () => {
      isActive = false;
      clearInterval(interval);
    };
  }, [dataRoot, pendingChain?.chatId, pendingChain?.jobId, pendingChain?.jobComplete]);

  const sendMessage = useCallback(async (text: string) => {
    if (isLoading || !text.trim()) return;

    const anthropicApiKey = localStorage.getItem(ANTHROPIC_API_KEY_STORAGE) || "";
    setError(null);
    setIsLoading(true);
    currentTraceRef.current = [];
    setCurrentTrace([]);
    const isChainContinue = text.startsWith("[Chain continuation]");
    if (!isChainContinue) {
      setMessages((prev) => [...prev, { role: "user", content: text }]);
    }

    try {
      const context = buildAgentChatContext(
        location.pathname,
        selectedModel?.modelName || null,
        selectedDataset || null,
        models,
        datasets,
      );

      const provider = localStorage.getItem(PROVIDER_STORAGE) || "anthropic";
      const effectiveApiKey = provider === "gemini"
        ? (localStorage.getItem(GEMINI_API_KEY_STORAGE) || "")
        : provider === "openai"
          ? (localStorage.getItem(OPENAI_API_KEY_STORAGE) || "")
          : anthropicApiKey;
      const res = await runAgentCommand(dataRoot, {
        action: "chat",
        chat_id: activeChatId || undefined,
        message: text,
        context,
        api_key: effectiveApiKey,
        provider,
        model: provider === "ollama" ? (localStorage.getItem(OLLAMA_MODEL_STORAGE) || "")
             : provider === "gemini" ? (localStorage.getItem(GEMINI_MODEL_STORAGE) || "")
             : provider === "openai" ? (localStorage.getItem(OPENAI_MODEL_STORAGE) || "")
             : "",
        ollama_url: provider === "ollama" ? (localStorage.getItem(OLLAMA_URL_STORAGE) || "") : "",
      }, (event) => {
        currentTraceRef.current = [...currentTraceRef.current, event];
        setCurrentTrace(currentTraceRef.current);
      });

      if (res.error) {
        setError(res.error as string);
        currentTraceRef.current = [];
        setCurrentTrace([]);
        if (!isChainContinue) {
          setMessages((prev) => prev.slice(0, -1));
        }
      } else if (res.content || Array.isArray(res.artifact_messages)) {
        const responseChatId = typeof res.chat_id === "string" ? res.chat_id : activeChatId;
        if (responseChatId) {
          setActiveChatId(responseChatId);
        }
        const nextChats = readChatSummaries(res.chats);
        if (nextChats.length > 0) {
          setChats(nextChats);
        }
        const toolsUsed = res.tools_used as string[] | undefined;
        const navigatedTo = res.navigate_to as string | undefined;
        if (navigatedTo) {
          navigate(navigatedTo);
        }
        const chainData = res.pending_chain as { job_id: string; steps: string[] } | undefined;
        if (chainData && responseChatId) {
          setPendingChain({
            chatId: responseChatId,
            jobId: chainData.job_id,
            steps: chainData.steps,
            jobComplete: false,
            jobState: null,
            jobModelPath: null,
            jobModelName: null,
            autoContinueAt: null,
          });
        }
        const artifactMessages = readAgentMessages(res.artifact_messages);
        const fallbackMessage: AgentMessage = {
          role: "assistant",
          content: typeof res.content === "string" ? res.content : "",
          toolsUsed: toolsUsed?.length ? toolsUsed : undefined,
          navigatedTo: navigatedTo || undefined,
          trace: currentTraceRef.current.length ? [...currentTraceRef.current] : undefined,
          workspaceDirective: readWorkspaceDirective(res),
        };
        const nextMessages = artifactMessages.length > 0
          ? artifactMessages
          : [fallbackMessage];
        setMessages((prev) => [...prev, ...nextMessages]);
        currentTraceRef.current = [];
        setCurrentTrace([]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      currentTraceRef.current = [];
      setCurrentTrace([]);
      if (!isChainContinue) {
        setMessages((prev) => prev.slice(0, -1));
      }
    } finally {
      setIsLoading(false);
    }
  }, [isLoading, dataRoot, activeChatId, location.pathname, selectedModel, selectedDataset, models, datasets]);

  const continueChain = useCallback(async () => {
    if (!pendingChain || !dataRoot || pendingChain.steps.length === 0) return;
    const continuationPrompt = buildChainContinuationPrompt(pendingChain);
    setPendingChain(null);
    try {
      await runAgentCommand(dataRoot, {
        action: "cancel_chain",
        chat_id: pendingChain.chatId,
      });
    } catch {}
    await sendMessage(continuationPrompt);
  }, [pendingChain, dataRoot, sendMessage]);

  useEffect(() => {
    if (!pendingChain?.jobComplete || pendingChain.jobState !== "completed" || !pendingChain.autoContinueAt) {
      return;
    }
    if (isLoading) return;

    // Give the user a brief chance to see the artifact preview before the
    // agent resumes the remaining plan automatically.
    const waitMs = Math.max(pendingChain.autoContinueAt - Date.now(), 0);
    const timeoutId = window.setTimeout(() => {
      void continueChain();
    }, waitMs);
    return () => window.clearTimeout(timeoutId);
  }, [continueChain, isLoading, pendingChain]);

  const cancelChain = useCallback(async () => {
    const chatId = pendingChain?.chatId ?? activeChatId;
    setPendingChain(null);
    try {
      await runAgentCommand(dataRoot, { action: "cancel_chain", chat_id: chatId || undefined });
    } catch {}
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "Chain cancelled." },
    ]);
  }, [activeChatId, dataRoot, pendingChain?.chatId]);

  const createChat = useCallback(async () => {
    try {
      const res = await runAgentCommand(dataRoot, { action: "create_chat" });
      applyLoadedChatState(res);
    } catch {}
  }, [dataRoot]);

  const switchChat = useCallback(async (chatId: string) => {
    if (!dataRoot || isLoading || chatId === activeChatId) return;
    try {
      const res = await runAgentCommand(dataRoot, { action: "load", chat_id: chatId });
      applyLoadedChatState(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [activeChatId, dataRoot, isLoading]);

  const deleteChat = useCallback(async (chatId: string) => {
    if (!dataRoot || isLoading) return;
    try {
      const res = await runAgentCommand(dataRoot, { action: "delete_chat", chat_id: chatId });
      applyLoadedChatState(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [dataRoot, isLoading]);

  const searchChats = useCallback(async (query: string) => {
    if (!dataRoot) return;
    try {
      const res = await runAgentCommand(dataRoot, { action: "search_chats", query });
      const nextChats = readChatSummaries(res.chats);
      setChats(nextChats);
    } catch {}
  }, [dataRoot]);

  const clearConversation = createChat;

  return {
    activeChatId,
    chats,
    messages,
    currentTrace,
    isLoading,
    error,
    sendMessage,
    clearConversation,
    createChat,
    switchChat,
    deleteChat,
    searchChats,
    pendingChain,
    continueChain,
    cancelChain,
  };
}

function readAgentMessages(value: unknown): AgentMessage[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.flatMap((item) => {
    const message = readAgentMessage(item);
    return message ? [message] : [];
  });
}

function readAgentMessage(value: unknown): AgentMessage | null {
  if (!isRecord(value)) {
    return null;
  }
  const role = value.role;
  const content = value.content;
  if ((role !== "assistant" && role !== "user") || typeof content !== "string") {
    return null;
  }
  return {
    role,
    content,
    toolsUsed: readStringArray(value.tools_used ?? value.toolsUsed),
    artifact: isAgentJobPreview(value.artifact) ? value.artifact : undefined,
    workspaceDirective: isWorkspaceDirective(value.workspaceDirective)
      ? value.workspaceDirective
      : undefined,
  };
}

function readChatSummaries(value: unknown): AgentChatSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.flatMap((item) => {
    const summary = readChatSummary(item);
    return summary ? [summary] : [];
  });
}

function readChatSummary(value: unknown): AgentChatSummary | null {
  if (!isRecord(value)) {
    return null;
  }
  const id = value.id;
  const title = value.title;
  const preview = value.preview;
  const createdAt = value.createdAt;
  const updatedAt = value.updatedAt;
  const messageCount = value.messageCount;
  if (
    typeof id !== "string"
    || typeof title !== "string"
    || typeof preview !== "string"
    || typeof createdAt !== "string"
    || typeof updatedAt !== "string"
    || typeof messageCount !== "number"
  ) {
    return null;
  }
  return { id, title, preview, createdAt, updatedAt, messageCount };
}

function readPendingChain(value: unknown, chatId: string | null): PendingChain | null {
  if (!chatId || !isRecord(value)) {
    return null;
  }
  const jobId = value.waiting_on_job_id;
  const steps = value.remaining_steps;
  if (typeof jobId !== "string" || !Array.isArray(steps)) {
    return null;
  }
  return {
    chatId,
    jobId,
    steps: steps.filter((step): step is string => typeof step === "string"),
    jobComplete: false,
    jobState: null,
    jobModelPath: null,
    jobModelName: null,
    autoContinueAt: null,
  };
}

function isAgentJobPreview(value: unknown): value is AgentJobPreview {
  if (!isRecord(value)) {
    return false;
  }
  return (
    (value.kind === "training" || value.kind === "eval" || value.kind === "interp")
    && typeof value.jobId === "string"
    && typeof value.title === "string"
  );
}

function readStringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const strings = value.filter((item): item is string => typeof item === "string");
  return strings.length > 0 ? strings : undefined;
}

function isWorkspaceDirective(value: unknown): value is AgentWorkspaceDirective {
  if (!isRecord(value)) {
    return false;
  }
  return isWorkspaceMode(value.mode)
    && Array.isArray(value.cards)
    && value.cards.every((card) => typeof card === "string");
}

function isWorkspaceMode(value: unknown): value is AgentWorkspaceDirective["mode"] {
  return value === "auto"
    || value === "focus"
    || value === "compare"
    || value === "board"
    || value === "plan";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
