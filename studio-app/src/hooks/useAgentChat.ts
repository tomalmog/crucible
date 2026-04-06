import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router";
import { useCrucible } from "../context/CrucibleContext";
import { useScript } from "../context/ScriptContext";
import { startCrucibleCommand, getCrucibleCommandStatus } from "../api/studioApi";
import { getJob } from "../api/jobsApi";
import { TERMINAL_JOB_STATES } from "../types/jobs";

const API_KEY_STORAGE = "crucible_anthropic_api_key";
const PROVIDER_STORAGE = "crucible_agent_provider";
const OLLAMA_MODEL_STORAGE = "crucible_agent_ollama_model";
const OLLAMA_URL_STORAGE = "crucible_agent_ollama_url";
const GEMINI_MODEL_STORAGE = "crucible_agent_gemini_model";
const GEMINI_API_KEY_STORAGE = "crucible_gemini_api_key";
const POLL_MS = 500;
const CHAIN_POLL_MS = 3000;

export interface AgentMessage {
  role: "user" | "assistant";
  content: string;
  toolsUsed?: string[];
  scriptUpdated?: boolean;
  navigatedTo?: string;
}

export interface PendingChain {
  jobId: string;
  steps: string[];
  jobComplete: boolean;
  jobState: string | null;
  jobModelPath: string | null;
  jobModelName: string | null;
}

export interface UseAgentChatReturn {
  messages: AgentMessage[];
  isLoading: boolean;
  error: string | null;
  sendMessage: (text: string) => Promise<void>;
  clearConversation: () => Promise<void>;
  pendingChain: PendingChain | null;
  continueChain: () => Promise<void>;
  cancelChain: () => Promise<void>;
}

async function runAgentCommand(
  dataRoot: string,
  payload: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  // config_json (4th arg) is written to .crucible/agent/_payload.json by Rust
  const { task_id } = await startCrucibleCommand(
    dataRoot,
    ["agent-chat", "--payload-file", "placeholder"],
    "agent-chat",
    payload,
  );
  while (true) {
    const status = await getCrucibleCommandStatus(task_id);
    if (status.status !== "running") {
      if (status.status === "failed") {
        const errMsg = status.stderr?.trim() || "Agent command failed";
        throw new Error(errMsg);
      }
      const stdout = status.stdout?.trim() || "{}";
      try {
        return JSON.parse(stdout) as Record<string, unknown>;
      } catch {
        throw new Error(`Invalid agent response: ${stdout.slice(0, 200)}`);
      }
    }
    await new Promise((r) => setTimeout(r, POLL_MS));
  }
}

export function useAgentChat(): UseAgentChatReturn {
  const { dataRoot, models, datasets, selectedModel, selectedDataset } = useCrucible();
  const location = useLocation();
  const navigate = useNavigate();
  const { registration: scriptReg } = useScript();
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingChain, setPendingChain] = useState<PendingChain | null>(null);
  const hasLoaded = useRef(false);

  // Load conversation history + chain state on mount
  useEffect(() => {
    if (hasLoaded.current || !dataRoot) return;
    hasLoaded.current = true;
    runAgentCommand(dataRoot, { action: "load" })
      .then((res) => {
        const msgs = res.messages as AgentMessage[] | undefined;
        if (msgs) setMessages(msgs);
      })
      .catch(() => { /* no history yet */ });
    runAgentCommand(dataRoot, { action: "load_chain" })
      .then((res) => {
        const chain = res.chain as { waiting_on_job_id: string; remaining_steps: string[] } | null;
        if (chain) {
          setPendingChain({
            jobId: chain.waiting_on_job_id,
            steps: chain.remaining_steps,
            jobComplete: false,
            jobState: null,
            jobModelPath: null,
            jobModelName: null,
          });
        }
      })
      .catch(() => {});
  }, [dataRoot]);

  // Poll the chain's job for completion
  useEffect(() => {
    if (!pendingChain || pendingChain.jobComplete || !dataRoot) return;
    const interval = setInterval(async () => {
      try {
        const job = await getJob(dataRoot, pendingChain.jobId);
        if (TERMINAL_JOB_STATES.has(job.state)) {
          if (job.state !== "completed") {
            // Job failed or was cancelled — auto-cancel the chain
            setPendingChain(null);
            runAgentCommand(dataRoot, { action: "cancel_chain" }).catch(() => {});
            const errorDetail = job.errorMessage ? `: ${job.errorMessage}` : "";
            setMessages((prev) => [
              ...prev,
              {
                role: "assistant",
                content: `The job ${pendingChain.jobId} **${job.state}**${errorDetail}. The remaining chain steps have been cancelled.\n\nYou can fix the issue and ask me to retry.`,
              },
            ]);
          } else {
            setPendingChain((prev) => prev ? {
              ...prev,
              jobComplete: true,
              jobState: job.state,
              jobModelPath: job.modelPath || null,
              jobModelName: job.modelName || null,
            } : null);
          }
        }
      } catch {
        // Job might not exist yet or was deleted
      }
    }, CHAIN_POLL_MS);
    return () => clearInterval(interval);
  }, [dataRoot, pendingChain?.jobId, pendingChain?.jobComplete]);

  const sendMessage = useCallback(async (text: string) => {
    if (isLoading || !text.trim()) return;

    const apiKey = localStorage.getItem(API_KEY_STORAGE) || "";
    setError(null);
    setIsLoading(true);
    // Don't show auto-continue messages as regular user messages
    const isChainContinue = text.startsWith("[Chain continuation]");
    if (!isChainContinue) {
      setMessages((prev) => [...prev, { role: "user", content: text }]);
    }

    try {
      // Include training script context only when the Code tab is active
      const scriptContext = scriptReg && scriptReg.viewTabRef.current === "code"
        ? { trainingScript: scriptReg.contentRef.current, trainingMethod: scriptReg.method }
        : null;

      const context = {
        currentPage: location.pathname,
        selectedModel: selectedModel?.modelName || null,
        selectedDataset: selectedDataset || null,
        modelNames: models.map((m) => m.modelName).slice(0, 20),
        modelPaths: models.slice(0, 20).reduce<Record<string, string>>((acc, m) => {
          if (m.modelPath) acc[m.modelName] = m.modelPath;
          if (m.remotePath) acc[`${m.modelName} (remote)`] = m.remotePath;
          return acc;
        }, {}),
        datasetNames: datasets.map((d) => d.name).slice(0, 20),
        ...(scriptContext ? { script: scriptContext } : {}),
      };

      const provider = localStorage.getItem(PROVIDER_STORAGE) || "anthropic";
      const effectiveApiKey = provider === "gemini"
        ? (localStorage.getItem(GEMINI_API_KEY_STORAGE) || "")
        : apiKey;
      const res = await runAgentCommand(dataRoot, {
        action: "chat",
        message: text,
        context,
        api_key: effectiveApiKey,
        provider,
        model: provider === "ollama" ? (localStorage.getItem(OLLAMA_MODEL_STORAGE) || "")
             : provider === "gemini" ? (localStorage.getItem(GEMINI_MODEL_STORAGE) || "")
             : "",
        ollama_url: provider === "ollama" ? (localStorage.getItem(OLLAMA_URL_STORAGE) || "") : "",
      });

      if (res.error) {
        setError(res.error as string);
        if (!isChainContinue) {
          setMessages((prev) => prev.slice(0, -1));
        }
      } else if (res.content) {
        const toolsUsed = res.tools_used as string[] | undefined;
        const didUpdateScript = !!(res.script_update && scriptReg?.setContent);
        if (didUpdateScript) {
          scriptReg!.setContent(res.script_update as string);
        }
        const navigatedTo = res.navigate_to as string | undefined;
        if (navigatedTo) {
          navigate(navigatedTo);
        }
        // Handle pending chain from response
        const chainData = res.pending_chain as { job_id: string; steps: string[] } | undefined;
        if (chainData) {
          setPendingChain({
            jobId: chainData.job_id,
            steps: chainData.steps,
            jobComplete: false,
            jobState: null,
            jobModelPath: null,
            jobModelName: null,
          });
        }
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: res.content as string,
            toolsUsed: toolsUsed?.length ? toolsUsed : undefined,
            scriptUpdated: didUpdateScript || undefined,
            navigatedTo: navigatedTo || undefined,
          },
        ]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      if (!isChainContinue) {
        setMessages((prev) => prev.slice(0, -1));
      }
    } finally {
      setIsLoading(false);
    }
  }, [isLoading, dataRoot, location.pathname, selectedModel, selectedDataset, models, datasets]);

  const continueChain = useCallback(async () => {
    if (!pendingChain || !dataRoot) return;
    const chain = pendingChain;

    // Build context message with job result info
    const parts = [
      `[Chain continuation] Job ${chain.jobId} finished with state: ${chain.jobState}.`,
    ];
    if (chain.jobModelPath) parts.push(`Output model path: ${chain.jobModelPath}`);
    if (chain.jobModelName) parts.push(`Registered model name: ${chain.jobModelName}`);
    parts.push(`\nPlease proceed with the next step: ${chain.steps[0]}`);
    if (chain.steps.length > 1) {
      parts.push(`\nRemaining steps after this: ${chain.steps.slice(1).join("; ")}`);
    }

    // Clear chain before sending — agent may create a new one
    setPendingChain(null);
    try {
      await runAgentCommand(dataRoot, { action: "cancel_chain" });
    } catch { /* best effort */ }

    await sendMessage(parts.join("\n"));
  }, [pendingChain, dataRoot, sendMessage]);

  const cancelChain = useCallback(async () => {
    setPendingChain(null);
    try {
      await runAgentCommand(dataRoot, { action: "cancel_chain" });
    } catch { /* best effort */ }
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "Chain cancelled." },
    ]);
  }, [dataRoot]);

  const clearConversation = useCallback(async () => {
    try {
      await runAgentCommand(dataRoot, { action: "clear" });
      setMessages([]);
      setError(null);
      setPendingChain(null);
    } catch { /* best effort */ }
  }, [dataRoot]);

  return { messages, isLoading, error, sendMessage, clearConversation, pendingChain, continueChain, cancelChain };
}
