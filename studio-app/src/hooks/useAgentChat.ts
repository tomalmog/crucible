import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router";
import { useCrucible } from "../context/CrucibleContext";
import { useScript } from "../context/ScriptContext";
import { startCrucibleCommand, getCrucibleCommandStatus } from "../api/studioApi";

const API_KEY_STORAGE = "crucible_anthropic_api_key";
const PROVIDER_STORAGE = "crucible_agent_provider";
const OLLAMA_MODEL_STORAGE = "crucible_agent_ollama_model";
const OLLAMA_URL_STORAGE = "crucible_agent_ollama_url";
const GEMINI_MODEL_STORAGE = "crucible_agent_gemini_model";
const GEMINI_API_KEY_STORAGE = "crucible_gemini_api_key";
const POLL_MS = 500;

export interface AgentMessage {
  role: "user" | "assistant";
  content: string;
  toolsUsed?: string[];
  scriptUpdated?: boolean;
  navigatedTo?: string;
}

export interface UseAgentChatReturn {
  messages: AgentMessage[];
  isLoading: boolean;
  error: string | null;
  sendMessage: (text: string) => Promise<void>;
  clearConversation: () => Promise<void>;
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
  const hasLoaded = useRef(false);

  // Load conversation history on mount
  useEffect(() => {
    if (hasLoaded.current || !dataRoot) return;
    hasLoaded.current = true;
    runAgentCommand(dataRoot, { action: "load" })
      .then((res) => {
        const msgs = res.messages as AgentMessage[] | undefined;
        if (msgs) setMessages(msgs);
      })
      .catch(() => { /* no history yet */ });
  }, [dataRoot]);

  const sendMessage = useCallback(async (text: string) => {
    if (isLoading || !text.trim()) return;

    const apiKey = localStorage.getItem(API_KEY_STORAGE) || "";
    setError(null);
    setIsLoading(true);
    setMessages((prev) => [...prev, { role: "user", content: text }]);

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
        setMessages((prev) => prev.slice(0, -1));
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
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setIsLoading(false);
    }
  }, [isLoading, dataRoot, location.pathname, selectedModel, selectedDataset, models, datasets]);

  const clearConversation = useCallback(async () => {
    try {
      await runAgentCommand(dataRoot, { action: "clear" });
      setMessages([]);
      setError(null);
    } catch { /* best effort */ }
  }, [dataRoot]);

  return { messages, isLoading, error, sendMessage, clearConversation };
}
