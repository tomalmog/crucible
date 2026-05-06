import { createContext, type ReactNode, useContext } from "react";
import { type UseAgentChatReturn, useAgentChat } from "../hooks/useAgentChat";

const AgentChatContext = createContext<UseAgentChatReturn | null>(null);

export function AgentChatProvider({ children }: { children: ReactNode }): React.ReactNode {
  const value = useAgentChat();
  return <AgentChatContext.Provider value={value}>{children}</AgentChatContext.Provider>;
}

export function useAgentChatState(): UseAgentChatReturn {
  const context = useContext(AgentChatContext);
  if (!context) {
    throw new Error("useAgentChatState must be used inside AgentChatProvider");
  }
  return context;
}
