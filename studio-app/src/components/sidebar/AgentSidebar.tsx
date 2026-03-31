import { useEffect, useRef, useState } from "react";
import { Loader2, Plus, Send, X } from "lucide-react";
import { useAgentChat } from "../../hooks/useAgentChat";

interface AgentSidebarProps {
  onClose: () => void;
}

export function AgentSidebar({ onClose }: AgentSidebarProps): React.ReactNode {
  const { messages, isLoading, error, sendMessage, clearConversation } = useAgentChat();
  const [draft, setDraft] = useState("");
  const threadRef = useRef<HTMLDivElement>(null);

  // Auto-scroll on new messages
  useEffect(() => {
    if (threadRef.current) {
      threadRef.current.scrollTop = threadRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  function handleSubmit(e: React.FormEvent): void {
    e.preventDefault();
    if (!draft.trim() || isLoading) return;
    const text = draft;
    setDraft("");
    sendMessage(text);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>): void {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  }

  return (
    <aside className="agent-sidebar">
      <div className="agent-sidebar-header">
        <h3>AI Agent</h3>
        <div style={{ display: "flex", gap: 4 }}>
          <button
            className="btn btn-ghost btn-sm btn-icon"
            onClick={() => clearConversation()}
            title="New conversation"
          >
            <Plus size={14} />
          </button>
          <button
            className="btn btn-ghost btn-sm btn-icon"
            onClick={onClose}
            title="Close"
          >
            <X size={14} />
          </button>
        </div>
      </div>

      <div className="agent-chat-thread" ref={threadRef}>
        {messages.length === 0 && !isLoading && (
          <p className="agent-chat-empty">
            Ask me anything about your models, datasets, or training.
          </p>
        )}
        {messages.map((msg, i) => (
          <article key={i} className={`chat-message ${msg.role}`}>
            <header>{msg.role === "user" ? "You" : "Agent"}</header>
            <p>{msg.content}</p>
            {msg.toolsUsed && msg.toolsUsed.length > 0 && (
              <div className="agent-tool-badge">
                Used: {msg.toolsUsed.join(", ")}
              </div>
            )}
          </article>
        ))}
        {isLoading && (
          <div className="agent-loading">
            <Loader2 size={14} className="spin" /> Thinking...
          </div>
        )}
      </div>

      {error && (
        <div className="agent-error">
          {error}
        </div>
      )}

      <form className="agent-input-row" onSubmit={handleSubmit}>
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.currentTarget.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask the agent..."
          rows={2}
          disabled={isLoading}
        />
        <button
          type="submit"
          className="btn btn-primary btn-sm btn-icon"
          disabled={isLoading || !draft.trim()}
          title="Send"
        >
          <Send size={14} />
        </button>
      </form>
    </aside>
  );
}
