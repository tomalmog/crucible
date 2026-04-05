import { useEffect, useMemo, useRef, useState } from "react";
import { Loader2, Plus, Send, X } from "lucide-react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useAgentChat } from "../../hooks/useAgentChat";

interface AgentSidebarProps {
  onClose: () => void;
}

export function AgentSidebar({ onClose }: AgentSidebarProps): React.ReactNode {
  const { messages, isLoading, error, sendMessage, clearConversation } = useAgentChat();
  const [draft, setDraft] = useState("");
  const threadRef = useRef<HTMLDivElement>(null);
  const historyIndexRef = useRef(-1);

  // Extract user messages for up/down arrow history
  const userHistory = useMemo(
    () => messages.filter((m) => m.role === "user").map((m) => m.content),
    [messages],
  );

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
    historyIndexRef.current = -1;
    sendMessage(text);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>): void {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
      return;
    }
    // Up arrow at start of input: cycle through previous user messages
    if (e.key === "ArrowUp" && e.currentTarget.selectionStart === 0) {
      e.preventDefault();
      const nextIdx = historyIndexRef.current + 1;
      if (nextIdx < userHistory.length) {
        historyIndexRef.current = nextIdx;
        setDraft(userHistory[userHistory.length - 1 - nextIdx]);
      }
      return;
    }
    // Down arrow: cycle forward
    if (e.key === "ArrowDown" && historyIndexRef.current >= 0) {
      e.preventDefault();
      const nextIdx = historyIndexRef.current - 1;
      if (nextIdx < 0) {
        historyIndexRef.current = -1;
        setDraft("");
      } else {
        historyIndexRef.current = nextIdx;
        setDraft(userHistory[userHistory.length - 1 - nextIdx]);
      }
    }
  }

  return (
    <aside className="agent-sidebar">
      <div className="agent-sidebar-inner">
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
              {msg.role === "user"
                ? <p>{msg.content}</p>
                : <div className="agent-markdown"><Markdown remarkPlugins={[remarkGfm]}>{msg.content}</Markdown></div>
              }
              {msg.toolsUsed && msg.toolsUsed.length > 0 && (
                <div className="agent-tool-badge">
                  Used: {msg.toolsUsed.join(", ")}
                </div>
              )}
              {msg.scriptUpdated && (
                <div className="agent-tool-badge">Updated training script</div>
              )}
              {msg.navigatedTo && (
                <div className="agent-tool-badge">
                  Navigated to {msg.navigatedTo}
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
      </div>
    </aside>
  );
}
