import { useEffect, useMemo, useRef, useState } from "react";
import { CheckCircle2, Loader2, Plus, Send, X } from "lucide-react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useAgentChatState } from "../../context/AgentChatContext";
import { AgentChatHistory } from "../shared/AgentChatHistory";
import { AgentEventTimeline } from "../shared/AgentEventTimeline";
import { AgentJobPreviewCard } from "../shared/AgentJobPreviewCard";

interface AgentSidebarProps {
  onClose: () => void;
}

export function AgentSidebar({ onClose }: AgentSidebarProps): React.ReactNode {
  const {
    messages,
    currentTrace,
    isLoading,
    error,
    sendMessage,
    createChat,
    pendingChain,
    continueChain,
    cancelChain,
  } = useAgentChatState();
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
          <div className="agent-sidebar-actions">
            <button
              className="btn btn-ghost btn-sm btn-icon"
              onClick={() => void createChat()}
              title="New chat"
              disabled={isLoading}
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

        <AgentChatHistory />

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
              {msg.artifact && <AgentJobPreviewCard artifact={msg.artifact} />}
              {msg.toolsUsed && msg.toolsUsed.length > 0 && (
                <div className="agent-tool-badge">
                  Used: {msg.toolsUsed.join(", ")}
                </div>
              )}
              {msg.navigatedTo && (
                <div className="agent-tool-badge">
                  Navigated to {msg.navigatedTo}
                </div>
              )}
              {msg.trace && msg.trace.length > 0 && (
                <AgentEventTimeline events={msg.trace} />
              )}
            </article>
          ))}
          {isLoading && (
            <>
              <div className="agent-loading">
                <Loader2 size={14} className="spin" /> Thinking...
              </div>
              <div className="agent-live-trace">
                <AgentEventTimeline events={currentTrace} />
              </div>
            </>
          )}
        </div>

        {error && (
          <div className="agent-error">
            {error}
          </div>
        )}

        {pendingChain && (
          <div className="agent-chain-banner">
            {!pendingChain.jobComplete ? (
              <>
                <div className="agent-chain-header">
                  <Loader2 size={14} className="spin" />
                  <span>Waiting for job to complete...</span>
                </div>
                <div className="agent-chain-steps">
                  {pendingChain.steps.map((step, i) => (
                    <div key={i} className="agent-chain-step">
                      <span className="agent-chain-step-num">{i + 1}</span>
                      {step}
                    </div>
                  ))}
                </div>
                <div className="agent-chain-actions">
                  <button className="btn btn-ghost btn-sm" onClick={cancelChain}>Cancel</button>
                </div>
              </>
            ) : pendingChain.jobState === "completed" ? (
              <>
                <div className="agent-chain-header agent-chain-ready">
                  <CheckCircle2 size={14} />
                  <span>Job completed — continuing automatically</span>
                </div>
                <div className="agent-chain-steps">
                  <div className="agent-chain-step">
                    <span className="agent-chain-step-num">→</span>
                    {pendingChain.steps[0]}
                  </div>
                  {pendingChain.steps.length > 1 && (
                    <div className="agent-chain-step text-tertiary">
                      + {pendingChain.steps.length - 1} more step{pendingChain.steps.length > 2 ? "s" : ""}
                    </div>
                  )}
                </div>
                <div className="agent-chain-actions">
                  <button className="btn btn-primary btn-sm" onClick={continueChain} disabled={isLoading}>Continue now</button>
                  <button className="btn btn-ghost btn-sm" onClick={cancelChain}>Cancel</button>
                </div>
              </>
            ) : null}
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
