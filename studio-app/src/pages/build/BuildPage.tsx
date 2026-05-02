import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { CheckCircle2, Loader2, Send } from "lucide-react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useAgentChat } from "../../hooks/useAgentChat";

const EXAMPLE_PROMPTS = [
  "Summarize support tickets",
  "Fine-tune Llama-3",
  "Write SQL from English",
  "Classify urgent emails",
];

const TEXTAREA_MAX_HEIGHT = 240;

function useAutoGrow(value: string) {
  const ref = useRef<HTMLTextAreaElement>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (!value) {
      // Empty: clear inline height so CSS controls it (single line via line-height).
      el.style.height = "";
      return;
    }
    el.style.height = "auto";
    const next = Math.min(el.scrollHeight, TEXTAREA_MAX_HEIGHT);
    el.style.height = `${next}px`;
  }, [value]);
  return ref;
}

export function BuildPage(): React.ReactNode {
  const {
    messages,
    isLoading,
    error,
    sendMessage,
    pendingChain,
    continueChain,
    cancelChain,
  } = useAgentChat();
  const [draft, setDraft] = useState("");
  const threadRef = useRef<HTMLDivElement>(null);
  const historyIndexRef = useRef(-1);
  const heroTextareaRef = useAutoGrow(draft);
  const replyTextareaRef = useAutoGrow(draft);

  const userHistory = useMemo(
    () => messages.filter((m) => m.role === "user").map((m) => m.content),
    [messages],
  );

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
    if (e.key === "ArrowUp" && e.currentTarget.selectionStart === 0) {
      e.preventDefault();
      const nextIdx = historyIndexRef.current + 1;
      if (nextIdx < userHistory.length) {
        historyIndexRef.current = nextIdx;
        setDraft(userHistory[userHistory.length - 1 - nextIdx]);
      }
      return;
    }
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

  const isEmpty = messages.length === 0 && !isLoading;

  return (
    <div className={`build-page ${isEmpty ? "build-page-empty" : ""}`}>
      {isEmpty ? (
        <div className="build-hero">
          <h1 className="build-hero-title">What do you want to build?</h1>
          <p className="build-hero-subtitle">
            Crucible will train, fine-tune, evaluate, and ship it for you.
          </p>

          <form className="build-composer build-composer-hero" onSubmit={handleSubmit}>
            <textarea
              ref={heroTextareaRef}
              value={draft}
              onChange={(e) => setDraft(e.currentTarget.value)}
              onKeyDown={handleKeyDown}
              placeholder="Describe what you want to build..."
              rows={1}
              autoFocus
              disabled={isLoading}
            />
            <button
              type="submit"
              className="build-send-btn"
              disabled={isLoading || !draft.trim()}
              title="Send"
              aria-label="Send"
            >
              <Send size={16} />
            </button>
          </form>

          <div className="build-examples">
            {EXAMPLE_PROMPTS.map((prompt) => (
              <button
                key={prompt}
                type="button"
                className="build-example-chip"
                onClick={() => sendMessage(prompt)}
                disabled={isLoading}
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="build-conversation">
          <div className="build-thread" ref={threadRef}>
            {messages.map((msg, i) => (
              <article key={i} className={`build-message ${msg.role}`}>
                <header>{msg.role === "user" ? "You" : "Crucible"}</header>
                {msg.role === "user" ? (
                  <p>{msg.content}</p>
                ) : (
                  <div className="agent-markdown">
                    <Markdown remarkPlugins={[remarkGfm]}>{msg.content}</Markdown>
                  </div>
                )}
                {msg.toolsUsed && msg.toolsUsed.length > 0 && (
                  <div className="agent-tool-badge">Used: {msg.toolsUsed.join(", ")}</div>
                )}
                {msg.scriptUpdated && (
                  <div className="agent-tool-badge">Updated training script</div>
                )}
                {msg.navigatedTo && (
                  <div className="agent-tool-badge">Navigated to {msg.navigatedTo}</div>
                )}
              </article>
            ))}
            {isLoading && (
              <div className="agent-loading">
                <Loader2 size={14} className="spin" /> Thinking...
              </div>
            )}
          </div>

          {error && <div className="agent-error">{error}</div>}

          {pendingChain && (
            <div className="agent-chain-banner build-chain-banner">
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
                    <button className="btn btn-ghost btn-sm" onClick={cancelChain}>
                      Cancel
                    </button>
                  </div>
                </>
              ) : pendingChain.jobState === "completed" ? (
                <>
                  <div className="agent-chain-header agent-chain-ready">
                    <CheckCircle2 size={14} />
                    <span>Job completed — ready to continue</span>
                  </div>
                  <div className="agent-chain-steps">
                    <div className="agent-chain-step">
                      <span className="agent-chain-step-num">→</span>
                      {pendingChain.steps[0]}
                    </div>
                    {pendingChain.steps.length > 1 && (
                      <div className="agent-chain-step text-tertiary">
                        + {pendingChain.steps.length - 1} more step
                        {pendingChain.steps.length > 2 ? "s" : ""}
                      </div>
                    )}
                  </div>
                  <div className="agent-chain-actions">
                    <button
                      className="btn btn-primary btn-sm"
                      onClick={continueChain}
                      disabled={isLoading}
                    >
                      Continue
                    </button>
                    <button className="btn btn-ghost btn-sm" onClick={cancelChain}>
                      Cancel
                    </button>
                  </div>
                </>
              ) : null}
            </div>
          )}

          <form className="build-composer" onSubmit={handleSubmit}>
            <textarea
              ref={replyTextareaRef}
              value={draft}
              onChange={(e) => setDraft(e.currentTarget.value)}
              onKeyDown={handleKeyDown}
              placeholder="Reply to Crucible..."
              rows={1}
              disabled={isLoading}
            />
            <button
              type="submit"
              className="build-send-btn"
              disabled={isLoading || !draft.trim()}
              title="Send"
              aria-label="Send"
            >
              <Send size={16} />
            </button>
          </form>
        </div>
      )}
    </div>
  );
}
