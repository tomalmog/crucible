import { useLayoutEffect, useMemo, useRef, useState } from "react";
import { AgentChatHistory } from "../../components/shared/AgentChatHistory";
import { useAgentChatState } from "../../context/AgentChatContext";
import { BuildComposer } from "./BuildComposer";
import { BuildConversationPane } from "./BuildConversationPane";

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
    currentTrace,
    messages,
    isLoading,
    error,
    sendMessage,
    pendingChain,
    continueChain,
    cancelChain,
  } = useAgentChatState();
  const [draft, setDraft] = useState("");
  const historyIndexRef = useRef(-1);
  const heroTextareaRef = useAutoGrow(draft);
  const replyTextareaRef = useAutoGrow(draft);

  const userHistory = useMemo(
    () => messages.filter((m) => m.role === "user").map((m) => m.content),
    [messages],
  );

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
      <div className="build-agent-shell">
        <aside className="build-chat-rail">
          <div className="build-chat-rail-header">
            <span>Chats</span>
            <p>Search old runs or start fresh.</p>
          </div>
          <AgentChatHistory className="agent-chat-history-build" />
        </aside>

        <main className="build-agent-stage">
          {isEmpty ? (
            <div className="build-hero">
              <h1 className="build-hero-title">What do you want to build?</h1>
              <p className="build-hero-subtitle">
                Crucible will train, fine-tune, evaluate, and ship it for you.
              </p>

              <BuildComposer
                draft={draft}
                disabled={isLoading}
                isHero
                placeholder="Describe what you want to build..."
                textareaRef={heroTextareaRef}
                onDraftChange={setDraft}
                onKeyDown={handleKeyDown}
                onSubmit={handleSubmit}
              />

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
            <div className="build-session">
              <BuildConversationPane
                currentTrace={currentTrace}
                draft={draft}
                error={error}
                isLoading={isLoading}
                messages={messages}
                pendingChain={pendingChain}
                replyTextareaRef={replyTextareaRef}
                onCancelChain={() => void cancelChain()}
                onContinueChain={() => void continueChain()}
                onDraftChange={setDraft}
                onKeyDown={handleKeyDown}
                onSubmit={handleSubmit}
              />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
