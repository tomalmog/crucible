import { useEffect, useState } from "react";
import { Plus, Search, Trash2 } from "lucide-react";
import { useAgentChatState } from "../../context/AgentChatContext";

interface AgentChatHistoryProps {
  className?: string;
}

export function AgentChatHistory({ className = "" }: AgentChatHistoryProps): React.ReactNode {
  const {
    activeChatId,
    chats,
    isLoading,
    createChat,
    switchChat,
    deleteChat,
    searchChats,
  } = useAgentChatState();
  const [searchTerm, setSearchTerm] = useState("");

  // React to search text changes so chat filtering feels instant without
  // spawning a CLI search command on every keystroke.
  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      void searchChats(searchTerm);
    }, 150);
    return () => window.clearTimeout(timeoutId);
  }, [searchChats, searchTerm]);

  const historyClassName = className
    ? `agent-chat-history ${className}`
    : "agent-chat-history";

  return (
    <section className={historyClassName}>
      <div className="agent-chat-search">
        <Search size={13} />
        <input
          value={searchTerm}
          onChange={(event) => setSearchTerm(event.currentTarget.value)}
          placeholder="Search chats..."
          aria-label="Search chats"
        />
      </div>
      <button
        type="button"
        className="agent-new-chat-btn"
        onClick={() => void createChat()}
        disabled={isLoading}
      >
        <Plus size={13} />
        New chat
      </button>
      <div className="agent-chat-history-list">
        {chats.map((chat) => (
          <div
            key={chat.id}
            className={chat.id === activeChatId
              ? "agent-chat-history-row active"
              : "agent-chat-history-row"}
          >
            <button
              type="button"
              className="agent-chat-history-item"
              onClick={() => void switchChat(chat.id)}
              disabled={isLoading}
            >
              <span className="agent-chat-history-title">{chat.title}</span>
              <span className="agent-chat-history-meta">
                {chat.preview || "Empty chat"} · {formatChatDate(chat.updatedAt)}
              </span>
            </button>
            <button
              type="button"
              className="agent-chat-delete"
              title={`Delete ${chat.title}`}
              aria-label={`Delete ${chat.title}`}
              disabled={isLoading}
              onClick={(event) => {
                event.stopPropagation();
                void deleteChat(chat.id);
              }}
            >
              <Trash2 size={12} />
            </button>
          </div>
        ))}
      </div>
    </section>
  );
}

function formatChatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}
