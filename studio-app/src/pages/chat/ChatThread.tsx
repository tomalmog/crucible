import type { RefObject } from "react";
import type { ChatMessage } from "./chatPersistence";

interface ChatThreadProps {
  messages: ChatMessage[];
  isSending: boolean;
  statusLine: string;
  chatError: string | null;
  threadRef: RefObject<HTMLDivElement | null>;
}

export function ChatThread(props: ChatThreadProps) {
  return (
    <>
      <div className="chat-thread" ref={props.threadRef}>
        {props.messages.length === 0 ? (
          <p className="chat-empty">Send a message to evaluate your trained model.</p>
        ) : (
          props.messages.map((message, index) => (
            <article key={`${message.role}-${index}`} className={`chat-message ${message.role}`}>
              <header>{message.role === "user" ? "You" : "Model"}</header>
              <p>{message.content}</p>
            </article>
          ))
        )}
      </div>

      {props.isSending && props.statusLine && (
        <p className="chat-status">{props.statusLine}</p>
      )}
      {props.chatError && <p className="chat-error">{props.chatError}</p>}
    </>
  );
}
