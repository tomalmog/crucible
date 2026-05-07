import type { FormEvent } from "react";

interface ChatInputRowProps {
  draftMessage: string;
  onDraftMessageChange: (value: string) => void;
  canSend: boolean;
  isSending: boolean;
  onSubmit: (event: FormEvent) => void;
  onClear: () => void;
}

export function ChatInputRow(props: ChatInputRowProps) {
  return (
    <form className="chat-input-row" onSubmit={props.onSubmit}>
      <input
        value={props.draftMessage}
        onChange={(event) => props.onDraftMessageChange(event.currentTarget.value)}
        placeholder="Type a prompt..."
      />
      <button className="btn btn-primary" type="submit" disabled={!props.canSend}>
        {props.isSending ? "Sending..." : "Send"}
      </button>
      <button className="btn" type="button" onClick={props.onClear}>
        Clear
      </button>
    </form>
  );
}
