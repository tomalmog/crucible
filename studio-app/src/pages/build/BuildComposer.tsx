import type { FormEvent, KeyboardEvent, RefObject } from "react";
import { Send } from "lucide-react";

interface BuildComposerProps {
  draft: string;
  disabled: boolean;
  isHero?: boolean;
  placeholder: string;
  textareaRef: RefObject<HTMLTextAreaElement | null>;
  onDraftChange: (value: string) => void;
  onKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
  onSubmit: (event: FormEvent) => void;
}

export function BuildComposer({
  draft,
  disabled,
  isHero = false,
  placeholder,
  textareaRef,
  onDraftChange,
  onKeyDown,
  onSubmit,
}: BuildComposerProps): React.ReactNode {
  return (
    <form className={`build-composer${isHero ? " build-composer-hero" : ""}`} onSubmit={onSubmit}>
      <textarea
        ref={textareaRef}
        value={draft}
        onChange={(event) => onDraftChange(event.currentTarget.value)}
        onKeyDown={onKeyDown}
        placeholder={placeholder}
        rows={1}
        autoFocus={isHero}
        disabled={disabled}
      />
      <button
        type="submit"
        className="build-send-btn"
        disabled={disabled || !draft.trim()}
        title="Send"
        aria-label="Send"
      >
        <Send size={16} />
      </button>
    </form>
  );
}
