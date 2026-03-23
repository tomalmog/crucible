import { ReactNode } from "react";

interface CommandFormPanelProps {
  /** Panel heading shown at top. */
  title: string;
  /** List of missing required field names — shown in an error alert when non-empty. */
  missing: string[];
  /** Whether the underlying command is currently executing. */
  isRunning: boolean;
  /** Label for the submit button in its idle state. */
  submitLabel: string;
  /** Label for the submit button while the command is running. */
  runningLabel: string;
  /** Called when the submit button is clicked. */
  onSubmit: () => void;
  /** Optional error message shown below the button. */
  error?: string | null;
  /** Optional raw console output shown below the button. */
  output?: string;
  /** Form field content rendered between the title and the action area. */
  children: ReactNode;
}

export function CommandFormPanel({
  title,
  missing,
  isRunning,
  submitLabel,
  runningLabel,
  onSubmit,
  error,
  output,
  children,
}: CommandFormPanelProps) {
  const canStart = missing.length === 0;

  return (
    <div className="panel stack-lg">
      <h3>{title}</h3>
      {children}
      <button
        className="btn btn-primary"
        onClick={onSubmit}
        disabled={isRunning || !canStart}
      >
        {isRunning ? runningLabel : submitLabel}
      </button>
      {error && <p className="error-text">{error}</p>}
      {output && <pre className="console">{output}</pre>}
    </div>
  );
}
