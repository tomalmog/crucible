import { ArrowRight, CheckCircle2, CircleDot, Loader2 } from "lucide-react";

interface CompactFormCardProps {
  actionLabel: string;
  children: React.ReactNode;
  description: string;
  missing: string[];
  title: string;
  className?: string;
  error?: string | null;
  isRunning?: boolean;
  onSubmit?: () => void;
  output?: string;
  readyLabel?: string;
  runningLabel?: string;
}

interface CompactFieldProps {
  children: React.ReactNode;
  hint?: string;
  label: string;
  required?: boolean;
}

interface ToggleOption<T extends string> {
  label: string;
  value: T;
}

interface CompactToggleGroupProps<T extends string> {
  label: string;
  onChange: (value: T) => void;
  options: readonly ToggleOption<T>[];
  value: T;
}

interface CompactInfoBannerProps {
  children: React.ReactNode;
}

export function CompactFormCard({
  actionLabel,
  children,
  description,
  missing,
  title,
  className,
  error,
  isRunning = false,
  onSubmit,
  output,
  readyLabel = "Ready to run",
  runningLabel = "Submitting...",
}: CompactFormCardProps): React.ReactNode {
  const canRun = missing.length === 0;
  const statusText = canRun ? readyLabel : `Missing ${missing.join(", ")}`;
  const cardClassName = className
    ? `platform-form-card ${className}`
    : "platform-form-card";

  return (
    <section className={cardClassName} aria-label={`${title} compact form`}>
      <header className="platform-form-header">
        <div>
          <h3>{title}</h3>
        </div>
        <p>{description}</p>
      </header>
      <div className="platform-form-body">{children}</div>
      <footer className="platform-form-footer">
        <div className={canRun ? "ready" : ""}>
          {canRun ? <CheckCircle2 size={16} /> : <CircleDot size={16} />}
          <span>{statusText}</span>
        </div>
        <button
          className="platform-form-run-button"
          disabled={isRunning || !canRun}
          onClick={onSubmit}
          type="button"
        >
          {isRunning && <Loader2 className="spin" size={16} />}
          {isRunning ? runningLabel : actionLabel}
          {!isRunning && <ArrowRight size={16} />}
        </button>
      </footer>
      {error && <p className="platform-form-error">{error}</p>}
      {output && <pre className="platform-form-console">{output}</pre>}
    </section>
  );
}

export function CompactField({
  children,
  hint,
  label,
  required = false,
}: CompactFieldProps): React.ReactNode {
  return (
    <div className="platform-form-field">
      <span className="platform-form-label">
        <span>
          {label}
          {required && " *"}
        </span>
        {hint && <span className="platform-form-hint">{hint}</span>}
      </span>
      <div className="platform-form-control">{children}</div>
    </div>
  );
}

export function CompactInlineField({
  children,
  hint,
  label,
  required = false,
}: CompactFieldProps): React.ReactNode {
  return (
    <CompactField hint={hint} label={label} required={required}>
      {children}
    </CompactField>
  );
}

export function CompactToggleGroup<T extends string>({
  label,
  onChange,
  options,
  value,
}: CompactToggleGroupProps<T>): React.ReactNode {
  return (
    <div className="platform-form-toggle-group" aria-label={label} role="tablist">
      {options.map((option) => (
        <button
          key={option.value}
          className={
            option.value === value
              ? "platform-form-toggle-button active"
              : "platform-form-toggle-button"
          }
          aria-selected={option.value === value}
          onClick={() => onChange(option.value)}
          role="tab"
          type="button"
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function CompactInfoBanner({
  children,
}: CompactInfoBannerProps): React.ReactNode {
  return <div className="platform-form-note">{children}</div>;
}
