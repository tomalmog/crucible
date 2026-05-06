import type { ReactNode } from "react";
import "./agentWorkspaceCards.css";

export type AgentWorkspaceCardVariant = "default" | "primary" | "trace";

export interface AgentWorkspaceCardProps {
  label: string;
  title: string;
  headerAside?: ReactNode;
  summary?: string;
  emptyState?: string;
  variant?: AgentWorkspaceCardVariant;
  className?: string;
  children?: ReactNode;
}

const VARIANT_CLASS_NAMES: Record<AgentWorkspaceCardVariant, string | null> = {
  default: null,
  primary: "build-workspace-card-primary",
  trace: "build-workspace-card-trace",
};

export function AgentWorkspaceCard({
  label,
  title,
  headerAside,
  summary,
  emptyState,
  variant = "default",
  className,
  children,
}: AgentWorkspaceCardProps): ReactNode {
  const cardClassName = buildClassName([
    "build-workspace-card",
    "agent-workspace-card",
    `agent-workspace-card--${variant}`,
    VARIANT_CLASS_NAMES[variant],
    className,
  ]);
  const hasBodyContent = children != null;

  return (
    <section className={cardClassName}>
      <div className="build-workspace-card-header">
        <div>
          <span className="build-workspace-label">{label}</span>
          <strong>{title}</strong>
        </div>
        {headerAside ?? null}
      </div>
      {summary && <p className="build-workspace-summary">{summary}</p>}
      {hasBodyContent ? children : emptyState ? <p className="build-workspace-empty">{emptyState}</p> : null}
    </section>
  );
}

function buildClassName(parts: ReadonlyArray<string | null | undefined>): string {
  return parts.filter(isDefinedClassName).join(" ");
}

function isDefinedClassName(value: string | null | undefined): value is string {
  return value != null && value.length > 0;
}
