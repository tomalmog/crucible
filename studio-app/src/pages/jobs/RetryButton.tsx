import { useNavigate } from "react-router";
import { ArrowLeft, RotateCcw } from "lucide-react";

const CONFIG_PAGE_ROUTES: Record<string, string> = {
  training: "/training",
  benchmarks: "/benchmarks",
  interpretability: "/interpretability",
};

export function hasRetryConfig(config: Record<string, unknown>): boolean {
  return Object.keys(config).length > 0 && typeof config.page === "string"
    && config.page in CONFIG_PAGE_ROUTES;
}

export function RetryButton({ config }: { config: Record<string, unknown> }) {
  const navigate = useNavigate();
  const route = CONFIG_PAGE_ROUTES[config.page as string];
  if (!route) return null;
  return (
    <button
      className="btn btn-ghost btn-sm"
      onClick={() => navigate(route, { state: { prefill: config } })}
    >
      <RotateCcw size={14} /> Retry with same settings
    </button>
  );
}

export function BackButton({ onBack }: { onBack: () => void }) {
  return (
    <button className="btn btn-ghost btn-sm" onClick={onBack} style={{ justifySelf: "start" }}>
      <ArrowLeft size={14} /> Back to Jobs
    </button>
  );
}

export function DetailHeader({ onBack, config }: { onBack: () => void; config?: Record<string, unknown> }) {
  return (
    <div className="flex-row" style={{ justifyContent: "space-between" }}>
      <BackButton onBack={onBack} />
      {config && hasRetryConfig(config) && <RetryButton config={config} />}
    </div>
  );
}
