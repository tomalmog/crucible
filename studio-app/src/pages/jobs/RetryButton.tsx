import { useNavigate } from "react-router";
import { ArrowLeft, RotateCcw } from "lucide-react";

const CONFIG_PAGE_ROUTES: Record<string, string> = {
  training: "/training",
  benchmarks: "/benchmarks",
  interpretability: "/interpretability",
};

const JOB_TYPE_TO_PAGE: Record<string, string> = {
  train: "training",
  sft: "training",
  "dpo-train": "training",
  "rlhf-train": "training",
  "lora-train": "training",
  distill: "training",
  "domain-adapt": "training",
  "grpo-train": "training",
  "qlora-train": "training",
  "kto-train": "training",
  "orpo-train": "training",
  "multimodal-train": "training",
  "rlvr-train": "training",
  eval: "benchmarks",
  "logit-lens": "interpretability",
  "activation-pca": "interpretability",
  "activation-patch": "interpretability",
  "linear-probe": "interpretability",
  "sae-train": "interpretability",
  "sae-analyze": "interpretability",
  "steer-compute": "interpretability",
  "steer-apply": "interpretability",
};

export function hasRetryConfig(config: Record<string, unknown>): boolean {
  return Object.keys(config).length > 0 && typeof config.page === "string"
    && config.page in CONFIG_PAGE_ROUTES;
}

function resolveRetryRoute(config: Record<string, unknown>, jobType?: string): string | null {
  if (hasRetryConfig(config)) return CONFIG_PAGE_ROUTES[config.page as string];
  if (jobType) {
    const page = JOB_TYPE_TO_PAGE[jobType];
    if (page) return CONFIG_PAGE_ROUTES[page];
  }
  return null;
}

function buildFallbackPrefill(jobType: string): Record<string, unknown> | undefined {
  const page = JOB_TYPE_TO_PAGE[jobType];
  if (!page) return undefined;
  const prefill: Record<string, unknown> = { page };
  if (page === "training") prefill.method = jobType;
  if (page === "interpretability") prefill.tab = jobType;
  return prefill;
}

export function RetryButton({ config, jobType }: { config: Record<string, unknown>; jobType?: string }) {
  const navigate = useNavigate();
  const route = resolveRetryRoute(config, jobType);
  if (!route) return null;
  const prefill = hasRetryConfig(config) ? config : (jobType ? buildFallbackPrefill(jobType) : undefined);
  return (
    <button
      className="btn btn-ghost btn-sm"
      onClick={() => navigate(route, { state: prefill ? { prefill } : undefined })}
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

export function DetailHeader({ onBack, config, jobType }: { onBack: () => void; config?: Record<string, unknown>; jobType?: string }) {
  return (
    <div className="flex-row" style={{ justifyContent: "space-between" }}>
      <BackButton onBack={onBack} />
      {config && <RetryButton config={config} jobType={jobType} />}
    </div>
  );
}
