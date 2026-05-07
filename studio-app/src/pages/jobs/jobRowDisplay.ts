import { jobLabel } from "../../utils/jobLabels";

export const ACTIVE_STATES = new Set(["running", "pending"]);

export const NON_TRAINING_TYPES = new Set([
  "eval",
  "logit-lens",
  "activation-pca",
  "activation-patch",
  "activation-patching",
  "linear-probe",
  "model-health-check",
  "sae-train",
  "sae-analyze",
  "steer-compute",
  "steer-apply",
  "hub-download",
  "ingest",
]);

export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  if (minutes < 60) return `${minutes}m ${remainder}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

export function configString(config: Record<string, unknown>, key: string): string {
  const value = config[key];
  return typeof value === "string" ? value : "";
}

export function runTypeLabel(jobType: string): string {
  if (jobType === "eval") return "Eval";
  if (jobType === "sweep") return "Sweep";
  if (jobType === "ingest") return "Dataset ingest";
  if (jobType.includes("export")) return "Export";
  if (NON_TRAINING_TYPES.has(jobType)) return jobLabel(jobType, "");
  return "Fine-tune";
}
