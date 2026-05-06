import type { ReactNode } from "react";
import type {
  LinearProbeResult,
  LogitLensResult,
  PatchingResult,
  PcaResult,
  SaeAnalyzeResult,
  SaeTrainResult,
  SteerApplyResult,
  SteerComputeResult,
} from "../../types/interp";
import { ActivationPatchingResults } from "../interp/ActivationPatchingResults";
import { ActivationPcaResults } from "../interp/ActivationPcaResults";
import { LinearProbeResults } from "../interp/LinearProbeResults";
import { LogitLensResults } from "../interp/LogitLensResults";
import { SaeAnalyzeResults, SaeTrainResults } from "../interp/SaeResults";
import { SteerApplyResults, SteerComputeResults } from "../interp/SteerResults";

type InterpArtifact =
  | { kind: "logit-lens"; result: LogitLensResult }
  | { kind: "activation-pca"; result: PcaResult }
  | { kind: "activation-patch"; result: PatchingResult }
  | { kind: "linear-probe"; result: LinearProbeResult }
  | { kind: "sae-train"; result: SaeTrainResult }
  | { kind: "sae-analyze"; result: SaeAnalyzeResult }
  | { kind: "steer-compute"; result: SteerComputeResult }
  | { kind: "steer-apply"; result: SteerApplyResult };

export function parseInterpArtifact(jobType: string, value: unknown): InterpArtifact | null {
  switch (jobType) {
    case "logit-lens":
      return isLogitLensResult(value) ? { kind: jobType, result: value } : null;
    case "activation-pca":
      return isPcaResult(value) ? { kind: jobType, result: value } : null;
    case "activation-patch":
      return isPatchingResult(value) ? { kind: jobType, result: value } : null;
    case "linear-probe":
      return isLinearProbeResult(value) ? { kind: jobType, result: value } : null;
    case "sae-train":
      return isSaeTrainResult(value) ? { kind: jobType, result: value } : null;
    case "sae-analyze":
      return isSaeAnalyzeResult(value) ? { kind: jobType, result: value } : null;
    case "steer-compute":
      return isSteerComputeResult(value) ? { kind: jobType, result: value } : null;
    case "steer-apply":
      return isSteerApplyResult(value) ? { kind: jobType, result: value } : null;
    default:
      return null;
  }
}

export function InterpJobArtifactRenderer({ artifact }: { artifact: InterpArtifact }): ReactNode {
  switch (artifact.kind) {
    case "logit-lens":
      return <LogitLensResults result={artifact.result} />;
    case "activation-pca":
      return <ActivationPcaResults result={artifact.result} />;
    case "activation-patch":
      return <ActivationPatchingResults result={artifact.result} />;
    case "linear-probe":
      return <LinearProbeResults result={artifact.result} />;
    case "sae-train":
      return <SaeTrainResults result={artifact.result} />;
    case "sae-analyze":
      return <SaeAnalyzeResults result={artifact.result} />;
    case "steer-compute":
      return <SteerComputeResults result={artifact.result} />;
    case "steer-apply":
      return <SteerApplyResults result={artifact.result} />;
  }
}

export function InvalidInterpArtifactNotice({ jobType }: { jobType: string }): ReactNode {
  return (
    <div className="empty-state">
      <h3>Artifact shape does not match {jobType}</h3>
      <p>
        The job completed, but the JSON artifact is missing fields required by
        the renderer. Open the logs below to inspect the raw output.
      </p>
    </div>
  );
}

function isLogitLensResult(value: unknown): value is LogitLensResult {
  const data = asRecord(value);
  return data !== null && isStringArray(data.input_tokens) && Array.isArray(data.layers);
}

function isPcaResult(value: unknown): value is PcaResult {
  const data = asRecord(value);
  return data !== null && hasString(data, "layer_name") && hasNumber(data, "layer_index")
    && hasString(data, "granularity") && isNumberArray(data.explained_variance)
    && Array.isArray(data.points);
}

function isPatchingResult(value: unknown): value is PatchingResult {
  const data = asRecord(value);
  return data !== null && hasString(data, "clean_text") && hasString(data, "corrupted_text")
    && hasString(data, "metric") && hasNumber(data, "clean_metric")
    && hasNumber(data, "corrupted_metric") && Array.isArray(data.layer_results);
}

function isLinearProbeResult(value: unknown): value is LinearProbeResult {
  const data = asRecord(value);
  return data !== null && Array.isArray(data.layers);
}

function isSaeTrainResult(value: unknown): value is SaeTrainResult {
  const data = asRecord(value);
  return data !== null && hasString(data, "sae_path") && hasString(data, "layer_name")
    && hasNumber(data, "layer_index") && hasNumber(data, "input_dim")
    && hasNumber(data, "latent_dim") && hasNumber(data, "epochs")
    && hasNumber(data, "final_loss") && hasNumber(data, "final_recon_loss")
    && hasNumber(data, "final_sparsity_loss") && Array.isArray(data.history);
}

function isSaeAnalyzeResult(value: unknown): value is SaeAnalyzeResult {
  const data = asRecord(value);
  return data !== null && hasString(data, "input_text") && hasString(data, "layer_name")
    && hasNumber(data, "reconstruction_error") && hasNumber(data, "sparsity")
    && hasNumber(data, "active_features") && hasNumber(data, "total_features")
    && Array.isArray(data.top_features);
}

function isSteerComputeResult(value: unknown): value is SteerComputeResult {
  const data = asRecord(value);
  return data !== null && hasString(data, "steering_vector_path") && hasString(data, "layer_name")
    && hasNumber(data, "layer_index") && hasNumber(data, "vector_norm")
    && hasNumber(data, "cosine_similarity") && hasNumber(data, "num_positive")
    && hasNumber(data, "num_negative");
}

function isSteerApplyResult(value: unknown): value is SteerApplyResult {
  const data = asRecord(value);
  return data !== null && hasString(data, "input_text") && hasString(data, "original_text")
    && hasString(data, "steered_text") && hasNumber(data, "coefficient")
    && hasString(data, "layer_name") && hasNumber(data, "max_new_tokens");
}

function hasString(data: Record<string, unknown>, key: string): boolean {
  return typeof data[key] === "string";
}

function hasNumber(data: Record<string, unknown>, key: string): boolean {
  return typeof data[key] === "number" && Number.isFinite(data[key]);
}

function isStringArray(value: unknown): boolean {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function isNumberArray(value: unknown): boolean {
  return Array.isArray(value) && value.every((item) => typeof item === "number");
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  // Safe after the runtime object/null/array checks above.
  return value as Record<string, unknown>;
}
