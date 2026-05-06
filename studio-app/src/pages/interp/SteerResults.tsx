import type { ReactNode } from "react";
import type { SteerApplyResult, SteerComputeResult } from "../../types/interp";
import { EvidenceSummary } from "./EvidenceSummary";
import { formatNumber } from "./interpDisplay";

export function SteerComputeResults({ result }: { result: SteerComputeResult }): ReactNode {
  return (
    <div className="interp-evidence-card stack-md">
      <div className="interp-evidence-header">
        <div>
          <span className="interp-kicker">Steer compute</span>
          <h3>Activation steering vector</h3>
          <p>
            Estimate a contrastive activation direction and compare generations
            after applying it at the selected layer.
          </p>
        </div>
        <span className="interp-evidence-badge">{result.layer_name}</span>
      </div>
      <EvidenceSummary
        items={[
          { label: "Vector norm", value: formatNumber(result.vector_norm, 4), tone: "positive" },
          { label: "Cosine similarity", value: formatNumber(result.cosine_similarity, 4) },
          { label: "Positive samples", value: String(result.num_positive) },
          { label: "Negative samples", value: String(result.num_negative) },
        ]}
      />
      <VectorBalance result={result} />
      <div className="steering-vector-stage">
        <div className="steering-vector-line" aria-hidden="true">
          <span />
        </div>
        <article>
          <span>positive set</span>
          <strong>{result.num_positive} examples</strong>
        </article>
        <article>
          <span>negative contrast</span>
          <strong>{result.num_negative} examples</strong>
        </article>
      </div>
      {result.steering_vector_path && (
        <div className="interp-artifact-path">Vector saved to: {result.steering_vector_path}</div>
      )}
    </div>
  );
}

export function SteerApplyResults({ result }: { result: SteerApplyResult }): ReactNode {
  return (
    <div className="interp-evidence-card stack-md">
      <div className="interp-evidence-header">
        <div>
          <span className="interp-kicker">Steer apply</span>
          <h3>Original and steered generation</h3>
          <p>
            Apply a direction to the residual stream and compare the generated behavior
            against the original continuation.
          </p>
        </div>
        <span className="interp-evidence-badge">coefficient {result.coefficient}</span>
      </div>
      <EvidenceSummary
        items={[
          { label: "Coefficient", value: String(result.coefficient), tone: "positive" },
          { label: "Layer", value: result.layer_name },
          { label: "Max new tokens", value: String(result.max_new_tokens) },
        ]}
      />
      <div className="steering-comparison">
        <GenerationPanel title="Original" input={result.input_text} output={result.original_text} />
        <GenerationPanel title="Steered" input={result.input_text} output={result.steered_text} isSteered />
      </div>
    </div>
  );
}

function VectorBalance({ result }: { result: SteerComputeResult }): ReactNode {
  const total = Math.max(result.num_positive + result.num_negative, 1);
  return (
    <div className="vector-balance">
      <span className="metric-label">Contrast set balance</span>
      <div>
        <span style={{ flexGrow: result.num_positive / total }}>positive</span>
        <span style={{ flexGrow: result.num_negative / total }}>negative</span>
      </div>
    </div>
  );
}

function GenerationPanel({
  input,
  isSteered = false,
  output,
  title,
}: {
  input: string;
  isSteered?: boolean;
  output: string;
  title: string;
}): ReactNode {
  return (
    <article className={`generation-panel${isSteered ? " steered" : ""}`}>
      <span>{title}</span>
      <strong>{input}</strong>
      <p>{output}</p>
    </article>
  );
}
