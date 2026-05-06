import type { ReactNode } from "react";
import { TrainingCurvesView } from "../../pages/training/TrainingCurvesView";
import {
  InterpJobArtifactRenderer,
  InvalidInterpArtifactNotice,
  parseInterpArtifact,
} from "../../pages/jobs/InterpJobArtifactRenderer";
import type { AgentEvalJobPreview, AgentJobPreview, AgentInterpJobPreview, AgentTrainingJobPreview } from "../../types/agent";

interface AgentJobPreviewCardProps {
  artifact: AgentJobPreview;
  displayMode?: "inline" | "workspace";
}

export function AgentJobPreviewCard({
  artifact,
  displayMode = "inline",
}: AgentJobPreviewCardProps): ReactNode {
  const className = displayMode === "workspace"
    ? "agent-job-preview agent-job-preview-workspace"
    : "agent-job-preview";
  if (artifact.kind === "training") {
    return <TrainingPreview artifact={artifact} className={className} />;
  }
  if (artifact.kind === "eval") {
    return <EvalPreview artifact={artifact} className={className} displayMode={displayMode} />;
  }
  return <InterpPreview artifact={artifact} className={className} displayMode={displayMode} />;
}

function TrainingPreview({
  artifact,
  className,
}: {
  artifact: AgentTrainingJobPreview;
  className: string;
}): ReactNode {
  return (
    <div className={className}>
      <div className="agent-job-preview-header">
        <strong>{artifact.title}</strong>
        <span>{artifact.jobType}</span>
      </div>
      <div className="agent-job-preview-metrics">
        {artifact.finalTrainLoss != null && (
          <div className="agent-job-preview-metric">
            <span>Train loss</span>
            <strong>{artifact.finalTrainLoss.toFixed(4)}</strong>
          </div>
        )}
        {artifact.finalValidationLoss != null && (
          <div className="agent-job-preview-metric">
            <span>Val loss</span>
            <strong>{artifact.finalValidationLoss.toFixed(4)}</strong>
          </div>
        )}
        {artifact.cluster && (
          <div className="agent-job-preview-metric">
            <span>Cluster</span>
            <strong>{artifact.cluster}</strong>
          </div>
        )}
      </div>
      <div className="agent-job-preview-chart">
        <TrainingCurvesView history={artifact.history} />
      </div>
      {artifact.modelPath && (
        <div className="agent-job-preview-footnote">
          Output model: <code>{artifact.modelPath}</code>
        </div>
      )}
    </div>
  );
}

function EvalPreview({
  artifact,
  className,
  displayMode,
}: {
  artifact: AgentEvalJobPreview;
  className: string;
  displayMode: "inline" | "workspace";
}): ReactNode {
  const rows = buildEvalRows(artifact, displayMode);
  return (
    <div className={className}>
      <div className="agent-job-preview-header">
        <strong>{artifact.title}</strong>
        <span>Average {artifact.averageScore.toFixed(1)}%</span>
      </div>
      <div className="agent-job-preview-table-wrap">
        <table className="agent-job-preview-table">
          <thead>
            <tr>
              <th>Benchmark</th>
              <th>Score</th>
              {displayMode === "workspace" && <th>Correct</th>}
            </tr>
          </thead>
          <tbody>
            {rows.map((benchmark) => (
              <tr key={benchmark.name}>
                <td>{benchmark.name}</td>
                <td>{benchmark.score.toFixed(1)}%</td>
                {displayMode === "workspace" && (
                  <td>{formatEvalCorrectCount(benchmark)}</td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

interface EvalTableRow {
  name: string;
  score: number;
  correct: number | null;
  numExamples: number | null;
}

function buildEvalRows(
  artifact: AgentEvalJobPreview,
  displayMode: "inline" | "workspace",
): EvalTableRow[] {
  if (displayMode === "workspace") {
    return artifact.benchmarks.slice(0, 8).map((benchmark) => ({
      name: benchmark.name,
      score: benchmark.score,
      correct: benchmark.correct,
      numExamples: benchmark.numExamples,
    }));
  }
  return artifact.topBenchmarks.map((benchmark) => ({
    name: benchmark.name,
    score: benchmark.score,
    correct: null,
    numExamples: null,
  }));
}

function formatEvalCorrectCount(row: EvalTableRow): string {
  if (row.correct == null || row.numExamples == null) {
    return "—";
  }
  return `${row.correct}/${row.numExamples}`;
}

function InterpPreview({
  artifact,
  className,
  displayMode,
}: {
  artifact: AgentInterpJobPreview;
  className: string;
  displayMode: "inline" | "workspace";
}): ReactNode {
  const richArtifact = parseInterpArtifact(artifact.jobType, artifact.result);
  if (displayMode === "workspace" && richArtifact) {
    return (
      <div className={className}>
        <div className="agent-job-preview-header">
          <strong>{artifact.title}</strong>
          <span>{artifact.jobType}</span>
        </div>
        <div className="agent-job-preview-rich">
          <InterpJobArtifactRenderer artifact={richArtifact} />
        </div>
      </div>
    );
  }
  return (
    <div className={className}>
      <div className="agent-job-preview-header">
        <strong>{artifact.title}</strong>
        <span>{artifact.jobType}</span>
      </div>
      {displayMode === "workspace" && !richArtifact && (
        <InvalidInterpArtifactNotice jobType={artifact.jobType} />
      )}
      <div className="agent-job-preview-list">
        {artifact.summaryLines.map((line) => (
          <div key={line} className="agent-job-preview-row">
            <span>{line}</span>
          </div>
        ))}
      </div>
      {artifact.cluster && (
        <div className="agent-job-preview-footnote">
          Ran on <code>{artifact.cluster}</code>
        </div>
      )}
    </div>
  );
}
