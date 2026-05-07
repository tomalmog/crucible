import { useMemo } from "react";
import { useNavigate } from "react-router";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  FlaskConical,
  Gauge,
} from "lucide-react";
import { useCrucible } from "../../context/CrucibleContext";
import { useUnifiedJobs } from "../../hooks/useUnifiedJobs";
import type { JobRecord } from "../../types/jobs";
import { AdvancedDiagnostics } from "./AdvancedDiagnostics";
import { ModelHealthSuiteRunner } from "./ModelHealthSuiteRunner";

interface HealthCheck {
  label: string;
  value: string;
  detail: string;
  tone: "success" | "warning" | "neutral";
}

interface ModelHealthOverviewProps {
  advancedPrefill?: Record<string, unknown>;
}

interface ModelHealthHeaderProps {
  modelName: string;
  onRunEval: () => void;
  onViewRuns: () => void;
}

export function ModelHealthOverview({
  advancedPrefill,
}: ModelHealthOverviewProps): React.ReactNode {
  const navigate = useNavigate();
  const { dataRoot, selectedModel } = useCrucible();
  const { jobs, isLoading } = useUnifiedJobs(dataRoot);
  const modelName = selectedModel?.modelName ?? "";

  const scopedJobs = useMemo(
    () => {
      if (!modelName) {
        return jobs;
      }
      return jobs.filter((job) => matchesModel(job, modelName, selectedModel?.modelPath ?? ""));
    },
    [jobs, modelName, selectedModel?.modelPath],
  );

  const checks = useMemo(
    () => buildHealthChecks(scopedJobs, isLoading, modelName),
    [scopedJobs, isLoading, modelName],
  );

  return (
    <div className="stack-lg">
      <ModelHealthHeader
        modelName={modelName}
        onRunEval={() => navigate("/evals")}
        onViewRuns={() => navigate("/runs")}
      />

      <ModelHealthSuiteRunner />

      <HealthCheckSnapshot checks={checks} />

      <details className="advanced-methods" open={isAdvancedPrefill(advancedPrefill)}>
        <summary>Debug model failures with targeted diagnostics</summary>
        <AdvancedDiagnostics prefill={advancedPrefill} />
      </details>
    </div>
  );
}

function ModelHealthHeader({
  modelName,
  onRunEval,
  onViewRuns,
}: ModelHealthHeaderProps): React.ReactNode {
  return (
    <section className="model-design-header">
      <div>
        <span className="section-kicker">Model design workspace</span>
        <h2>
          {modelName
            ? `${modelName} readiness`
            : "Understand, debug, and improve candidate models"}
        </h2>
        <p>
          Start with a health report, use findings to choose targeted diagnostics,
          and keep every decision tied to the run history.
        </p>
      </div>
      <div className="flex-row">
        <button className="btn btn-primary" onClick={onRunEval}>
          <FlaskConical size={14} /> Run Eval
        </button>
        <button className="btn" onClick={onViewRuns}>
          <Activity size={14} /> View Runs
        </button>
      </div>
    </section>
  );
}

function HealthCheckSnapshot({ checks }: { checks: HealthCheck[] }): React.ReactNode {
  return (
    <div className="health-check-grid health-check-grid-compact">
      {checks.map((check) => (
        <HealthCheckCard key={check.label} check={check} />
      ))}
    </div>
  );
}

function HealthCheckCard({ check }: { check: HealthCheck }): React.ReactNode {
  const icon = check.tone === "success"
    ? <CheckCircle2 size={16} />
    : check.tone === "warning"
      ? <AlertTriangle size={16} />
      : <Gauge size={16} />;
  return (
    <article className={`health-check-card health-check-card-${check.tone}`}>
      <span className="health-check-icon">{icon}</span>
      <span className="metric-label">{check.label}</span>
      <strong>{check.value}</strong>
      <p>{check.detail}</p>
    </article>
  );
}

function buildHealthChecks(
  jobs: JobRecord[],
  isLoading: boolean,
  modelName: string,
): HealthCheck[] {
  if (isLoading) {
    return [{
      label: "Loading",
      value: "Checking runs",
      detail: "Run history is loading from the current workspace.",
      tone: "neutral",
    }];
  }

  const completed = jobs.filter((job) => job.state === "completed");
  const failed = jobs.filter((job) => job.state === "failed");
  const evals = completed.filter((job) => job.jobType === "eval");
  const healthChecks = completed.filter(isModelHealthRun);
  const hasRecentFailure = failed.length > 0;
  const subject = modelName || "this workspace";

  return [
    {
      label: "Health report",
      value: healthChecks.length > 0 ? "Available" : "Not run",
      detail: healthChecks.length > 0
        ? `Latest report: ${displayJobName(healthChecks[0])}`
        : `Run a standard health check before promoting ${subject}.`,
      tone: healthChecks.length > 0 ? "success" : "warning",
    },
    {
      label: "Eval coverage",
      value: evals.length > 0 ? `${evals.length} completed` : "Missing",
      detail: evals.length > 0
        ? `Latest eval: ${displayJobName(evals[0])}`
        : `Run a baseline and candidate eval before promoting ${subject}.`,
      tone: evals.length > 0 ? "success" : "warning",
    },
    {
      label: "Run stability",
      value: hasRecentFailure ? `${failed.length} failed` : "No failures",
      detail: hasRecentFailure
        ? "Inspect failed runs before reusing the same dataset, model, or cluster."
        : "No failed runs are recorded for the current scope.",
      tone: hasRecentFailure ? "warning" : "success",
    },
  ];
}

function matchesModel(job: JobRecord, modelName: string, modelPath: string): boolean {
  const haystack = [
    job.label,
    job.modelName,
    job.modelPath,
    job.modelPathLocal,
    JSON.stringify(job.config ?? {}),
  ].join(" ");
  return haystack.includes(modelName) || (!!modelPath && haystack.includes(modelPath));
}

function isModelHealthRun(job: JobRecord): boolean {
  return job.config.workflow === "model-health-check";
}

function displayJobName(job: JobRecord): string {
  return job.label || job.modelName || job.jobType;
}

function isAdvancedPrefill(prefill?: Record<string, unknown>): boolean {
  return typeof prefill?.tab === "string" && prefill.tab !== "health";
}
