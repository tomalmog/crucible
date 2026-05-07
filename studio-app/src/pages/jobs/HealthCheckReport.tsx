import type { ReactNode } from "react";
import { Activity, BrainCircuit, CheckCircle2, FileText } from "lucide-react";
import type { JobRecord } from "../../types/jobs";
import { ReportMetric, ReportRow, ReportSection, artifactRows, configString } from "./HealthReportParts";
import { DetailHeader } from "./RetryButton";
import { SuiteHealthReport, buildSuiteReport } from "./SuiteHealthReport";

interface HealthCheckReportProps {
  children: ReactNode;
  config: Record<string, unknown>;
  job: JobRecord;
  jobType: string;
  onBack: () => void;
  result: Record<string, unknown> | null;
}

interface HealthCheckCopy {
  label: string;
  reason: string;
  readout: string;
  nextStep: string;
}

const HEALTH_CHECK_COPY: Record<string, HealthCheckCopy> = {
  "logit-lens": {
    label: "Prediction trace",
    reason: "Checks how the model's next-token predictions evolve across layers.",
    readout: "Look for abrupt confidence jumps, unstable late-layer predictions, or tokens that contradict the intended behavior.",
    nextStep: "If predictions settle late or flip unexpectedly, inspect the failing prompt slice and compare against the base model.",
  },
  "activation-pca": {
    label: "Representation map",
    reason: "Projects calibration activations to show clustering, drift, and shortcut structure.",
    readout: "Healthy maps usually separate by task-relevant labels instead of source file, template, or other accidental metadata.",
    nextStep: "If clusters follow an unwanted artifact, rebalance the dataset or add eval slices for that artifact.",
  },
  "activation-patch": {
    label: "Causal contrast",
    reason: "Swaps activations between clean and corrupted prompts to locate layers that drive a behavior change.",
    readout: "Large recovery bars indicate layers where the contrast is causally concentrated.",
    nextStep: "Use the highest-recovery layers to design focused regression prompts or targeted diagnostics.",
  },
  "activation-patching": {
    label: "Causal contrast",
    reason: "Swaps activations between clean and corrupted prompts to locate layers that drive a behavior change.",
    readout: "Large recovery bars indicate layers where the contrast is causally concentrated.",
    nextStep: "Use the highest-recovery layers to design focused regression prompts or targeted diagnostics.",
  },
  "linear-probe": {
    label: "Label separability",
    reason: "Tests whether a supervised label is encoded in the model's frozen representations.",
    readout: "High probe accuracy means the label is present internally; low accuracy means the model may not represent that distinction.",
    nextStep: "If accuracy is weak, inspect labels, add examples, or run a task eval focused on that label.",
  },
};

export function HealthCheckReport({
  children,
  config,
  job,
  jobType,
  onBack,
  result,
}: HealthCheckReportProps): React.ReactNode {
  const suiteReport = buildSuiteReport(result, config, job);
  if (suiteReport) {
    return (
      <SuiteHealthReport
        config={config}
        job={job}
        onBack={onBack}
        report={suiteReport}
      />
    );
  }

  const checkId = healthCheckId(config, jobType);
  const copy = HEALTH_CHECK_COPY[checkId] ?? {
    label: configString(config, "healthCheckLabel") || jobType,
    reason: "Runs a targeted model-health diagnostic.",
    readout: "Review the diagnostic output for signs of regression, instability, or shortcut behavior.",
    nextStep: "Use the result to decide whether to run a narrower follow-up diagnostic.",
  };
  const suite = configString(config, "suiteTitle") || configString(config, "suite") || "Model health";
  const artifactEntries = artifactRows(result);

  return (
    <div className="panel stack-lg">
      <DetailHeader onBack={onBack} config={config} jobType={job.jobType} />
      <div className="health-report-header">
        <div>
          <span className="metric-label">Model Health Report</span>
          <h3>{copy.label}</h3>
          <p>{copy.reason}</p>
        </div>
        <span className="goal-card-icon"><BrainCircuit size={16} /></span>
      </div>

      <div className="health-report-summary">
        <ReportMetric icon={<CheckCircle2 size={14} />} label="Suite" value={suite} />
        <ReportMetric icon={<Activity size={14} />} label="Run" value={job.label || job.jobId} />
        <ReportMetric icon={<FileText size={14} />} label="Status" value={job.state} />
      </div>

      <div className="health-report-grid">
        <ReportSection title="How to read it" body={copy.readout} />
        <ReportSection title="Recommended next step" body={copy.nextStep} />
      </div>

      <div className="docs-table-wrap">
        <table className="docs-table">
          <tbody>
            <ReportRow label="Model" value={configString(config, "modelPath") || job.modelName || job.modelPath} />
            <ReportRow label="Dataset" value={configString(config, "dataset")} />
            <ReportRow label="Probe" value={configString(config, "probeText")} />
            <ReportRow label="Clean contrast" value={configString(config, "cleanText")} />
            <ReportRow label="Corrupted contrast" value={configString(config, "corruptedText")} />
            <ReportRow label="Label field" value={configString(config, "labelField")} />
          </tbody>
        </table>
      </div>

      <section className="health-report-output">
        <h4>Diagnostic Output</h4>
        {children}
      </section>

      {artifactEntries.length > 0 && (
        <details>
          <summary>Artifacts and raw fields</summary>
          <div className="docs-table-wrap">
            <table className="docs-table">
              <tbody>
                {artifactEntries.map(([key, value]) => (
                  <tr key={key}>
                    <td>{key.replace(/_/g, " ")}</td>
                    <td className="text-mono text-sm">{value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </div>
  );
}

export function isHealthCheckConfig(config: Record<string, unknown>): boolean {
  return configString(config, "workflow") === "model-health-check" ||
    configString(config, "healthCheckId").length > 0;
}

function healthCheckId(config: Record<string, unknown>, jobType: string): string {
  const id = configString(config, "healthCheckId");
  if (id) return id;
  const tool = configString(config, "tool");
  if (tool) return tool;
  return jobType;
}
