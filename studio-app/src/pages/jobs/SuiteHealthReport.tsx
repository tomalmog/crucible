import { Activity, BrainCircuit, CheckCircle2, FileText } from "lucide-react";
import type { JobRecord } from "../../types/jobs";
import {
  ReportMetric,
  ReportRow,
  artifactRows,
  configString,
} from "./HealthReportParts";
import { DetailHeader } from "./RetryButton";

interface SuiteCheckView {
  checkId: string;
  error: string;
  label: string;
  implication: string;
  recommendedAction: string;
  severity: string;
  status: string;
  summary: string;
  why: string;
}

interface SuiteReportView {
  checks: SuiteCheckView[];
  datasetName: string;
  modelPath: string;
  overallResult: string;
  plainEnglishSummary: string;
  status: string;
  suiteTitle: string;
}

interface SuiteHealthReportProps {
  config: Record<string, unknown>;
  job: JobRecord;
  onBack: () => void;
  report: SuiteReportView;
}

export function SuiteHealthReport({
  config,
  job,
  onBack,
  report,
}: SuiteHealthReportProps): React.ReactNode {
  const completedChecks = report.checks.filter((check) => check.status === "completed").length;
  const actions = priorityActions(report);
  const reviewCount = report.checks.filter((check) => check.severity !== "ok").length;
  const artifactEntries = artifactRows({
    status: report.status,
    overall_result: report.overallResult,
    plain_english_summary: report.plainEnglishSummary,
  });
  return (
    <div className="panel stack-lg">
      <DetailHeader onBack={onBack} config={config} jobType={job.jobType} />
      <div className="health-report-header">
        <div>
          <span className="metric-label">Model Health Assessment</span>
          <h3>{report.suiteTitle}</h3>
          <p>Promotion-readiness assessment for the selected candidate model.</p>
        </div>
        <span className="goal-card-icon"><BrainCircuit size={16} /></span>
      </div>

      <section className="health-report-brief">
        <div>
          <span className="metric-label">Recommendation</span>
          <h4>{report.overallResult}</h4>
          <p>{report.plainEnglishSummary}</p>
        </div>
        <div className="health-report-action-panel">
          <span className="metric-label">Priority Actions</span>
          <ol>
            {actions.map((action) => <li key={action}>{action}</li>)}
          </ol>
        </div>
      </section>

      <div className="health-report-summary">
        <ReportMetric icon={<CheckCircle2 size={14} />} label="Checks Completed" value={`${completedChecks} of ${report.checks.length}`} />
        <ReportMetric icon={<Activity size={14} />} label="Items for Review" value={String(reviewCount)} />
        <ReportMetric icon={<FileText size={14} />} label="Run" value={job.label || job.jobId} />
      </div>

      <section className="health-report-output">
        <h4>Assessment Findings</h4>
        <div className="health-report-finding-list">
          {report.checks.map((check) => (
            <article className={`health-report-finding severity-${check.severity}`} key={check.checkId}>
              <div className="health-report-finding-header">
                <div>
                  <span className="metric-label">{check.label}</span>
                  <strong>{check.summary}</strong>
                </div>
                <span className={badgeClass(check)}>{statusLabel(check)}</span>
              </div>
              <div className="health-report-finding-grid">
                <FindingBlock label="Why it matters" value={check.why} />
                <FindingBlock label="Interpretation" value={check.implication} />
                <FindingBlock label="Action" value={check.recommendedAction} />
              </div>
            </article>
          ))}
        </div>
      </section>

      <details className="health-report-appendix">
        <summary>Assessment inputs</summary>
        <div className="docs-table-wrap">
          <table className="docs-table">
            <tbody>
              <ReportRow label="Candidate model" value={report.modelPath} />
              <ReportRow label="Calibration dataset" value={report.datasetName} />
              <ReportRow label="Suite" value={report.suiteTitle} />
              <ReportRow label="Sample cap" value={configValue(config, "maxSamples")} />
              <ReportRow label="Probe prompt" value={configString(config, "probeText")} />
              <ReportRow label="Clean contrast" value={configString(config, "cleanText")} />
              <ReportRow label="Corrupted contrast" value={configString(config, "corruptedText")} />
              <ReportRow label="Label field" value={configString(config, "labelField")} />
            </tbody>
          </table>
        </div>
      </details>

      {artifactEntries.length > 0 && <ReportFields entries={artifactEntries} />}
    </div>
  );
}

export function buildSuiteReport(
  result: Record<string, unknown> | null,
  config: Record<string, unknown>,
  job: JobRecord,
): SuiteReportView | null {
  const rawChecks = Array.isArray(result?.checks) ? result.checks : [];
  const checks = rawChecks.map(toSuiteCheck).filter((check): check is SuiteCheckView => check !== null);
  if (checks.length === 0) return null;
  return {
    checks,
    datasetName: resultString(result, "dataset_name") || configString(config, "dataset"),
    modelPath: resultString(result, "model_path") || configString(config, "modelPath") || job.modelPath,
    overallResult: resultString(result, "overall_result") || "Ready for review",
    plainEnglishSummary: resultString(result, "plain_english_summary") || "Review the check results above.",
    status: resultString(result, "status") || job.state,
    suiteTitle: resultString(result, "suite_title") || configString(config, "suiteTitle") || "Model Health",
  };
}

function ReportFields({ entries }: { entries: [string, string][] }): React.ReactNode {
  return (
    <details className="health-report-appendix">
      <summary>Report fields</summary>
      <div className="docs-table-wrap">
        <table className="docs-table">
          <tbody>
            {entries.map(([key, value]) => (
              <tr key={key}>
                <td>{key.replace(/_/g, " ")}</td>
                <td className="text-mono text-sm">{value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

function FindingBlock({ label, value }: { label: string; value: string }): React.ReactNode {
  return (
    <div>
      <span className="metric-label">{label}</span>
      <p>{value || "-"}</p>
    </div>
  );
}

function priorityActions(report: SuiteReportView): string[] {
  const ranked = [...report.checks].sort((left, right) => severityRank(right) - severityRank(left));
  const actions = ranked
    .map((check) => check.recommendedAction)
    .filter((action, index, all) => action && all.indexOf(action) === index);
  if (actions.length > 0) {
    return actions.slice(0, 3);
  }
  return [
    "Review these findings alongside eval scores, latency, and cost before promotion.",
    "Use this report as the baseline comparison for the next candidate model.",
  ];
}

function severityRank(check: SuiteCheckView): number {
  if (check.status !== "completed" || check.severity === "critical") return 4;
  if (check.severity === "warning") return 3;
  if (check.severity === "review") return 2;
  return 1;
}

function statusLabel(check: SuiteCheckView): string {
  if (check.status !== "completed") return "blocked";
  if (check.severity === "ok") return "clear";
  return "review";
}

function badgeClass(check: SuiteCheckView): string {
  if (check.status !== "completed" || check.severity === "critical") return "badge badge-error";
  if (check.severity === "ok") return "badge badge-success";
  return "badge badge-warning";
}

function configValue(config: Record<string, unknown>, key: string): string {
  const value = config[key];
  if (typeof value === "string") return value;
  if (typeof value === "number") return String(value);
  return "";
}

function toSuiteCheck(value: unknown): SuiteCheckView | null {
  if (!isRecord(value)) return null;
  return {
    checkId: stringField(value, "check_id") || stringField(value, "id"),
    error: stringField(value, "error"),
    implication: stringField(value, "implication") || fallbackImplication(value),
    label: stringField(value, "label") || stringField(value, "check_id"),
    recommendedAction: stringField(value, "recommended_action") || fallbackAction(value),
    severity: stringField(value, "severity") || fallbackSeverity(value),
    status: stringField(value, "status") || "completed",
    summary: stringField(value, "summary") || "No summary was returned.",
    why: stringField(value, "why") || stringField(value, "reason"),
  };
}

function fallbackImplication(value: Record<string, unknown>): string {
  const summary = stringField(value, "summary");
  if (summary.toLowerCase().includes("unknown")) {
    return "The diagnostic may not reflect real product behavior because the prompt/tokenizer pairing is weak.";
  }
  return "This result should be reviewed as part of the release decision, not treated as a standalone pass/fail.";
}

function fallbackAction(value: Record<string, unknown>): string {
  const summary = stringField(value, "summary").toLowerCase();
  if (summary.includes("unknown")) return "Use a calibration-dataset prompt and rerun the health check.";
  if (summary.includes("recovery")) return "Add regression prompts for this contrast and compare against the base model.";
  if (summary.includes("variance")) return "Inspect clusters by label/source and rebalance data if they follow artifacts.";
  return "Review the diagnostic output against expected product behavior before promotion.";
}

function fallbackSeverity(value: Record<string, unknown>): string {
  const status = stringField(value, "status");
  const summary = stringField(value, "summary").toLowerCase();
  if (status && status !== "completed") return "critical";
  if (summary.includes("unknown") || summary.includes("failed")) return "warning";
  if (summary.includes("recovery") || summary.includes("variance")) return "review";
  return "ok";
}

function resultString(result: Record<string, unknown> | null, key: string): string {
  if (!result) return "";
  return stringField(result, key);
}

function stringField(record: Record<string, unknown>, key: string): string {
  const value = record[key];
  return typeof value === "string" ? value : "";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
