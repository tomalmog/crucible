import type { RefObject, UIEvent } from "react";
import { Check } from "lucide-react";
import type { CommandTaskStatus } from "../../types";
import type { JobRecord } from "../../types/jobs";
import type { TrainingProgress } from "../training/TrainingRunMonitor";
import { formatDuration } from "./jobRowDisplay";

interface LocalProgressPanelProps {
  job: JobRecord;
  localTask?: CommandTaskStatus;
  progress: TrainingProgress | null;
}

export function LocalProgressPanel({
  job,
  localTask,
  progress,
}: LocalProgressPanelProps) {
  if (!localTask || localTask.status !== "running" || job.state === "pending") return null;

  return (
    <>
      <div className="job-progress-strip">
        <div className="job-progress-strip-fill" style={{ width: `${localTask.progress_percent}%` }} />
      </div>
      <div className="job-card-meta">
        <span>{localTask.progress_percent.toFixed(0)}% · Elapsed {formatDuration(localTask.elapsed_seconds)}</span>
        <span>~{formatDuration(localTask.remaining_seconds)} remaining</span>
      </div>
      {progress && (
        <div className="job-progress-meta">
          <span>Epoch {progress.epoch}/{progress.totalEpochs}</span>
          {progress.loss != null && <span>Loss: {progress.loss.toFixed(4)}</span>}
          {progress.meanReward != null && <span>Reward: {progress.meanReward.toFixed(4)}</span>}
        </div>
      )}
    </>
  );
}

interface PullProgressPanelProps {
  pullDone: boolean;
  pullError: string | null;
  pulling: boolean;
  pullProgress: string[];
}

export function PullProgressPanel({
  pullDone,
  pullError,
  pulling,
  pullProgress,
}: PullProgressPanelProps) {
  if (!pulling && !pullDone && !pullError) return null;

  return (
    <div>
      {pulling && (
        <div className="pull-steps">
          {pullProgress.map((step) => (
            <div key={step} className="pull-step">{step}</div>
          ))}
          {pullProgress.length === 0 && <div className="pull-step">Starting pull...</div>}
        </div>
      )}
      {pullDone && (
        <div className="pull-success flex-row" style={{ gap: "var(--space-xs)" }}>
          <Check size={14} /> Model pulled successfully!
        </div>
      )}
      {pullError && <div className="error-alert-prominent">{pullError}</div>}
    </div>
  );
}

interface LocalJobOutputPanelProps {
  isExpanded: boolean;
  job: JobRecord;
  localLogRef: RefObject<HTMLPreElement | null>;
  localTask?: CommandTaskStatus;
}

export function LocalJobOutputPanel({
  isExpanded,
  job,
  localLogRef,
  localTask,
}: LocalJobOutputPanelProps) {
  if (!isExpanded) return null;
  if (localTask) {
    return (
      <div className="job-expanded">
        {localTask.stdout && (
          <div>
            <div className="job-output-label">stdout</div>
            <pre ref={localLogRef} className="console console-tall">{localTask.stdout}</pre>
          </div>
        )}
        {localTask.stderr && (
          <div>
            <details open={localTask.status !== "failed"}>
              <summary className={`job-traceback-toggle ${localTask.status === "failed" ? "error-text" : ""}`}>
                {localTask.status === "failed" ? "full traceback" : "logs"}
              </summary>
              <pre className="console console-short">{localTask.stderr}</pre>
            </details>
          </div>
        )}
        {!localTask.stdout && !localTask.stderr && (
          <div className="job-no-output">No output yet.</div>
        )}
      </div>
    );
  }

  if (!job.stdout && !job.stderr) return null;
  return (
    <div className="job-expanded">
      {job.stdout && (
        <div>
          <div className="job-output-label">Last captured output</div>
          <pre className="console console-tall">{job.stdout}</pre>
        </div>
      )}
      {job.stderr && (
        <details>
          <summary className="job-traceback-toggle error-text">stderr</summary>
          <pre className="console console-short">{job.stderr}</pre>
        </details>
      )}
    </div>
  );
}

interface RemoteJobLogPanelProps {
  handleLogScroll: (event: UIEvent<HTMLPreElement>) => void;
  loading: boolean;
  logContainerRef: RefObject<HTMLPreElement | null>;
  logs: string;
  showLogs: boolean;
}

export function RemoteJobLogPanel({
  handleLogScroll,
  loading,
  logContainerRef,
  logs,
  showLogs,
}: RemoteJobLogPanelProps) {
  if (!showLogs) return null;

  return (
    <div className="job-expanded">
      {loading && !logs && (
        <div className="job-no-output">Fetching logs from cluster...</div>
      )}
      {logs ? (
        <div>
          <div className="job-output-label">Remote Logs {loading && "(refreshing...)"}</div>
          <pre ref={logContainerRef} className="console console-tall" onScroll={handleLogScroll}>
            {logs}
          </pre>
        </div>
      ) : (
        !loading && <div className="job-no-output">No logs available yet.</div>
      )}
    </div>
  );
}
