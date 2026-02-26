import { useEffect, useRef, useMemo } from "react";
import { CommandProgress } from "../../components/shared/CommandProgress";

interface TrainingRunMonitorProps {
  command: {
    isRunning: boolean;
    status: { progress_percent: number; elapsed_seconds: number; remaining_seconds: number } | null;
    output: string;
    error: string | null;
  };
}

export interface TrainingProgress {
  epoch: number;
  totalEpochs: number;
  batch?: number;
  totalBatches?: number;
  loss?: number;
  validationLoss?: number;
  etaSeconds?: number;
  meanReward?: number;
}

export function parseTrainingProgress(stdout: string): TrainingProgress | null {
  const lines = stdout.split("\n");
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i].trim();
    if (!line.startsWith("{")) continue;
    try {
      const parsed = JSON.parse(line);
      if (
        parsed.event === "training_epoch_completed" ||
        parsed.event === "training_batch_progress"
      ) {
        return {
          epoch: parsed.epoch,
          totalEpochs: parsed.total_epochs,
          batch: parsed.batch,
          totalBatches: parsed.total_batches,
          loss: parsed.train_loss ?? parsed.loss,
          validationLoss: parsed.validation_loss,
          etaSeconds: parsed.eta_seconds,
          meanReward: parsed.mean_reward,
        };
      }
    } catch {
      continue;
    }
  }
  return null;
}

function formatEta(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

export function TrainingRunMonitor({ command }: TrainingRunMonitorProps) {
  const consoleRef = useRef<HTMLPreElement>(null);
  const progress = useMemo(
    () => parseTrainingProgress(command.output),
    [command.output],
  );

  useEffect(() => {
    if (consoleRef.current) {
      consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
    }
  }, [command.output]);

  const epochPercent = progress
    ? (progress.epoch / progress.totalEpochs) * 100
    : null;

  return (
    <div className="stack-lg">
      {progress && (
        <div className="panel">
          <h4 className="panel-title">Training Progress</h4>
          <div style={{ display: "grid", gap: 12, padding: "4px 0" }}>
            <div className="progress-bar">
              <div className="progress-bar-header">
                <span className="progress-label">
                  Epoch {progress.epoch} / {progress.totalEpochs}
                </span>
                <span className="progress-value">
                  {epochPercent !== null ? `${epochPercent.toFixed(0)}%` : ""}
                </span>
              </div>
              <div className="progress-track">
                <div
                  className="progress-fill"
                  style={{
                    width: `${Math.min(100, Math.max(0, epochPercent ?? 0))}%`,
                  }}
                />
              </div>
            </div>

            {progress.batch != null && progress.totalBatches != null && (
              <div className="progress-bar">
                <div className="progress-bar-header">
                  <span className="progress-label">
                    Batch {progress.batch} / {progress.totalBatches}
                  </span>
                  <span className="progress-value">
                    {((progress.batch / progress.totalBatches) * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="progress-track">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${Math.min(
                        100,
                        (progress.batch / progress.totalBatches) * 100,
                      )}%`,
                    }}
                  />
                </div>
              </div>
            )}

            <div
              style={{
                display: "flex",
                gap: 24,
                fontSize: "0.8125rem",
                color: "var(--text-secondary)",
              }}
            >
              {progress.loss != null && (
                <span>
                  Loss: <strong>{progress.loss.toFixed(4)}</strong>
                </span>
              )}
              {progress.validationLoss != null && (
                <span>
                  Val Loss: <strong>{progress.validationLoss.toFixed(4)}</strong>
                </span>
              )}
              {progress.meanReward != null && (
                <span>
                  Reward: <strong>{progress.meanReward.toFixed(4)}</strong>
                </span>
              )}
              {progress.etaSeconds != null && (
                <span>ETA: {formatEta(progress.etaSeconds)}</span>
              )}
            </div>
          </div>
        </div>
      )}

      {!progress && command.status && (
        <CommandProgress
          label="Training in progress..."
          percent={command.status.progress_percent}
          elapsed={command.status.elapsed_seconds}
          remaining={command.status.remaining_seconds}
        />
      )}

      <div className="panel">
        <h4 className="panel-title">Console Output</h4>
        <pre className="console-tall" ref={consoleRef}>
          {command.output || "Waiting for output..."}
        </pre>
      </div>

      {command.error && (
        <div className="panel">
          <h4 className="panel-title error-text">Error</h4>
          <p className="error-text text-sm">{command.error}</p>
        </div>
      )}
    </div>
  );
}
