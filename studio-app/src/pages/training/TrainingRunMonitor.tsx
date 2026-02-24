import { CommandProgress } from "../../components/shared/CommandProgress";

interface TrainingRunMonitorProps {
  command: {
    isRunning: boolean;
    status: { progress_percent: number; elapsed_seconds: number; remaining_seconds: number } | null;
    output: string;
    error: string | null;
  };
}

export function TrainingRunMonitor({ command }: TrainingRunMonitorProps) {
  return (
    <div className="stack-lg">
      {command.status && (
        <CommandProgress
          label="Training in progress..."
          percent={command.status.progress_percent}
          elapsed={command.status.elapsed_seconds}
          remaining={command.status.remaining_seconds}
        />
      )}

      <div className="panel">
        <h4 className="panel-title">Console Output</h4>
        <pre className="console-tall">
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
