import { useForgeCommand } from "../../hooks/useForgeCommand";
import { StatusConsole } from "../../components/shared/StatusConsole";
import { CommandProgress } from "../../components/shared/CommandProgress";

interface ReadinessChecklistProps {
  dataRoot: string;
}

export function ReadinessChecklist({ dataRoot }: ReadinessChecklistProps) {
  const command = useForgeCommand();

  async function runChecklist() {
    await command.run(dataRoot, ["deploy", "checklist"]);
  }

  return (
    <div className="panel">
      <h3 className="panel-title">Deployment Readiness Checklist</h3>
      <p className="text-tertiary">
        Run a comprehensive readiness check before deploying your model.
      </p>
      <button className="btn btn-primary btn-lg" onClick={() => runChecklist().catch(console.error)} disabled={command.isRunning}>
        {command.isRunning ? "Checking..." : "Run Checklist"}
      </button>
      {command.isRunning && command.status && (
        <div className="gap-top">
          <CommandProgress label="Running checks..." percent={command.status.progress_percent} />
        </div>
      )}
      {command.output && <div className="gap-top"><StatusConsole output={command.output} /></div>}
      {command.error && <p className="error-text gap-top-sm">{command.error}</p>}
    </div>
  );
}
