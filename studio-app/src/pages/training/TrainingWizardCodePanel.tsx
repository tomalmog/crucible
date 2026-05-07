import { CodeEditor } from "../../components/shared/CodeEditor";

interface TrainingWizardCodePanelProps {
  onRunScript: () => void;
  scriptContent: string;
  setScriptContent: (content: string) => void;
  startError: string | null;
  submitting: boolean;
}

export function TrainingWizardCodePanel({
  onRunScript,
  scriptContent,
  setScriptContent,
  startError,
  submitting,
}: TrainingWizardCodePanelProps) {
  return (
    <div className="panel stack-lg">
      <CodeEditor value={scriptContent} onChange={setScriptContent} maxHeight="600px" />
      {startError && <div className="error-alert">{startError}</div>}
      <div className="flex-row">
        <button
          className="btn btn-primary btn-lg"
          onClick={onRunScript}
          disabled={submitting}
        >
          {submitting ? "Running..." : "Run Script"}
        </button>
      </div>
    </div>
  );
}
