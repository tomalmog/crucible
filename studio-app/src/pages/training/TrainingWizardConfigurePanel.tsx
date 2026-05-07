import { RotateCcw } from "lucide-react";
import type { SharedTrainingConfig, TrainingMethod } from "../../types/training";
import { FormField } from "../../components/shared/FormField";
import { SharedTrainingFields } from "./forms/SharedTrainingFields";
import { TrainingMethodConfig } from "./TrainingMethodConfig";
import { TrainingPreflightPanel } from "./TrainingPreflightPanel";
import { TrainingRunContextFields } from "./TrainingRunContextFields";

interface TrainingWizardConfigurePanelProps {
  canStart: boolean;
  evalObjective: string;
  extra: Record<string, string>;
  method: TrainingMethod;
  methodName: string;
  missing: string[];
  modelName: string;
  onReset: () => void;
  onStart: () => void;
  projectName: string;
  remoteEnabled: boolean;
  setEvalObjective: (value: string) => void;
  setExtra: (extra: Record<string, string>) => void;
  setModelName: (value: string) => void;
  setProjectName: (value: string) => void;
  setShared: (config: SharedTrainingConfig) => void;
  shared: SharedTrainingConfig;
  startError: string | null;
  submitting: boolean;
}

export function TrainingWizardConfigurePanel({
  canStart,
  evalObjective,
  extra,
  method,
  methodName,
  missing,
  modelName,
  onReset,
  onStart,
  projectName,
  remoteEnabled,
  setEvalObjective,
  setExtra,
  setModelName,
  setProjectName,
  setShared,
  shared,
  startError,
  submitting,
}: TrainingWizardConfigurePanelProps) {
  return (
    <div className="panel stack-lg has-remote-tab">
      <TrainingMethodConfig method={method} extra={extra} setExtra={setExtra} />
      <TrainingRunContextFields
        evalObjective={evalObjective}
        projectName={projectName}
        setEvalObjective={setEvalObjective}
        setProjectName={setProjectName}
      />
      <TrainingPreflightPanel
        method={method}
        methodName={methodName}
        shared={shared}
        extra={extra}
        modelName={modelName}
        projectName={projectName}
        evalObjective={evalObjective}
        remoteEnabled={remoteEnabled}
      />
      <SharedTrainingFields config={shared} onChange={setShared} />
      <FormField label="Model Name" required>
        <input
          value={modelName}
          onChange={(event) => setModelName(event.currentTarget.value)}
          placeholder="My-Model-0"
        />
      </FormField>
      {!canStart && (
        <div className="error-alert">
          Missing required fields: {missing.join(", ")}
        </div>
      )}
      {startError && <div className="error-alert">{startError}</div>}
      <div className="flex-row">
        <button
          className="btn btn-primary btn-lg"
          onClick={onStart}
          disabled={!canStart || submitting}
        >
          {submitting ? "Submitting..." : remoteEnabled ? "Submit to Cluster" : "Start Fine-tune"}
        </button>
        <button
          className="btn btn-ghost btn-sm"
          onClick={onReset}
          title="Reset to defaults"
        >
          <RotateCcw size={12} /> Reset
        </button>
      </div>
    </div>
  );
}
