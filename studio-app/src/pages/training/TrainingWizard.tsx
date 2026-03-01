import { useMemo, useState } from "react";
import { TrainingMethod, TRAINING_METHODS, REQUIRED_METHOD_FIELDS } from "../../types/training";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { useTrainingConfig } from "../../hooks/useTrainingConfig";
import { buildTrainingArgs } from "../../api/commandArgs";
import { SharedTrainingFields } from "./forms/SharedTrainingFields";
import { BasicTrainForm } from "./forms/BasicTrainForm";
import { SftTrainForm } from "./forms/SftTrainForm";
import { DpoTrainForm } from "./forms/DpoTrainForm";
import { RlhfTrainForm } from "./forms/RlhfTrainForm";
import { LoraTrainForm } from "./forms/LoraTrainForm";
import { DistillTrainForm } from "./forms/DistillTrainForm";
import { DomainAdaptForm } from "./forms/DomainAdaptForm";
import { GrpoTrainForm } from "./forms/GrpoTrainForm";
import { QloraTrainForm } from "./forms/QloraTrainForm";
import { KtoTrainForm } from "./forms/KtoTrainForm";
import { OrpoTrainForm } from "./forms/OrpoTrainForm";
import { MultimodalTrainForm } from "./forms/MultimodalTrainForm";
import { RlvrTrainForm } from "./forms/RlvrTrainForm";
import { TrainingRunMonitor } from "./TrainingRunMonitor";
import { ArrowLeft, ChevronRight, RotateCcw } from "lucide-react";

function getMissingFields(
  method: TrainingMethod,
  extra: Record<string, string>,
): string[] {
  const missing: string[] = [];
  for (const flag of REQUIRED_METHOD_FIELDS[method]) {
    if (!(extra[flag] ?? "").trim()) {
      const label = flag.replace(/^--/, "").replace(/-/g, " ");
      missing.push(label);
    }
  }
  return missing;
}

type Step = "config" | "running" | "done";

interface TrainingWizardProps {
  method: TrainingMethod;
  dataRoot: string;
  onBack: () => void;
}

export function TrainingWizard({ method, dataRoot, onBack }: TrainingWizardProps) {
  const methodInfo = TRAINING_METHODS.find((m) => m.id === method)!;
  const command = useForgeCommand();
  const [step, setStep] = useState<Step>("config");
  const config = useTrainingConfig(method, dataRoot);
  const { shared, setShared, extra, setExtra } = config;

  const missing = useMemo(
    () => getMissingFields(method, extra),
    [method, extra],
  );
  const canStart = missing.length === 0;

  async function startTraining() {
    if (!canStart) return;
    setStep("running");
    const args = buildTrainingArgs(method, shared, extra);
    const status = await command.run(dataRoot, args);
    if (status.status === "completed" && status.exit_code === 0) {
      setStep("done");
    }
  }

  const STEPS: { label: string; key: Step }[] = [
    { label: "Configure", key: "config" },
    { label: "Running", key: "running" },
    { label: "Results", key: "done" },
  ];

  if (!config.isLoaded) return null;

  return (
    <div>
      <div className="wizard-header">
        <button className="btn btn-ghost btn-sm" onClick={onBack}>
          <ArrowLeft size={14} /> Back
        </button>
        <h2>{methodInfo.name}</h2>
        <div className="spacer" />
        <div className="wizard-steps">
          {STEPS.map((s, i) => (
            <span key={s.key}>
              {i > 0 && (
                <span className="wizard-step-separator">
                  <ChevronRight size={12} />
                </span>
              )}
              <span className={`wizard-step${step === s.key ? " active" : ""}`}>
                {s.label}
              </span>
            </span>
          ))}
        </div>
      </div>

      {step === "config" && (
        <div className="panel stack-lg">
          {method === "train" && <BasicTrainForm extra={extra} setExtra={setExtra} />}
          {method === "sft" && <SftTrainForm extra={extra} setExtra={setExtra} />}
          {method === "dpo-train" && <DpoTrainForm extra={extra} setExtra={setExtra} />}
          {method === "rlhf-train" && <RlhfTrainForm extra={extra} setExtra={setExtra} />}
          {method === "lora-train" && <LoraTrainForm extra={extra} setExtra={setExtra} />}
          {method === "distill" && <DistillTrainForm extra={extra} setExtra={setExtra} />}
          {method === "domain-adapt" && <DomainAdaptForm extra={extra} setExtra={setExtra} />}
          {method === "grpo-train" && <GrpoTrainForm extra={extra} setExtra={setExtra} />}
          {method === "qlora-train" && <QloraTrainForm extra={extra} setExtra={setExtra} />}
          {method === "kto-train" && <KtoTrainForm extra={extra} setExtra={setExtra} />}
          {method === "orpo-train" && <OrpoTrainForm extra={extra} setExtra={setExtra} />}
          {method === "multimodal-train" && <MultimodalTrainForm extra={extra} setExtra={setExtra} />}
          {method === "rlvr-train" && <RlvrTrainForm extra={extra} setExtra={setExtra} />}

          <SharedTrainingFields config={shared} onChange={setShared} />

          {!canStart && (
            <div className="error-alert">
              Missing required fields: {missing.join(", ")}
            </div>
          )}
          <div className="flex-row">
            <button
              className="btn btn-primary btn-lg"
              onClick={() => startTraining().catch(console.error)}
              disabled={!canStart}
            >
              Start Training
            </button>
            <button
              className="btn btn-ghost btn-sm"
              onClick={config.resetToDefaults}
              title="Reset to defaults"
            >
              <RotateCcw size={12} /> Reset
            </button>
          </div>
        </div>
      )}

      {step === "running" && (
        <TrainingRunMonitor command={command} />
      )}

      {step === "done" && (
        <div className="panel stack-lg">
          <h3 className="text-success">Training Complete</h3>
          {command.status && (
            <div className="stats-grid">
              <div className="metric-card">
                <span className="metric-label">Status</span>
                <span className="metric-value text-success">
                  {command.status.status}
                </span>
              </div>
              <div className="metric-card">
                <span className="metric-label">Duration</span>
                <span className="metric-value">{command.status.elapsed_seconds}s</span>
              </div>
            </div>
          )}
          <pre className="console">{command.output}</pre>
          <div className="row gap-top">
            <button className="btn" onClick={onBack}>New Training</button>
          </div>
        </div>
      )}
    </div>
  );
}
