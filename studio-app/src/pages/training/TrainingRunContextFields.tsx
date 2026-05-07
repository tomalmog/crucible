import { FormField } from "../../components/shared/FormField";
import { FormSection } from "../../components/shared/FormSection";

interface TrainingRunContextFieldsProps {
  evalObjective: string;
  projectName: string;
  setEvalObjective: (value: string) => void;
  setProjectName: (value: string) => void;
}

export function TrainingRunContextFields({
  evalObjective,
  projectName,
  setEvalObjective,
  setProjectName,
}: TrainingRunContextFieldsProps) {
  return (
    <FormSection title="Run Context">
      <div className="grid-2">
        <FormField label="Project" required>
          <input
            value={projectName}
            onChange={(event) => setProjectName(event.currentTarget.value)}
            placeholder="Default Project"
          />
        </FormField>
        <FormField label="Success Metric" required>
          <input
            value={evalObjective}
            onChange={(event) => setEvalObjective(event.currentTarget.value)}
            placeholder="What must improve before this model can ship?"
          />
        </FormField>
      </div>
    </FormSection>
  );
}
