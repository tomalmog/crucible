import { useNavigate } from "react-router";
import type { ReactNode } from "react";
import { BookOpen, Cpu, Database, Layers, Scale, SlidersHorizontal, Target } from "lucide-react";
import {
  ADVANCED_TRAINING_METHOD_IDS,
  FINE_TUNING_GOALS,
  TRAINING_METHODS,
  TRAINING_METHOD_CATEGORIES,
  TrainingMethod,
} from "../../types/training";
import { TRAINING_METHOD_ANCHORS } from "../docs/docsRegistry";

interface TrainingMethodPickerProps {
  onSelect: (method: TrainingMethod) => void;
}

const CATEGORY_LABELS: Record<string, string> = Object.fromEntries(
  TRAINING_METHOD_CATEGORIES.map((c) => [c.id, c.label]),
);

export function TrainingMethodPicker({ onSelect }: TrainingMethodPickerProps) {
  const navigate = useNavigate();
  const advancedMethods = TRAINING_METHODS.filter((method) =>
    ADVANCED_TRAINING_METHOD_IDS.has(method.id),
  );

  return (
    <div className="stack-lg">
      <section className="workflow-intro">
        <div>
          <h2>Choose the model behavior you want to improve</h2>
          <p>
            Crucible will pick a practical fine-tuning path first. The underlying
            training method is still visible before launch.
          </p>
        </div>
      </section>

      <div className="method-grid goal-grid">
        {FINE_TUNING_GOALS.map((goal) => (
          <button
            key={goal.id}
            type="button"
            className="method-card goal-card"
            onClick={() => onSelect(goal.method)}
          >
            <span className="goal-card-icon">{goalIcon(goal.id)}</span>
            <span className="method-card-name">{goal.title}</span>
            <span className="method-card-description">{goal.description}</span>
            <span className="goal-card-outcome">{goal.outcome}</span>
            <span className="method-card-footer">
              <span className="method-card-tag">{goal.badge}</span>
              <span className="method-card-docs">
                Configure <SlidersHorizontal size={12} />
              </span>
            </span>
          </button>
        ))}
      </div>

      <details className="advanced-methods">
        <summary>Advanced methods</summary>
        <div className="method-grid method-grid-compact">
          {advancedMethods.map((method) => (
            <div
              key={method.id}
              className="method-card"
              onClick={() => onSelect(method.id)}
            >
              <span className="method-card-name">{method.name}</span>
              <span className="method-card-description">{method.description}</span>
              <div className="method-card-footer">
                <span className="method-card-tag">
                  {CATEGORY_LABELS[method.category] ?? method.category}
                </span>
                <button
                  className="method-card-docs"
                  onClick={(e) => {
                    e.stopPropagation();
                    navigate(`/docs?doc=${TRAINING_METHOD_ANCHORS[method.id]}`);
                  }}
                >
                  <BookOpen size={12} />
                  Docs
                </button>
              </div>
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}

function goalIcon(goalId: string): ReactNode {
  switch (goalId) {
    case "task-accuracy":
      return <Target size={16} />;
    case "private-efficient":
      return <Cpu size={16} />;
    case "large-model-efficient":
      return <Layers size={16} />;
    case "preferred-answers":
      return <Scale size={16} />;
    case "domain-knowledge":
      return <Database size={16} />;
    default:
      return <Target size={16} />;
  }
}
