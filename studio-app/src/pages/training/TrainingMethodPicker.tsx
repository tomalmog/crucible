import { useNavigate } from "react-router";
import { BookOpen } from "lucide-react";
import { TRAINING_METHODS, TRAINING_METHOD_CATEGORIES, TrainingMethod } from "../../types/training";
import { TRAINING_METHOD_ANCHORS } from "../docs/docsRegistry";

interface TrainingMethodPickerProps {
  onSelect: (method: TrainingMethod) => void;
}

const CATEGORY_LABELS: Record<string, string> = Object.fromEntries(
  TRAINING_METHOD_CATEGORIES.map((c) => [c.id, c.label]),
);

export function TrainingMethodPicker({ onSelect }: TrainingMethodPickerProps) {
  const navigate = useNavigate();

  return (
    <div className="method-grid">
      {TRAINING_METHODS.map((method) => (
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
  );
}
