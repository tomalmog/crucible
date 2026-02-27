import { useNavigate } from "react-router";
import { Info } from "lucide-react";
import { TRAINING_METHODS, TRAINING_METHOD_CATEGORIES, TrainingMethod } from "../../types/training";
import { TRAINING_METHOD_ANCHORS } from "../docs/docsRegistry";

interface TrainingMethodPickerProps {
  onSelect: (method: TrainingMethod) => void;
}

export function TrainingMethodPicker({ onSelect }: TrainingMethodPickerProps) {
  const navigate = useNavigate();

  return (
    <div>
      <p className="page-description">
        Choose a training method to get started.
      </p>
      <div className="method-list">
        {TRAINING_METHOD_CATEGORIES.map((category) => {
          const methods = TRAINING_METHODS.filter((m) => m.category === category.id);
          if (methods.length === 0) return null;
          return (
            <div key={category.id}>
              <div className="method-category-header">{category.label}</div>
              {methods.map((method) => (
                <div key={method.id} className="method-item-row">
                  <button
                    className="method-item"
                    onClick={() => onSelect(method.id)}
                  >
                    <span className="method-item-name">{method.name}</span>
                    <span className="method-item-description">{method.description}</span>
                  </button>
                  <button
                    className="btn btn-ghost btn-sm method-info-btn"
                    title={`View ${method.name} documentation`}
                    onClick={(e) => {
                      e.stopPropagation();
                      navigate(`/docs?doc=${TRAINING_METHOD_ANCHORS[method.id]}`);
                    }}
                  >
                    <Info size={14} />
                  </button>
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}
