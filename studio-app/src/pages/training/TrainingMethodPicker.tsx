import { TRAINING_METHODS, TrainingMethod } from "../../types/training";

interface TrainingMethodPickerProps {
  onSelect: (method: TrainingMethod) => void;
}

export function TrainingMethodPicker({ onSelect }: TrainingMethodPickerProps) {
  return (
    <div>
      <p className="page-description">
        Choose a training method to get started.
      </p>
      <div className="method-list">
        {TRAINING_METHODS.map((method) => (
          <button
            key={method.id}
            className="method-item"
            onClick={() => onSelect(method.id)}
          >
            <span className="method-item-name">{method.name}</span>
            <span className="method-item-description">{method.description}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
