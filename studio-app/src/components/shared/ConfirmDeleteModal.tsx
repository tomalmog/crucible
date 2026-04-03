import { useEffect } from "react";
import { X } from "lucide-react";

interface CheckboxOption {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}

interface ConfirmDeleteModalProps {
  title: string;
  itemName: string;
  description?: string;
  checkbox?: CheckboxOption;
  warning?: string;
  isDeleting: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmDeleteModal({
  title, itemName, description, checkbox, warning, isDeleting, onConfirm, onCancel,
}: ConfirmDeleteModalProps) {
  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (e.key === "Escape" && !isDeleting) onCancel();
    }
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [isDeleting, onCancel]);

  return (
    <div className="modal-backdrop" onClick={isDeleting ? undefined : onCancel}>
      <div className="confirm-modal" onClick={(e) => e.stopPropagation()}>
        <div className="confirm-modal-header">
          <h3 className="confirm-modal-title">{title}</h3>
          {!isDeleting && (
            <button className="btn btn-ghost btn-sm btn-icon" onClick={onCancel}>
              <X size={16} />
            </button>
          )}
        </div>
        <div className="confirm-modal-body">
          <p>
            Are you sure you want to delete <strong>{itemName}</strong>?
            {description && ` ${description}`}
          </p>
          {checkbox && (
            <label className="error-alert-prominent checkbox-row" style={{ marginTop: 12 }}>
              <input
                type="checkbox"
                checked={checkbox.checked}
                disabled={isDeleting}
                onChange={(e) => checkbox.onChange(e.currentTarget.checked)}
              />
              <span>{checkbox.label}</span>
            </label>
          )}
          {warning && (
            <div className="warning-alert" style={{ marginTop: 12 }}>{warning}</div>
          )}
        </div>
        <div className="confirm-modal-footer">
          {!isDeleting && (
            <button className="btn btn-sm" onClick={onCancel}>Cancel</button>
          )}
          <button className="btn btn-sm btn-error" onClick={onConfirm} disabled={isDeleting}>
            {isDeleting ? "Deleting..." : "Delete"}
          </button>
        </div>
      </div>
    </div>
  );
}
