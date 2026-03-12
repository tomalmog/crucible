import { useEffect, useState } from "react";
import { X } from "lucide-react";
import { useCrucible } from "../../context/CrucibleContext";
import { useCrucibleCommand } from "../../hooks/useCrucibleCommand";
import { FormField } from "./FormField";
import { PathInput } from "./PathInput";
import { StatusConsole } from "./StatusConsole";

interface RegisterModelModalProps {
  onComplete: () => void;
  onClose: () => void;
}

export function RegisterModelModal({ onComplete, onClose }: RegisterModelModalProps) {
  const { dataRoot, refreshModels } = useCrucible();
  const command = useCrucibleCommand();
  const [name, setName] = useState("");
  const [modelPath, setModelPath] = useState("");

  const busy = command.isRunning;
  const canClose = !busy;

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (e.key === "Escape" && canClose) onClose();
    }
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [canClose, onClose]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim() || !modelPath.trim()) return;
    const result = await command.run(dataRoot, ["model", "register", "--name", name.trim(), "--model-path", modelPath.trim()]);
    if (result.status === "completed") {
      await refreshModels();
      onComplete();
    }
  }

  return (
    <div className="modal-backdrop" onClick={canClose ? onClose : undefined}>
      <div className="confirm-modal" style={{ width: 480 }} onClick={(e) => e.stopPropagation()}>
        <div className="confirm-modal-header">
          <h3 className="confirm-modal-title">Register Model</h3>
          {canClose && (
            <button className="btn btn-ghost btn-sm btn-icon" onClick={onClose}>
              <X size={16} />
            </button>
          )}
        </div>
        <form onSubmit={(e) => handleSubmit(e).catch(console.error)}>
          <div className="confirm-modal-body" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <FormField label="Model name">
              <input
                value={name}
                onChange={(e) => setName(e.currentTarget.value)}
                placeholder="my-model"
                disabled={busy}
              />
            </FormField>
            <FormField label="Model path">
              <PathInput
                value={modelPath}
                onChange={setModelPath}
                placeholder="/path/to/model/directory"
                kind="folder"
                disabled={busy}
              />
            </FormField>

            {command.output && <StatusConsole output={command.output} />}
            {command.error && <p className="error-text">{command.error}</p>}
          </div>
          <div className="confirm-modal-footer">
            {canClose && (
              <button type="button" className="btn btn-sm" onClick={onClose}>Cancel</button>
            )}
            <button
              type="submit"
              className="btn btn-sm btn-primary"
              disabled={busy || !name.trim() || !modelPath.trim()}
            >
              {busy ? "Registering..." : "Register"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
