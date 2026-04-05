import { useEffect, useState } from "react";
import { X } from "lucide-react";
import { useCrucible } from "../../context/CrucibleContext";
import { useCrucibleCommand } from "../../hooks/useCrucibleCommand";
import { FormField } from "../../components/shared/FormField";
import { PathInput } from "../../components/shared/PathInput";

interface CreateBenchmarkModalProps {
  onCreated: () => void;
  onClose: () => void;
}

/** Extract the final error message from a Python traceback. */
function extractError(raw: string): string {
  const lines = raw.trim().split("\n");
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i].trim();
    if (line.match(/^[\w.]*Error:\s/) || line.match(/^[\w.]*Exception:\s/)) {
      return line.replace(/^[\w.]*(?:Error|Exception):\s*/, "");
    }
  }
  return raw;
}

export function CreateBenchmarkModal({ onCreated, onClose }: CreateBenchmarkModalProps) {
  const { dataRoot } = useCrucible();
  const command = useCrucibleCommand();
  const [name, setName] = useState("");
  const [source, setSource] = useState("");

  const busy = command.isRunning;
  const canClose = !busy;

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (e.key === "Escape" && canClose) onClose();
    }
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [canClose, onClose]);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim() || !source.trim()) return;
    const args = ["benchmark-registry", "create", "--name", name.trim(), "--source", source.trim()];
    const result = await command.run(dataRoot, args);
    if (result.status === "completed") {
      onCreated();
    }
  }

  return (
    <div className="modal-backdrop" onClick={canClose ? onClose : undefined}>
      <div className="confirm-modal" style={{ width: 480 }} onClick={(e) => e.stopPropagation()}>
        <div className="confirm-modal-header">
          <h3 className="confirm-modal-title">New Benchmark</h3>
          {canClose && (
            <button className="btn btn-ghost btn-sm btn-icon" onClick={onClose}>
              <X size={16} />
            </button>
          )}
        </div>
        <form onSubmit={(e) => handleCreate(e).catch(console.error)}>
          <div className="confirm-modal-body" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <FormField label="Name">
              <input
                value={name}
                onChange={(e) => setName(e.currentTarget.value)}
                placeholder="my-benchmark"
                disabled={busy}
              />
            </FormField>
            <FormField label="Data file" hint="JSONL with prompt and response fields">
              <PathInput
                value={source}
                onChange={setSource}
                placeholder="/path/to/questions.jsonl"
                kind="file"
                filters={[{ name: "JSONL files", extensions: ["jsonl"] }]}
                disabled={busy}
              />
            </FormField>

            {!busy && command.error && (
              <p className="error-text">{extractError(command.error)}</p>
            )}
          </div>
          <div className="confirm-modal-footer">
            {canClose && (
              <button type="button" className="btn btn-sm" onClick={onClose}>Cancel</button>
            )}
            <button
              type="submit"
              className="btn btn-sm btn-primary"
              disabled={busy || !name.trim() || !source.trim()}
            >
              {busy ? "Creating..." : "Create"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
