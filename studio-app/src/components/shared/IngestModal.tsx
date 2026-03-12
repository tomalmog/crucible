import { useEffect, useState } from "react";
import { X } from "lucide-react";
import { useCrucible } from "../../context/CrucibleContext";
import { useCrucibleCommand } from "../../hooks/useCrucibleCommand";
import { buildIngestArgs } from "../../api/commandArgs";
import { FormField } from "./FormField";
import { PathInput } from "./PathInput";
import { StatusConsole } from "./StatusConsole";
import { CommandProgress } from "./CommandProgress";

interface IngestModalProps {
  onComplete: () => void;
  onClose: () => void;
}

export function IngestModal({ onComplete, onClose }: IngestModalProps) {
  const { dataRoot, refreshDatasets } = useCrucible();
  const command = useCrucibleCommand();
  const [source, setSource] = useState("");
  const [dataset, setDataset] = useState("");

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
    if (!source.trim() || !dataset.trim()) return;
    const args = buildIngestArgs(source.trim(), dataset.trim(), {});
    const result = await command.run(dataRoot, args);
    if (result.status === "completed") {
      await refreshDatasets();
      onComplete();
    }
  }

  return (
    <div className="modal-backdrop" onClick={canClose ? onClose : undefined}>
      <div className="confirm-modal" style={{ width: 480 }} onClick={(e) => e.stopPropagation()}>
        <div className="confirm-modal-header">
          <h3 className="confirm-modal-title">Ingest Dataset</h3>
          {canClose && (
            <button className="btn btn-ghost btn-sm btn-icon" onClick={onClose}>
              <X size={16} />
            </button>
          )}
        </div>
        <form onSubmit={(e) => handleSubmit(e).catch(console.error)}>
          <div className="confirm-modal-body" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <FormField label="Dataset name">
              <input
                value={dataset}
                onChange={(e) => setDataset(e.currentTarget.value)}
                placeholder="my-dataset"
                disabled={busy}
              />
            </FormField>
            <FormField label="Source path or URL">
              <PathInput
                value={source}
                onChange={setSource}
                placeholder="/path/to/data.jsonl or s3://bucket/prefix"
                kind="file"
                filters={[{ name: "Data files", extensions: ["jsonl", "json", "csv", "parquet"] }]}
                disabled={busy}
              />
            </FormField>

            {busy && command.status && (
              <CommandProgress
                label="Ingesting data..."
                percent={command.status.progress_percent}
                elapsed={command.status.elapsed_seconds}
                remaining={command.status.remaining_seconds}
              />
            )}

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
              disabled={busy || !source.trim() || !dataset.trim()}
            >
              {busy ? "Ingesting..." : "Start Ingest"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
