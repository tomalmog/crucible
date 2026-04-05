import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { X } from "lucide-react";
import { useCrucible } from "../../context/CrucibleContext";
import { startCrucibleCommand } from "../../api/studioApi";
import { buildIngestArgs } from "../../api/commandArgs";
import { FormField } from "./FormField";
import { PathInput } from "./PathInput";

interface IngestModalProps {
  onClose: () => void;
}

export function IngestModal({ onClose }: IngestModalProps) {
  const { dataRoot } = useCrucible();
  const navigate = useNavigate();
  const [source, setSource] = useState("");
  const [dataset, setDataset] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onClose]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!source.trim() || !dataset.trim()) return;
    setSubmitting(true);
    setError(null);
    try {
      const args = buildIngestArgs(source.trim(), dataset.trim(), {});
      const label = `Ingest · ${dataset.trim()}`;
      await startCrucibleCommand(dataRoot, args, label);
      onClose();
      navigate("/jobs", { state: { statusFilter: "running" } });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start ingest");
      setSubmitting(false);
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="confirm-modal" style={{ width: 480 }} onClick={(e) => e.stopPropagation()}>
        <div className="confirm-modal-header">
          <h3 className="confirm-modal-title">Ingest Dataset</h3>
          <button className="btn btn-ghost btn-sm btn-icon" onClick={onClose}>
            <X size={16} />
          </button>
        </div>
        <form onSubmit={(e) => handleSubmit(e).catch(console.error)}>
          <div className="confirm-modal-body" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <FormField label="Dataset name">
              <input
                value={dataset}
                onChange={(e) => setDataset(e.currentTarget.value)}
                placeholder="my-dataset"
                disabled={submitting}
              />
            </FormField>
            <FormField label="Source path or URL">
              <PathInput
                value={source}
                onChange={setSource}
                placeholder="/path/to/data.jsonl or s3://bucket/prefix"
                kind="file"
                filters={[{ name: "Data files", extensions: ["jsonl", "json", "csv", "parquet"] }]}
                disabled={submitting}
              />
            </FormField>

            {error && <p className="error-text">{error}</p>}
          </div>
          <div className="confirm-modal-footer">
            <button type="button" className="btn btn-sm" onClick={onClose}>Cancel</button>
            <button
              type="submit"
              className="btn btn-sm btn-primary"
              disabled={submitting || !source.trim() || !dataset.trim()}
            >
              {submitting ? "Starting..." : "Start Ingest"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
