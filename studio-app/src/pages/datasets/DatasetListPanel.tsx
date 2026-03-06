import { useState } from "react";
import { Trash2 } from "lucide-react";
import { useForge } from "../../context/ForgeContext";
import { deleteDataset } from "../../api/studioApi";

interface DatasetListPanelProps {
  onSelect?: (dataset: string) => void;
}

export function DatasetListPanel({ onSelect }: DatasetListPanelProps) {
  const { dataRoot, datasets, selectedDataset, setSelectedDataset, refreshDatasets } = useForge();
  const [confirmingDelete, setConfirmingDelete] = useState<string | null>(null);

  async function handleDelete(ds: string) {
    if (confirmingDelete !== ds) {
      setConfirmingDelete(ds);
      return;
    }
    await deleteDataset(dataRoot, ds);
    setConfirmingDelete(null);
    await refreshDatasets();
  }

  return (
    <div className="panel overflow-auto">
      <h3 className="panel-title">Datasets</h3>
      {datasets.length === 0 ? (
        <p className="text-tertiary">
          No datasets found. Use the Ingest tab to add data.
        </p>
      ) : (
        <div>
          {datasets.map((ds) => (
            <div key={ds} className="flex-row" style={{ alignItems: "center" }}>
              <button
                className={`nav-item ${ds === selectedDataset ? "active" : ""}`}
                style={{ flex: 1 }}
                onClick={() => { if (onSelect) { onSelect(ds); } else { setSelectedDataset(ds); } setConfirmingDelete(null); }}
              >
                {ds}
              </button>
              {confirmingDelete === ds ? (
                <button
                  className="btn btn-sm"
                  style={{ color: "var(--color-error)", flexShrink: 0 }}
                  onClick={() => handleDelete(ds).catch(console.error)}
                >
                  Delete?
                </button>
              ) : (
                <button
                  className="btn btn-ghost btn-sm btn-icon"
                  style={{ flexShrink: 0 }}
                  onClick={() => handleDelete(ds).catch(console.error)}
                  title="Delete dataset"
                >
                  <Trash2 size={12} />
                </button>
              )}
            </div>
          ))}
        </div>
      )}

    </div>
  );
}
