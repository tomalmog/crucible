import { useState } from "react";
import { useForge } from "../../context/ForgeContext";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { buildIngestArgs } from "../../api/commandArgs";
import { FormField } from "../../components/shared/FormField";
import { PathInput } from "../../components/shared/PathInput";
import { StatusConsole } from "../../components/shared/StatusConsole";
import { CommandProgress } from "../../components/shared/CommandProgress";

export function IngestForm() {
  const { dataRoot, refreshDatasets } = useForge();
  const command = useForgeCommand();
  const [source, setSource] = useState("");
  const [dataset, setDataset] = useState("");

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!source.trim() || !dataset.trim()) return;
    const args = buildIngestArgs(source.trim(), dataset.trim(), {});
    await command.run(dataRoot, args);
    await refreshDatasets();
  }

  return (
    <div className="panel">
      <h3 className="panel-title">Ingest Data</h3>
      <form onSubmit={(e) => onSubmit(e).catch(console.error)} className="stack">
        <FormField label="Source path or URL">
          <PathInput
            value={source}
            onChange={setSource}
            placeholder="/path/to/data.jsonl or s3://bucket/prefix"
            kind="file"
            filters={[{ name: "Data files", extensions: ["jsonl", "json", "csv", "parquet"] }]}
          />
        </FormField>
        <FormField label="Dataset name">
          <input value={dataset} onChange={(e) => setDataset(e.currentTarget.value)} placeholder="my-dataset" />
        </FormField>
        <button className="btn btn-primary btn-lg" type="submit" disabled={command.isRunning || !source.trim() || !dataset.trim()}>
          {command.isRunning ? "Ingesting..." : "Start Ingest"}
        </button>
      </form>

      {command.isRunning && command.status && (
        <div className="gap-top-lg">
          <CommandProgress
            label="Ingesting data..."
            percent={command.status.progress_percent}
            elapsed={command.status.elapsed_seconds}
            remaining={command.status.remaining_seconds}
          />
        </div>
      )}

      {command.output && (
        <div className="gap-top">
          <StatusConsole output={command.output} />
        </div>
      )}

      {command.error && (
        <p className="error-text gap-top-sm">{command.error}</p>
      )}
    </div>
  );
}
