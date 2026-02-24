import { useState } from "react";
import { useForge } from "../../context/ForgeContext";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { buildFilterArgs } from "../../api/commandArgs";
import { FormField } from "../../components/shared/FormField";
import { StatusConsole } from "../../components/shared/StatusConsole";
import { CommandProgress } from "../../components/shared/CommandProgress";

export function FilterForm() {
  const { dataRoot, selectedDataset, refreshDatasets } = useForge();
  const command = useForgeCommand();
  const [minQuality, setMinQuality] = useState("0.0");
  const [language, setLanguage] = useState("");
  const [maxRecords, setMaxRecords] = useState("");

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!selectedDataset) return;
    const extra: Record<string, string> = {};
    if (minQuality.trim()) extra["--min-quality"] = minQuality.trim();
    if (language.trim()) extra["--language"] = language.trim();
    if (maxRecords.trim()) extra["--max-records"] = maxRecords.trim();
    const args = buildFilterArgs(selectedDataset, extra);
    await command.run(dataRoot, args);
    await refreshDatasets();
  }

  return (
    <div className="panel">
      <h3 className="panel-title">Filter Dataset</h3>
      <p className="text-tertiary">
        Filtering: {selectedDataset}
      </p>
      <form onSubmit={(e) => onSubmit(e).catch(console.error)} className="stack">
        <FormField label="Min quality score">
          <input value={minQuality} onChange={(e) => setMinQuality(e.currentTarget.value)} placeholder="0.0" />
        </FormField>
        <FormField label="Language filter (optional)">
          <input value={language} onChange={(e) => setLanguage(e.currentTarget.value)} placeholder="en" />
        </FormField>
        <FormField label="Max records (optional)">
          <input value={maxRecords} onChange={(e) => setMaxRecords(e.currentTarget.value)} placeholder="all" />
        </FormField>
        <button className="btn btn-primary btn-lg" type="submit" disabled={command.isRunning}>
          {command.isRunning ? "Filtering..." : "Apply Filter"}
        </button>
      </form>

      {command.isRunning && command.status && (
        <div className="gap-top-lg">
          <CommandProgress
            label="Filtering data..."
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
    </div>
  );
}
