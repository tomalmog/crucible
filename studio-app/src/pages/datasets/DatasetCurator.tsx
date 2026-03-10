import { useState } from "react";
import { useCrucibleCommand } from "../../hooks/useCrucibleCommand";
import { useCrucible } from "../../context/CrucibleContext";
import { FormField } from "../../components/shared/FormField";

export function DatasetCurator() {
  const { dataRoot, selectedDataset } = useCrucible();
  const command = useCrucibleCommand();
  const [dataset, setDataset] = useState(selectedDataset ?? "");
  const [scoreResults, setScoreResults] = useState("");
  const [statsResults, setStatsResults] = useState("");
  const [loading, setLoading] = useState(false);

  async function runScore() {
    if (!dataRoot || !dataset.trim()) return;
    setLoading(true);
    const status = await command.run(dataRoot, ["curate", "score", "--dataset", dataset]);
    if (status.status === "completed" && command.output) {
      setScoreResults(command.output);
    }
    setLoading(false);
  }

  async function runStats() {
    if (!dataRoot || !dataset.trim()) return;
    setLoading(true);
    const status = await command.run(dataRoot, ["curate", "stats", "--dataset", dataset]);
    if (status.status === "completed" && command.output) {
      setStatsResults(command.output);
    }
    setLoading(false);
  }

  return (
    <div className="panel stack-md">
      <h3>Dataset Curator</h3>
      <FormField label="Dataset">
        <input value={dataset} onChange={(e) => setDataset(e.currentTarget.value)} placeholder="my-dataset" />
      </FormField>
      <div className="row">
        <button className="btn" onClick={() => runScore().catch(console.error)} disabled={loading || !dataset.trim()}>
          Score Quality
        </button>
        <button className="btn" onClick={() => runStats().catch(console.error)} disabled={loading || !dataset.trim()}>
          Compute Stats
        </button>
      </div>
      {scoreResults && (
        <div>
          <h4>Quality Scores</h4>
          <pre className="console">{scoreResults}</pre>
        </div>
      )}
      {statsResults && (
        <div>
          <h4>Distribution Stats</h4>
          <pre className="console">{statsResults}</pre>
        </div>
      )}
    </div>
  );
}
