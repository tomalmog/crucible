import { useState, useEffect } from "react";
import { useForgeCommand } from "../../hooks/useForgeCommand";

interface ExperimentCompareProps {
  runIds: string[];
  dataRoot: string;
}

export function ExperimentCompare({ runIds, dataRoot }: ExperimentCompareProps) {
  const command = useForgeCommand();
  const [comparisons, setComparisons] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      setLoading(true);
      const status = await command.run(dataRoot, ["experiment", "compare", ...runIds]);
      if (status.status === "completed" && command.output) {
        try {
          setComparisons(JSON.parse(command.output));
        } catch {
          setComparisons([]);
        }
      }
      setLoading(false);
    }
    load().catch(console.error);
  }, [runIds.join(","), dataRoot]);

  if (loading) return <div className="panel"><p>Loading comparison...</p></div>;
  if (comparisons.length === 0) return <div className="panel"><p>No comparison data.</p></div>;

  const allMetrics = new Set<string>();
  for (const c of comparisons) {
    for (const name of c.metric_names ?? []) {
      allMetrics.add(name);
    }
  }

  return (
    <div className="panel stack-lg">
      <h2>Run Comparison</h2>
      <table className="data-table">
        <thead>
          <tr>
            <th>Metric</th>
            {comparisons.map((c) => (
              <th key={c.run_id}>{c.run_id}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from(allMetrics).map((metric) => (
            <tr key={metric}>
              <td>{metric} (final)</td>
              {comparisons.map((c) => (
                <td key={c.run_id}>{c[`${metric}_final`] ?? "-"}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
