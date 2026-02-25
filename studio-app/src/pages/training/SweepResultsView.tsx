interface SweepResult {
  trial: number;
  lr: string;
  batchSize: string;
  loss: string;
  isBest: boolean;
}

interface SweepResultsViewProps {
  output: string;
}

export function SweepResultsView({ output }: SweepResultsViewProps) {
  const lines = output.split("\n").filter((l) => l.trim());
  const results: SweepResult[] = lines
    .filter((l) => l.includes("trial="))
    .map((l, i) => {
      const getValue = (key: string) => {
        const match = l.match(new RegExp(`${key}=([^\\s]+)`));
        return match?.[1] ?? "-";
      };
      return {
        trial: i + 1,
        lr: getValue("lr"),
        batchSize: getValue("batch_size"),
        loss: getValue("loss"),
        isBest: l.includes("best=true"),
      };
    });

  return (
    <div className="panel">
      <h3>Sweep Results</h3>
      {results.length === 0 ? (
        <pre className="console">{output}</pre>
      ) : (
        <table className="data-table">
          <thead>
            <tr>
              <th>Trial</th>
              <th>Learning Rate</th>
              <th>Batch Size</th>
              <th>Loss</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {results.map((r) => (
              <tr key={r.trial} className={r.isBest ? "row-highlight" : ""}>
                <td>{r.trial}</td>
                <td>{r.lr}</td>
                <td>{r.batchSize}</td>
                <td>{r.loss}</td>
                <td>{r.isBest ? "Best" : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
