interface JudgeScoreEntry {
  criteria: string;
  score: string;
  explanation: string;
}

interface LlmJudgeResultsProps {
  output: string;
}

export function LlmJudgeResults({ output }: LlmJudgeResultsProps) {
  const lines = output.split("\n").filter((l) => l.trim());
  const avgLine = lines.find((l) => l.startsWith("average_score="));
  const avgScore = avgLine?.replace("average_score=", "") ?? "-";

  const scores: JudgeScoreEntry[] = lines
    .filter((l) => l.startsWith("criteria="))
    .map((l) => {
      const getValue = (key: string) => {
        const match = l.match(new RegExp(`${key}=([^\\s]+)`));
        return match?.[1] ?? "-";
      };
      const explanation = l.match(/explanation=(.+)/)?.[1] ?? "";
      return {
        criteria: getValue("criteria"),
        score: getValue("score"),
        explanation,
      };
    });

  return (
    <div className="panel stack-md">
      <h3>Judge Results</h3>
      <div className="metric-card">
        <span className="metric-label">Average Score</span>
        <span className="metric-value">{avgScore}</span>
      </div>
      {scores.length > 0 && (
        <table className="data-table">
          <thead>
            <tr>
              <th>Criteria</th>
              <th>Score</th>
              <th>Explanation</th>
            </tr>
          </thead>
          <tbody>
            {scores.map((s) => (
              <tr key={s.criteria}>
                <td>{s.criteria}</td>
                <td>{s.score}</td>
                <td>{s.explanation}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
