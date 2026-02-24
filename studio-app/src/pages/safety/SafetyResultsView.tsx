interface SafetyResultsViewProps {
  output: string;
}

export function SafetyResultsView({ output }: SafetyResultsViewProps) {
  return (
    <div className="panel">
      <h3 className="panel-title">Results</h3>
      <pre className="console console-tall">{output}</pre>
    </div>
  );
}
