interface ModelVersionListProps {
  output: string;
}

export function ModelVersionList({ output }: ModelVersionListProps) {
  return (
    <div className="panel">
      <h3 className="panel-title">Registered Models</h3>
      <pre className="console console-tall">{output}</pre>
    </div>
  );
}
