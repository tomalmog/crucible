interface StatusConsoleProps {
  output: string;
  title?: string;
}

export function StatusConsole({ output, title }: StatusConsoleProps) {
  return (
    <div className="panel">
      {title && <h3 className="panel-title">{title}</h3>}
      <pre className="console">{output || "No output yet."}</pre>
    </div>
  );
}
