import type { OnnxExportResult } from "../../types/export";

interface OnnxExportResultsProps {
  result: OnnxExportResult;
}

export function OnnxExportResults({ result }: OnnxExportResultsProps) {
  const verificationPassed = result.verification === "passed";

  return (
    <>
      <div className="stats-grid">
        <div className="metric-card">
          <span className="metric-label">File Size</span>
          <span className="metric-value">{result.file_size_mb} MB</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Opset Version</span>
          <span className="metric-value">{result.opset_version}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Verification</span>
          <span className="metric-value" style={{ color: verificationPassed ? "var(--clr-success)" : "var(--clr-error)" }}>
            {verificationPassed ? "Passed" : "Failed"}
          </span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Tokenizer</span>
          <span className="metric-value">{result.tokenizer_copied ? "Copied" : "Not found"}</span>
        </div>
      </div>
      <div className="docs-table-wrap">
        <table className="docs-table">
          <thead><tr><th>Field</th><th>Value</th></tr></thead>
          <tbody>
            <tr><td>ONNX Path</td><td className="text-mono text-sm">{result.onnx_path}</td></tr>
            <tr><td>Input Names</td><td className="text-mono text-sm">{result.input_names.join(", ")}</td></tr>
            <tr><td>Output Names</td><td className="text-mono text-sm">{result.output_names.join(", ")}</td></tr>
          </tbody>
        </table>
      </div>
    </>
  );
}
