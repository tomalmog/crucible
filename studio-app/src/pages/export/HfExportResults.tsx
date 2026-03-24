import type { HfExportResult } from "../../types/export";

interface HfExportResultsProps {
  result: HfExportResult;
}

export function HfExportResults({ result }: HfExportResultsProps) {
  return (
    <>
      <div className="stats-grid">
        <div className="metric-card">
          <span className="metric-label">File Size</span>
          <span className="metric-value">{result.file_size_mb} MB</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Tensors</span>
          <span className="metric-value">{result.num_tensors}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Config Generated</span>
          <span className="metric-value">{result.config_generated ? "Yes" : "No"}</span>
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
            <tr><td>Output Path</td><td className="text-mono text-sm">{result.output_path}</td></tr>
          </tbody>
        </table>
      </div>
    </>
  );
}
