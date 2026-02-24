import { useState } from "react";
import { useForge } from "../../context/ForgeContext";
import { versionDiff } from "../../api/studioApi";
import { VersionDiff } from "../../types";
import { MetricCard } from "../../components/shared/MetricCard";

export function VersionDiffPanel() {
  const { dataRoot, selectedDataset, versions } = useForge();
  const [baseVersion, setBaseVersion] = useState<string>("");
  const [targetVersion, setTargetVersion] = useState<string>("");
  const [diff, setDiff] = useState<VersionDiff | null>(null);

  async function computeDiff() {
    if (!selectedDataset || !baseVersion || !targetVersion) return;
    const result = await versionDiff(dataRoot, selectedDataset, baseVersion, targetVersion);
    setDiff(result);
  }

  return (
    <div className="panel">
      <h3 className="panel-title">Version Diff</h3>
      <div className="diff-controls">
        <label>
          Base
          <select value={baseVersion} onChange={(e) => setBaseVersion(e.currentTarget.value)}>
            <option value="">Select base version</option>
            {versions.map((v) => (
              <option key={v.version_id} value={v.version_id}>{v.version_id.slice(0, 20)}</option>
            ))}
          </select>
        </label>
        <label>
          Target
          <select value={targetVersion} onChange={(e) => setTargetVersion(e.currentTarget.value)}>
            <option value="">Select target version</option>
            {versions.map((v) => (
              <option key={v.version_id} value={v.version_id}>{v.version_id.slice(0, 20)}</option>
            ))}
          </select>
        </label>
        <button className="btn btn-primary" onClick={() => computeDiff().catch(console.error)}>
          Compute
        </button>
      </div>

      {diff ? (
        <div className="stats-grid gap-top">
          <MetricCard label="Added" value={String(diff.added_records)} />
          <MetricCard label="Removed" value={String(diff.removed_records)} />
          <MetricCard label="Shared" value={String(diff.shared_records)} />
        </div>
      ) : (
        <p className="text-tertiary gap-top">
          Pick two versions and compute diff.
        </p>
      )}
    </div>
  );
}
