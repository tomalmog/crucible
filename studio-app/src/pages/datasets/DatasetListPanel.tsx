import { useForge } from "../../context/ForgeContext";

export function DatasetListPanel() {
  const { datasets, selectedDataset, setSelectedDataset, versions,
    selectedVersion, setSelectedVersion } = useForge();

  return (
    <div className="panel overflow-auto">
      <h3 className="panel-title">Datasets</h3>
      {datasets.length === 0 ? (
        <p className="text-tertiary">
          No datasets found. Use the Ingest tab to add data.
        </p>
      ) : (
        <div>
          {datasets.map((ds) => (
            <button
              key={ds}
              className={`nav-item ${ds === selectedDataset ? "active" : ""}`}
              onClick={() => { setSelectedDataset(ds); setSelectedVersion(null); }}
            >
              {ds}
            </button>
          ))}
        </div>
      )}

      {versions.length > 0 && (
        <>
          <h4 className="gap-top-lg">Versions</h4>
          <button
            className={`nav-item ${selectedVersion === null ? "active" : ""}`}
            onClick={() => setSelectedVersion(null)}
          >
            Latest
          </button>
          <div>
            {versions.map((v) => (
              <button
                key={v.version_id}
                className={`nav-item ${v.version_id === selectedVersion ? "active" : ""}`}
                onClick={() => setSelectedVersion(v.version_id)}
              >
                <span className="text-mono text-sm">
                  {v.version_id.slice(0, 16)}...
                </span>
                <span className="text-xs text-tertiary">
                  {v.record_count} rows
                </span>
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
