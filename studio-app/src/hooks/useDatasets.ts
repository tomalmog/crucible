import { useForge } from "../context/ForgeContext";

export function useDatasets() {
  const { datasets, selectedDataset, setSelectedDataset, versions,
    selectedVersion, setSelectedVersion, dashboard, samples,
    refreshDatasets, dataRoot } = useForge();

  return {
    dataRoot,
    datasets,
    selectedDataset,
    setSelectedDataset,
    versions,
    selectedVersion,
    setSelectedVersion,
    dashboard,
    samples,
    refreshDatasets,
  };
}
