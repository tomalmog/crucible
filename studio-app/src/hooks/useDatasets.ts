import { useForge } from "../context/ForgeContext";

export function useDatasets() {
  const { datasets, selectedDataset, setSelectedDataset,
    dashboard, samples, refreshDatasets, dataRoot } = useForge();

  return {
    dataRoot,
    datasets,
    selectedDataset,
    setSelectedDataset,
    dashboard,
    samples,
    refreshDatasets,
  };
}
