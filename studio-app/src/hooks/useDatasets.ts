import { useCrucible } from "../context/CrucibleContext";

export function useDatasets() {
  const { datasets, selectedDataset, setSelectedDataset,
    dashboard, samples, refreshDatasets, dataRoot } = useCrucible();

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
