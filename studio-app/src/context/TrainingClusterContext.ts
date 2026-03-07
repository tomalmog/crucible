import { createContext, useContext } from "react";

/** Provides the active cluster name to nested components (e.g. DatasetSelect).
 *  Empty string means no cluster is selected (local training). */
export const TrainingClusterContext = createContext("");

export function useTrainingCluster(): string {
  return useContext(TrainingClusterContext);
}
