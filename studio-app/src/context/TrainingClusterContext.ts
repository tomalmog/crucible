import { createContext, useContext } from "react";

export interface TrainingClusterContextValue {
  /** The cluster name DatasetSelect should use for fetching remote datasets.
   *  Empty string = no cluster preference (DatasetSelect will pick the first). */
  cluster: string;
  /** Called by DatasetSelect when the user picks a dataset.
   *  Signals whether the pick was local or remote (with cluster name). */
  onDatasetLocationChanged?: (isRemote: boolean, cluster: string) => void;
}

const defaultValue: TrainingClusterContextValue = { cluster: "" };

export const TrainingClusterContext = createContext<TrainingClusterContextValue>(defaultValue);

export function useTrainingCluster(): TrainingClusterContextValue {
  return useContext(TrainingClusterContext);
}
