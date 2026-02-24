import { useCallback, useEffect, useState } from "react";
import { listTrainingRuns } from "../api/studioApi";
import { TrainingRunSummary } from "../types";

export function useTrainingRuns(dataRoot: string) {
  const [runs, setRuns] = useState<TrainingRunSummary[]>([]);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const result = await listTrainingRuns(dataRoot);
      setRuns(result);
    } catch (err) {
      console.error("Failed to load training runs:", err);
    } finally {
      setLoading(false);
    }
  }, [dataRoot]);

  useEffect(() => {
    refresh().catch(console.error);
  }, [refresh]);

  return { runs, loading, refresh };
}
