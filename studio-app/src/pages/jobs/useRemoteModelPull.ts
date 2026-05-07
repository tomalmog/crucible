import { useCallback, useEffect, useRef, useState } from "react";
import { getCrucibleCommandStatus, startCrucibleCommand } from "../../api/studioApi";
import type { JobRecord } from "../../types/jobs";

interface UseRemoteModelPullOptions {
  dataRoot: string;
  job: JobRecord;
  refreshModels: () => Promise<void>;
}

interface RemoteModelPullState {
  handlePull: () => Promise<void>;
  pullDone: boolean;
  pullError: string | null;
  pulling: boolean;
  pullProgress: string[];
}

export function useRemoteModelPull({
  dataRoot,
  job,
  refreshModels,
}: UseRemoteModelPullOptions): RemoteModelPullState {
  const [pullProgress, setPullProgress] = useState<string[]>([]);
  const [pullDone, setPullDone] = useState(false);
  const [pullError, setPullError] = useState<string | null>(null);
  const [pulling, setPulling] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearPullPoll = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const handlePull = useCallback(async () => {
    if (!dataRoot || !job.backendJobId) return;
    setPulling(true);
    setPullProgress([]);
    setPullError(null);
    setPullDone(false);
    try {
      const pullArgs = ["remote", "pull-model", "--job-id", job.backendJobId];
      if (job.modelName) pullArgs.push("--model-name", job.modelName);
      const task = await startCrucibleCommand(dataRoot, pullArgs);
      clearPullPoll();
      pollRef.current = setInterval(async () => {
        try {
          const status = await getCrucibleCommandStatus(task.task_id);
          const lines = (status.stdout || "")
            .split("\n")
            .filter((line: string) => line.startsWith("CRUCIBLE_PULL_PROGRESS: "))
            .map((line: string) => line.replace("CRUCIBLE_PULL_PROGRESS: ", ""));
          if (lines.length > 0) setPullProgress(lines);
          if (status.status !== "running") {
            clearPullPoll();
            if (status.status === "completed") {
              setPullDone(true);
              refreshModels().catch(console.error);
            } else {
              setPullError(status.stderr || "Pull failed");
            }
            setPulling(false);
          }
        } catch {
          clearPullPoll();
          setPulling(false);
          setPullError("Lost connection to pull task");
        }
      }, 2_000);
    } catch (err) {
      setPulling(false);
      setPullError(`Failed to start pull: ${err}`);
    }
  }, [clearPullPoll, dataRoot, job.backendJobId, job.modelName, refreshModels]);

  // Clear the pull polling interval when the row leaves the page.
  useEffect(() => () => clearPullPoll(), [clearPullPoll]);

  return { handlePull, pullDone, pullError, pulling, pullProgress };
}
