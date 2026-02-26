import { useCallback, useEffect, useRef, useState } from "react";
import { listForgeTasks, killForgeTask, renameForgeTask, deleteForgeTask } from "../api/studioApi";
import { CommandTaskStatus } from "../types";

const POLL_INTERVAL_MS = 2000;

export function useJobs() {
  const [jobs, setJobs] = useState<CommandTaskStatus[]>([]);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const refresh = useCallback(async () => {
    try {
      const result = await listForgeTasks();
      setJobs(result);
    } catch (err) {
      console.warn("[useJobs] polling error:", err);
    }
  }, []);

  const kill = useCallback(
    async (taskId: string) => {
      await killForgeTask(taskId);
      await refresh();
    },
    [refresh],
  );

  const rename = useCallback(
    async (taskId: string, label: string) => {
      await renameForgeTask(taskId, label);
      await refresh();
    },
    [refresh],
  );

  const remove = useCallback(
    async (taskId: string) => {
      await deleteForgeTask(taskId);
      await refresh();
    },
    [refresh],
  );

  // Poll for job status updates every 2 seconds
  useEffect(() => {
    refresh();
    intervalRef.current = setInterval(refresh, POLL_INTERVAL_MS);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [refresh]);

  return { jobs, refresh, kill, rename, remove };
}
