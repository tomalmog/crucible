import { useCallback, useEffect, useRef, useState } from "react";
import { listForgeTasks, killForgeTask } from "../api/studioApi";
import { CommandTaskStatus } from "../types";

const POLL_INTERVAL_MS = 2000;

export function useJobs() {
  const [jobs, setJobs] = useState<CommandTaskStatus[]>([]);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const refresh = useCallback(async () => {
    try {
      const result = await listForgeTasks();
      setJobs(result);
    } catch {
      // silently ignore polling errors
    }
  }, []);

  const kill = useCallback(
    async (taskId: string) => {
      await killForgeTask(taskId);
      await refresh();
    },
    [refresh],
  );

  useEffect(() => {
    refresh();
    intervalRef.current = setInterval(refresh, POLL_INTERVAL_MS);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [refresh]);

  return { jobs, refresh, kill };
}
