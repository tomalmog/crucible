import { useState, useCallback } from "react";
import { startCrucibleCommand, getCrucibleCommandStatus } from "../api/studioApi";
import { CommandTaskStatus } from "../types";

const POLL_MS = 400;

interface CrucibleCommandState {
  isRunning: boolean;
  status: CommandTaskStatus | null;
  output: string;
  error: string | null;
  run: (dataRoot: string, args: string[], label?: string, config?: Record<string, unknown>) => Promise<CommandTaskStatus>;
  reset: () => void;
}

export function useCrucibleCommand(): CrucibleCommandState {
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState<CommandTaskStatus | null>(null);
  const [output, setOutput] = useState("");
  const [error, setError] = useState<string | null>(null);

  const reset = useCallback(() => {
    setIsRunning(false);
    setStatus(null);
    setOutput("");
    setError(null);
  }, []);

  const run = useCallback(async (dataRoot: string, args: string[], label?: string, config?: Record<string, unknown>): Promise<CommandTaskStatus> => {
    setIsRunning(true);
    setError(null);
    setOutput("");
    setStatus(null);

    try {
      const taskStart = await startCrucibleCommand(dataRoot, args, label, config);
      let taskStatus: CommandTaskStatus;

      while (true) {
        taskStatus = await getCrucibleCommandStatus(taskStart.task_id);
        setStatus(taskStatus);
        setOutput([taskStatus.stdout, taskStatus.stderr].filter(Boolean).join("\n"));
        if (taskStatus.status !== "running") break;
        await new Promise((r) => setTimeout(r, POLL_MS));
      }

      if (taskStatus.status === "failed") {
        setError(taskStatus.stderr || "Command failed");
      }
      setIsRunning(false);
      return taskStatus;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      setIsRunning(false);
      throw err;
    }
  }, []);

  return { isRunning, status, output, error, run, reset };
}
