import { createContext, useContext, useState, useCallback, ReactNode } from "react";
import { startCrucibleCommand, getCrucibleCommandStatus } from "../api/studioApi";
import { CommandTaskStatus } from "../types";

const POLL_INTERVAL_MS = 500;

interface ActiveTask {
  taskId: string;
  command: string;
  status: CommandTaskStatus | null;
}

interface CommandContextValue {
  activeTask: ActiveTask | null;
  runCommand: (dataRoot: string, args: string[]) => Promise<CommandTaskStatus>;
  isRunning: boolean;
}

const CommandCtx = createContext<CommandContextValue | null>(null);

export function useCommand(): CommandContextValue {
  const ctx = useContext(CommandCtx);
  if (!ctx) throw new Error("useCommand must be inside CommandProvider");
  return ctx;
}

export function CommandProvider({ children }: { children: ReactNode }) {
  const [activeTask, setActiveTask] = useState<ActiveTask | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  const runCommand = useCallback(
    async (dataRoot: string, args: string[]): Promise<CommandTaskStatus> => {
      setIsRunning(true);
      const command = args[0] ?? "unknown";
      const taskStart = await startCrucibleCommand(dataRoot, args);
      setActiveTask({ taskId: taskStart.task_id, command, status: null });

      let status: CommandTaskStatus;
      while (true) {
        status = await getCrucibleCommandStatus(taskStart.task_id);
        setActiveTask({ taskId: taskStart.task_id, command, status });
        if (status.status !== "running") break;
        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
      }

      setIsRunning(false);
      return status;
    },
    [],
  );

  return (
    <CommandCtx.Provider value={{ activeTask, runCommand, isRunning }}>
      {children}
    </CommandCtx.Provider>
  );
}
