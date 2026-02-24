import { useCallback, useState } from "react";
import { useForgeCommand } from "./useForgeCommand";

export function useModelRegistry(dataRoot: string) {
  const command = useForgeCommand();
  const [listOutput, setListOutput] = useState("");

  const listModels = useCallback(async () => {
    const status = await command.run(dataRoot, ["model", "list"]);
    setListOutput(status.stdout);
    return status;
  }, [dataRoot, command]);

  const tagModel = useCallback(async (name: string, version: string, tag: string) => {
    return command.run(dataRoot, ["model", "tag", "--name", name, "--version", version, "--tag", tag]);
  }, [dataRoot, command]);

  const rollbackModel = useCallback(async (name: string, version: string) => {
    return command.run(dataRoot, ["model", "rollback", "--name", name, "--version", version]);
  }, [dataRoot, command]);

  const diffModel = useCallback(async (name: string, versionA: string, versionB: string) => {
    return command.run(dataRoot, ["model", "diff", "--name", name, "--version-a", versionA, "--version-b", versionB]);
  }, [dataRoot, command]);

  return {
    listModels,
    tagModel,
    rollbackModel,
    diffModel,
    listOutput,
    isRunning: command.isRunning,
    output: command.output,
    error: command.error,
  };
}
