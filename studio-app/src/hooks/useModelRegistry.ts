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

  const tagModel = useCallback(async (versionId: string, tag: string) => {
    return command.run(dataRoot, ["model", "tag", "--version-id", versionId, "--tag", tag]);
  }, [dataRoot, command]);

  const rollbackModel = useCallback(async (versionId: string) => {
    return command.run(dataRoot, ["model", "rollback", "--version-id", versionId]);
  }, [dataRoot, command]);

  const diffModel = useCallback(async (versionA: string, versionB: string) => {
    return command.run(dataRoot, ["model", "diff", "--version-a", versionA, "--version-b", versionB]);
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
