import type { ModelVersion } from "../../types/models";

export function parseModelList(stdout: string): ModelVersion[] {
  return stdout
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => {
      const isActive = line.endsWith("[active]");
      const clean = isActive ? line.replace(/\s*\[active\]$/, "") : line;
      const [versionId, modelPath, runId, parentVersionId, createdAt] = clean.split("\t");
      if (!versionId || !modelPath) return null;
      return {
        versionId,
        modelPath,
        runId: runId === "-" ? null : (runId ?? null),
        parentVersionId: parentVersionId === "-" ? null : (parentVersionId ?? null),
        createdAt: createdAt ?? "",
        isActive,
      };
    })
    .filter((v): v is ModelVersion => v !== null);
}
