import { startCrucibleCommand, getCrucibleCommandStatus } from "../api/studioApi";
import { parseAgentResult, parseAgentStdout, summarizeAgentError } from "./agentStream";
import type { AgentTraceEvent } from "../types/agent";

const POLL_MS = 500;

export async function runAgentCommand(
  dataRoot: string,
  payload: Record<string, unknown>,
  onEvent?: (event: AgentTraceEvent) => void,
): Promise<Record<string, unknown>> {
  const { task_id } = await startCrucibleCommand(
    dataRoot,
    ["agent-chat", "--payload-file", "placeholder"],
    "agent-chat",
    payload,
  );
  let seenEvents = 0;
  while (true) {
    const status = await getCrucibleCommandStatus(task_id);
    const parsedStdout = parseAgentStdout(status.stdout || "");
    if (onEvent && parsedStdout.events.length > seenEvents) {
      for (const event of parsedStdout.events.slice(seenEvents)) {
        onEvent(event);
      }
      seenEvents = parsedStdout.events.length;
    }
    if (status.status !== "running") {
      if (status.status === "failed") {
        throw new Error(summarizeAgentError(status.stderr || ""));
      }
      return parseAgentResult(status.stdout || "{}");
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_MS));
  }
}
