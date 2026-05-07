import { useEffect } from "react";
import { getCrucibleCommandStatus, killCrucibleTask } from "../../api/studioApi";
import type { ChatMessage } from "./chatPersistence";
import { extractStatusLine } from "./chatRuntime";
import {
  clearChatSending,
  getActiveChatTaskId,
  setActiveChatTaskId,
} from "./chatTaskSession";

const CHAT_POLL_MS = 100;

interface UseChatTaskResumeOptions {
  isSending: boolean;
  setIsSending: (value: boolean) => void;
  setMessages: (updater: (messages: ChatMessage[]) => ChatMessage[]) => void;
  setChatError: (value: string | null) => void;
  setStatusLine: (value: string) => void;
}

export function useChatTaskResume(options: UseChatTaskResumeOptions): void {
  // Resume an in-flight chat subprocess after route navigation; clear stale state after reloads.
  useEffect(() => {
    const taskId = getActiveChatTaskId();
    if (!taskId) {
      if (options.isSending) {
        options.setIsSending(false);
        clearChatSending();
      }
      options.setMessages(removeTrailingPendingAssistant);
      return;
    }

    let isCancelled = false;
    resumeChatTask(taskId, options, () => isCancelled).catch(() => {
      if (!isCancelled) clearActiveChat(options);
    });
    return () => { isCancelled = true; };
  }, []);
}

async function resumeChatTask(
  taskId: string,
  options: UseChatTaskResumeOptions,
  isCancelled: () => boolean,
): Promise<void> {
  const first = await getCrucibleCommandStatus(taskId);
  if (first.status !== "running") {
    handleFinalStatus(taskId, first, options);
    return;
  }

  while (!isCancelled()) {
    await new Promise((resolve) => setTimeout(resolve, CHAT_POLL_MS));
    const status = await getCrucibleCommandStatus(taskId);
    if (status.stdout.trim()) updateAssistantMessage(options, status.stdout.trim());
    options.setStatusLine(extractStatusLine(status.stderr));
    if (status.status !== "running") {
      handleFinalStatus(taskId, status, options);
      return;
    }
  }
}

function handleFinalStatus(
  taskId: string,
  status: Awaited<ReturnType<typeof getCrucibleCommandStatus>>,
  options: UseChatTaskResumeOptions,
): void {
  if (status.status === "completed" && status.exit_code === 0) {
    updateAssistantMessage(options, status.stdout.trim() || "(no response generated)");
  } else if (status.stderr) {
    options.setChatError(status.stderr);
    killCrucibleTask(taskId).catch(() => {});
  }
  clearActiveChat(options);
}

function updateAssistantMessage(options: UseChatTaskResumeOptions, content: string): void {
  options.setMessages((current) => {
    const updated = [...current];
    updated[updated.length - 1] = { role: "assistant", content };
    return updated;
  });
}

function clearActiveChat(options: UseChatTaskResumeOptions): void {
  setActiveChatTaskId(null);
  options.setIsSending(false);
  options.setStatusLine("");
  clearChatSending();
}

function removeTrailingPendingAssistant(messages: ChatMessage[]): ChatMessage[] {
  const lastMessage = messages[messages.length - 1];
  if (lastMessage?.role !== "assistant" || lastMessage.content.trim()) {
    return messages;
  }
  return messages.slice(0, -1);
}
