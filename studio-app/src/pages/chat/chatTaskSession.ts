const CHAT_TASK_KEY = "crucible_chat_task";
const CHAT_SENDING_KEY = "crucible_chat_sending";

let activeTaskId: string | null = getSessionValue(CHAT_TASK_KEY);

export function getActiveChatTaskId(): string | null {
  return activeTaskId;
}

export function setActiveChatTaskId(id: string | null): void {
  activeTaskId = id;
  if (id) setSessionValue(CHAT_TASK_KEY, id);
  else removeSessionValue(CHAT_TASK_KEY);
}

export function readChatSendingFlag(): boolean {
  return getSessionValue(CHAT_SENDING_KEY) === "1";
}

export function markChatSending(): void {
  setSessionValue(CHAT_SENDING_KEY, "1");
}

export function clearChatSending(): void {
  removeSessionValue(CHAT_SENDING_KEY);
}

function getSessionValue(key: string): string | null {
  if (typeof window === "undefined") return null;
  return window.sessionStorage.getItem(key);
}

function setSessionValue(key: string, value: string): void {
  if (typeof window === "undefined") return;
  window.sessionStorage.setItem(key, value);
}

function removeSessionValue(key: string): void {
  if (typeof window === "undefined") return;
  window.sessionStorage.removeItem(key);
}
