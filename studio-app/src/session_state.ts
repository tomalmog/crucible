const SESSION_STORAGE_KEY = "forge_studio_session_v2";

export interface StudioSessionState {
  data_root: string;
  selected_dataset: string | null;
  selected_version: string | null;
  selected_model_version_id: string | null;
  last_route: string;
}

export const DEFAULT_SESSION_STATE: StudioSessionState = {
  data_root: ".forge",
  selected_dataset: null,
  selected_version: null,
  selected_model_version_id: null,
  last_route: "#/training",
};

export function loadSessionState(): StudioSessionState {
  if (typeof window === "undefined" || !window.localStorage) {
    return DEFAULT_SESSION_STATE;
  }
  const rawValue = window.localStorage.getItem(SESSION_STORAGE_KEY);
  if (!rawValue) {
    return DEFAULT_SESSION_STATE;
  }
  try {
    const parsed = JSON.parse(rawValue) as Partial<StudioSessionState>;
    return {
      data_root: asString(parsed.data_root, DEFAULT_SESSION_STATE.data_root),
      selected_dataset: asNullableString(parsed.selected_dataset),
      selected_version: asNullableString(parsed.selected_version),
      selected_model_version_id: asNullableString(parsed.selected_model_version_id),
      last_route: asString(parsed.last_route, DEFAULT_SESSION_STATE.last_route),
    };
  } catch {
    return DEFAULT_SESSION_STATE;
  }
}

export function saveSessionState(state: StudioSessionState): void {
  if (typeof window === "undefined" || !window.localStorage) {
    return;
  }
  window.localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(state));
}

function asString(value: unknown, fallback: string): string {
  return typeof value === "string" ? value : fallback;
}

function asNullableString(value: unknown): string | null {
  if (value === null) return null;
  return typeof value === "string" ? value : null;
}
