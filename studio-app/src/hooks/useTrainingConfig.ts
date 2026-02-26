import { useCallback, useEffect, useRef, useState } from "react";
import {
  TrainingMethod,
  SharedTrainingConfig,
  TrainingConfigFile,
  getDefaultConfigForMethod,
} from "../types/training";
import { loadTrainingConfig, saveTrainingConfig } from "../api/studioApi";

export interface TrainingConfigState {
  shared: SharedTrainingConfig;
  setShared: (s: SharedTrainingConfig) => void;
  extra: Record<string, string>;
  setExtra: (e: Record<string, string>) => void;
  isLoaded: boolean;
  resetToDefaults: () => void;
}

export function useTrainingConfig(
  method: TrainingMethod,
  dataRoot: string,
): TrainingConfigState {
  const [shared, setShared] = useState(getDefaultConfigForMethod(method));
  const [extra, setExtra] = useState<Record<string, string>>({});
  const [isLoaded, setIsLoaded] = useState(false);
  const loadedRef = useRef(false);
  const timerRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  // Load draft from disk on mount
  useEffect(() => {
    let cancelled = false;
    loadTrainingConfig(dataRoot, method)
      .then((file: TrainingConfigFile) => {
        if (cancelled) return;
        if (file.draft) {
          setShared(file.draft.shared);
          setExtra(file.draft.extra);
        }
        loadedRef.current = true;
        setIsLoaded(true);
      })
      .catch(() => {
        if (!cancelled) {
          loadedRef.current = true;
          setIsLoaded(true);
        }
      });
    return () => { cancelled = true; };
  }, [method, dataRoot]);

  // Auto-save draft to disk on every change, debounced 500ms.
  // Skip saves until initial load completes to avoid overwriting with defaults.
  useEffect(() => {
    if (!loadedRef.current) return;
    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      saveTrainingConfig(dataRoot, method, {
        draft: { shared, extra },
      }).catch(console.error);
    }, 500);
    return () => clearTimeout(timerRef.current);
  }, [shared, extra, dataRoot, method]);

  const resetToDefaults = useCallback(() => {
    setShared(getDefaultConfigForMethod(method));
    setExtra({});
  }, [method]);

  return { shared, setShared, extra, setExtra, isLoaded, resetToDefaults };
}
