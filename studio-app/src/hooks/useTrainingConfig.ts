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
  // Track the method that was loaded so we don't save stale state from a
  // previous method into the new method's config during a switch.
  const loadedMethodRef = useRef<string | null>(null);

  // Load draft from disk on mount or method change
  useEffect(() => {
    let cancelled = false;
    loadedRef.current = false;
    setIsLoaded(false);
    loadTrainingConfig(dataRoot, method)
      .then((file: TrainingConfigFile) => {
        if (cancelled) return;
        if (file.draft) {
          setShared({ ...getDefaultConfigForMethod(method), ...file.draft.shared });
          setExtra(file.draft.extra);
        } else {
          setShared(getDefaultConfigForMethod(method));
          setExtra({});
        }
        loadedRef.current = true;
        loadedMethodRef.current = method;
        setIsLoaded(true);
      })
      .catch(() => {
        if (!cancelled) {
          loadedRef.current = true;
          loadedMethodRef.current = method;
          setIsLoaded(true);
        }
      });
    return () => { cancelled = true; };
  }, [method, dataRoot]);

  // Auto-save draft to disk on every change, debounced 500ms.
  // Skip saves until initial load completes and only save to the method
  // that was actually loaded — prevents stale state from a previous method
  // being written to the new method's config file during a switch.
  useEffect(() => {
    if (!loadedRef.current) return;
    if (loadedMethodRef.current !== method) return;
    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      if (loadedMethodRef.current !== method) return;
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
