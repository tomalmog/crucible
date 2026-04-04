import { createContext, useContext } from "react";

export interface ScriptRegistration {
  /** Ref to the current script content string */
  contentRef: { current: string };
  /** Function to update the script in the editor */
  setContent: (content: string) => void;
  /** Ref to the current view tab */
  viewTabRef: { current: string };
  /** Training method (stable for the wizard's lifetime) */
  method: string;
}

export interface ScriptContextValue {
  /** Current wizard registration, or null if no wizard is active */
  registration: ScriptRegistration | null;
  /** Called by TrainingWizard on mount */
  register: (reg: ScriptRegistration) => void;
  /** Called by TrainingWizard on unmount */
  unregister: () => void;
}

const noop = () => {};
const defaultValue: ScriptContextValue = {
  registration: null,
  register: noop,
  unregister: noop,
};

export const ScriptContext = createContext<ScriptContextValue>(defaultValue);

export function useScript(): ScriptContextValue {
  return useContext(ScriptContext);
}
