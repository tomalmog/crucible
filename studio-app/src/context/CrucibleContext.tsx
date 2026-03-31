import { createContext, useContext, useEffect, useRef, useState, ReactNode, useCallback } from "react";
import {
  listDatasets,
  listModels,
  getDatasetDashboard,
  sampleRecords,
  getHardwareProfile,
  getModelIndexMtime,
} from "../api/studioApi";
import {
  DatasetDashboard,
  DatasetEntry,
  RecordSample,
} from "../types";
import type { ModelEntry } from "../types/models";
import { loadSessionState, saveSessionState } from "../session_state";

interface CrucibleContextValue {
  dataRoot: string;
  setDataRoot: (value: string) => void;
  datasets: DatasetEntry[];
  selectedDataset: string | null;
  setSelectedDataset: (name: string | null) => void;
  dashboard: DatasetDashboard | null;
  samples: RecordSample[];
  models: ModelEntry[];
  selectedModel: ModelEntry | null;
  setSelectedModel: (entry: ModelEntry | null) => void;
  refreshModels: () => Promise<void>;
  hardwareProfile: Record<string, string> | null;
  refreshDatasets: () => Promise<void>;
  refreshHardwareProfile: () => Promise<void>;
}

const CrucibleCtx = createContext<CrucibleContextValue | null>(null);

export function useCrucible(): CrucibleContextValue {
  const ctx = useContext(CrucibleCtx);
  if (!ctx) throw new Error("useCrucible must be inside CrucibleProvider");
  return ctx;
}

const INITIAL = loadSessionState();

export function CrucibleProvider({ children }: { children: ReactNode }) {
  const [dataRoot, setDataRoot] = useState(INITIAL.data_root);
  const [datasets, setDatasets] = useState<DatasetEntry[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(
    INITIAL.selected_dataset,
  );
  const [dashboard, setDashboard] = useState<DatasetDashboard | null>(null);
  const [samples, setSamples] = useState<RecordSample[]>([]);
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [selectedModel, setSelectedModel] = useState<ModelEntry | null>(null);
  const [hardwareProfile, setHardwareProfile] = useState<Record<string, string> | null>(null);

  const refreshModels = useCallback(async () => {
    const entries = await listModels(dataRoot);
    setModels(entries);
    if (entries.length === 0) {
      setSelectedModel(null);
      return;
    }
    setSelectedModel((current) => {
      if (current && entries.some((e) => e.modelName === current.modelName)) {
        // Update to latest data for the current selection
        return entries.find((e) => e.modelName === current.modelName) ?? null;
      }
      const saved = INITIAL.selected_model_name;
      const found = saved ? entries.find((e) => e.modelName === saved) : null;
      return found ?? entries[0];
    });
  }, [dataRoot]);

  const refreshDatasets = useCallback(async () => {
    const rows = await listDatasets(dataRoot);
    setDatasets(rows);
    if (rows.length === 0) {
      setSelectedDataset(null);
      setDashboard(null);
      setSamples([]);
      return;
    }
    if (!selectedDataset || !rows.some((d) => d.name === selectedDataset)) {
      setSelectedDataset(rows[0].name);
    }
  }, [dataRoot, selectedDataset]);

  const refreshHardwareProfile = useCallback(async () => {
    const profile = await getHardwareProfile(dataRoot);
    setHardwareProfile(profile);
  }, [dataRoot]);

  // Fetch datasets, models, and hardware profile on initial mount
  useEffect(() => {
    refreshDatasets().catch(console.error);
    refreshModels().catch(console.error);
    refreshHardwareProfile().catch(console.error);
  }, []);

  // Auto-refresh models by polling index.json mtime every 5s
  const lastModelMtime = useRef<string>("");
  useEffect(() => {
    // Seed the initial mtime immediately so the first interval poll can detect changes
    getModelIndexMtime(dataRoot)
      .then((mtime) => { lastModelMtime.current = mtime; })
      .catch(() => {});
    const poll = setInterval(async () => {
      try {
        const mtime = await getModelIndexMtime(dataRoot);
        if (mtime !== lastModelMtime.current) {
          refreshModels().catch(console.error);
        }
        lastModelMtime.current = mtime;
      } catch {
        // index.json may not exist yet — ignore
      }
    }, 5_000);
    return () => clearInterval(poll);
  }, [dataRoot, refreshModels]);

  // Reload dashboard and samples when dataset changes
  useEffect(() => {
    if (!selectedDataset) {
      setDashboard(null);
      setSamples([]);
      return;
    }
    let cancelled = false;
    (async () => {
      const dashboardRow = await getDatasetDashboard(dataRoot, selectedDataset);
      const sampleRows = await sampleRecords(dataRoot, selectedDataset, 0, 12);
      if (cancelled) return;
      setDashboard(dashboardRow);
      setSamples(sampleRows);
    })().catch(console.error);
    return () => { cancelled = true; };
  }, [dataRoot, selectedDataset]);

  // Persist session state to localStorage when selection changes
  useEffect(() => {
    saveSessionState({
      data_root: dataRoot,
      selected_dataset: selectedDataset,
      selected_model_name: selectedModel?.modelName ?? null,
      last_route: window.location.hash,
    });
  }, [dataRoot, selectedDataset, selectedModel]);

  return (
    <CrucibleCtx.Provider
      value={{
        dataRoot,
        setDataRoot,
        datasets,
        selectedDataset,
        setSelectedDataset,
        dashboard,
        samples,
        models,
        selectedModel,
        setSelectedModel,
        refreshModels,
        hardwareProfile,
        refreshDatasets,
        refreshHardwareProfile,
      }}
    >
      {children}
    </CrucibleCtx.Provider>
  );
}
