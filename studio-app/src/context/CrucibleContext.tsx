import { createContext, useContext, useEffect, useRef, useState, ReactNode, useCallback } from "react";
import {
  listDatasets,
  listModelGroups,
  listModelVersions,
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
import type { ModelGroup, ModelVersion } from "../types/models";
import { loadSessionState, saveSessionState } from "../session_state";

interface CrucibleContextValue {
  dataRoot: string;
  setDataRoot: (value: string) => void;
  datasets: DatasetEntry[];
  selectedDataset: string | null;
  setSelectedDataset: (name: string | null) => void;
  dashboard: DatasetDashboard | null;
  samples: RecordSample[];
  modelGroups: ModelGroup[];
  selectedModelName: string | null;
  setSelectedModelName: (name: string | null) => void;
  modelVersions: ModelVersion[];
  selectedModel: ModelVersion | null;
  setSelectedModel: (version: ModelVersion | null) => void;
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
  const [modelGroups, setModelGroups] = useState<ModelGroup[]>([]);
  const [selectedModelName, setSelectedModelName] = useState<string | null>(
    INITIAL.selected_model_name,
  );
  const [modelVersions, setModelVersions] = useState<ModelVersion[]>([]);
  const [selectedModel, setSelectedModel] = useState<ModelVersion | null>(null);
  const [hardwareProfile, setHardwareProfile] = useState<Record<string, string> | null>(null);

  const refreshModels = useCallback(async () => {
    const groups = await listModelGroups(dataRoot);
    setModelGroups(groups);
    if (groups.length === 0) {
      setSelectedModelName(null);
      setModelVersions([]);
      setSelectedModel(null);
      return;
    }
    // Auto-select model name if none selected or current no longer exists
    setSelectedModelName((current) => {
      if (current && groups.some((g) => g.modelName === current)) {
        return current;
      }
      const saved = INITIAL.selected_model_name;
      const found = saved ? groups.find((g) => g.modelName === saved) : null;
      return found ? found.modelName : groups[0].modelName;
    });
  }, [dataRoot]);

  // Fetch versions when selectedModelName changes
  useEffect(() => {
    if (!selectedModelName) {
      setModelVersions([]);
      setSelectedModel(null);
      return;
    }
    let cancelled = false;
    listModelVersions(dataRoot, selectedModelName)
      .then((rows) => {
        if (cancelled) return;
        setModelVersions(rows);
        setSelectedModel((current) => {
          if (current && rows.some((r) => r.versionId === current.versionId)) {
            return current;
          }
          const savedId = INITIAL.selected_model_version_id;
          const saved = savedId ? rows.find((r) => r.versionId === savedId) : null;
          return saved ?? rows[0] ?? null;
        });
      })
      .catch(console.error);
    return () => { cancelled = true; };
  }, [dataRoot, selectedModelName]);

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
    const poll = setInterval(async () => {
      try {
        const mtime = await getModelIndexMtime(dataRoot);
        if (lastModelMtime.current && mtime !== lastModelMtime.current) {
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
      selected_model_name: selectedModelName,
      selected_model_version_id: selectedModel?.versionId ?? null,
      last_route: window.location.hash,
    });
  }, [dataRoot, selectedDataset, selectedModelName, selectedModel]);

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
        modelGroups,
        selectedModelName,
        setSelectedModelName,
        modelVersions,
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
