import { createContext, useContext, useEffect, useState, ReactNode, useCallback } from "react";
import {
  listDatasets,
  listVersions,
  getDatasetDashboard,
  sampleRecords,
  getHardwareProfile,
} from "../api/studioApi";
import {
  DatasetDashboard,
  RecordSample,
  VersionSummary,
} from "../types";
import { loadSessionState, saveSessionState } from "../session_state";

interface ForgeContextValue {
  dataRoot: string;
  setDataRoot: (value: string) => void;
  datasets: string[];
  selectedDataset: string | null;
  setSelectedDataset: (name: string | null) => void;
  versions: VersionSummary[];
  selectedVersion: string | null;
  setSelectedVersion: (id: string | null) => void;
  dashboard: DatasetDashboard | null;
  samples: RecordSample[];
  hardwareProfile: Record<string, string> | null;
  refreshDatasets: () => Promise<void>;
  refreshHardwareProfile: () => Promise<void>;
}

const ForgeCtx = createContext<ForgeContextValue | null>(null);

export function useForge(): ForgeContextValue {
  const ctx = useContext(ForgeCtx);
  if (!ctx) throw new Error("useForge must be inside ForgeProvider");
  return ctx;
}

const INITIAL = loadSessionState();

export function ForgeProvider({ children }: { children: ReactNode }) {
  const [dataRoot, setDataRoot] = useState(INITIAL.data_root);
  const [datasets, setDatasets] = useState<string[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(
    INITIAL.selected_dataset,
  );
  const [versions, setVersions] = useState<VersionSummary[]>([]);
  const [selectedVersion, setSelectedVersion] = useState<string | null>(
    INITIAL.selected_version,
  );
  const [dashboard, setDashboard] = useState<DatasetDashboard | null>(null);
  const [samples, setSamples] = useState<RecordSample[]>([]);
  const [hardwareProfile, setHardwareProfile] = useState<Record<string, string> | null>(null);

  const refreshDatasets = useCallback(async () => {
    const rows = await listDatasets(dataRoot);
    setDatasets(rows);
    if (rows.length === 0) {
      setSelectedDataset(null);
      setVersions([]);
      setDashboard(null);
      setSamples([]);
      return;
    }
    if (!selectedDataset || !rows.includes(selectedDataset)) {
      setSelectedDataset(rows[0]);
      setSelectedVersion(null);
    }
  }, [dataRoot, selectedDataset]);

  const refreshHardwareProfile = useCallback(async () => {
    const profile = await getHardwareProfile(dataRoot);
    setHardwareProfile(profile);
  }, [dataRoot]);

  useEffect(() => {
    refreshDatasets().catch(console.error);
    refreshHardwareProfile().catch(console.error);
  }, []);

  useEffect(() => {
    if (!selectedDataset) {
      setVersions([]);
      setDashboard(null);
      setSamples([]);
      return;
    }
    (async () => {
      const versionRows = await listVersions(dataRoot, selectedDataset);
      setVersions(versionRows);
      if (versionRows.length === 0) {
        setDashboard(null);
        setSamples([]);
        return;
      }
      const dashboardRow = await getDatasetDashboard(dataRoot, selectedDataset, selectedVersion);
      const sampleRows = await sampleRecords(dataRoot, selectedDataset, selectedVersion, 0, 12);
      setDashboard(dashboardRow);
      setSamples(sampleRows);
    })().catch(console.error);
  }, [dataRoot, selectedDataset, selectedVersion]);

  useEffect(() => {
    saveSessionState({
      data_root: dataRoot,
      selected_dataset: selectedDataset,
      selected_version: selectedVersion,
      last_route: window.location.hash,
    });
  }, [dataRoot, selectedDataset, selectedVersion]);

  return (
    <ForgeCtx.Provider
      value={{
        dataRoot,
        setDataRoot,
        datasets,
        selectedDataset,
        setSelectedDataset,
        versions,
        selectedVersion,
        setSelectedVersion,
        dashboard,
        samples,
        hardwareProfile,
        refreshDatasets,
        refreshHardwareProfile,
      }}
    >
      {children}
    </ForgeCtx.Provider>
  );
}
