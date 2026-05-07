import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router";
import { BrainCircuit, Loader2, PlayCircle } from "lucide-react";
import { DatasetSelect } from "../../components/shared/DatasetSelect";
import { FormField } from "../../components/shared/FormField";
import { ModelSelect } from "../../components/shared/ModelSelect";
import { useCrucible } from "../../context/CrucibleContext";
import { startCrucibleCommand } from "../../api/studioApi";
import { useInterpLocation } from "../../hooks/useInterpLocation";
import type { BackendKind } from "../../types/jobs";
import {
  DEFAULT_HEALTH_FORM,
  HEALTH_CHECKS,
  HEALTH_SUITES,
  buildHealthSuiteCommands,
  checksForSuite,
  getHealthSuite,
} from "./modelHealthSuites";
import type { HealthCheckId, HealthSuiteFormState, HealthSuiteId } from "./modelHealthSuites";

export function ModelHealthSuiteRunner(): React.ReactNode {
  const navigate = useNavigate();
  const { dataRoot, selectedDataset, selectedModel } = useCrucible();
  const [suiteId, setSuiteId] = useState<HealthSuiteId>("standard");
  const [form, setForm] = useState<HealthSuiteFormState>(DEFAULT_HEALTH_FORM);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const location = useInterpLocation(form.modelPath);
  const suiteLocation = useMemo(() => ({
    clusterBackend: normalizeBackend(location.clusterBackend),
    clusterName: location.clusterName,
    isRemote: location.isRemote,
  }), [location.clusterBackend, location.clusterName, location.isRemote]);
  const suite = getHealthSuite(suiteId);
  const selectedChecks = useMemo(() => checksForSuite(suite, form), [form, suite]);
  const requiresDataset = selectedChecks.some((check) => check.requiresDataset);
  const requiresProbe = selectedChecks.some((check) => check.requiresProbe);
  const requiresContrast = selectedChecks.some((check) => check.requiresContrast);
  const requiresLabel = selectedChecks.some((check) => check.requiresLabel);
  const supportsLayers = selectedChecks.some((check) => check.supportsLayers);

  useEffect(() => {
    const modelPath = selectedModel?.modelPath || selectedModel?.remotePath || "";
    if (!form.modelPath && modelPath) update("modelPath", modelPath);
    if (!form.dataset && selectedDataset) update("dataset", selectedDataset);
  }, [form.dataset, form.modelPath, selectedDataset, selectedModel]);

  const missing = useMemo(() => {
    const fields: string[] = [];
    if (!form.modelPath.trim()) fields.push("model");
    if (suiteId === "targeted" && selectedChecks.length === 0) fields.push("at least one check");
    if (requiresDataset && !form.dataset.trim()) fields.push("dataset");
    if (requiresProbe && !form.probeText.trim()) fields.push("probe text");
    if (requiresContrast && !form.cleanText.trim()) fields.push("clean contrast");
    if (requiresContrast && !form.corruptedText.trim()) fields.push("corrupted contrast");
    if (requiresLabel && !form.labelField.trim()) fields.push("label field");
    if (suiteLocation.isRemote && !suiteLocation.clusterName) fields.push("cluster");
    return fields;
  }, [
    form,
    requiresContrast,
    requiresDataset,
    requiresLabel,
    requiresProbe,
    selectedChecks.length,
    suiteId,
    suiteLocation.clusterName,
    suiteLocation.isRemote,
  ]);

  function update(key: keyof HealthSuiteFormState, value: string): void {
    setForm((current) => ({ ...current, [key]: value }));
  }

  function toggleCheck(checkId: HealthCheckId): void {
    setForm((current) => {
      const selected = current.selectedChecks.includes(checkId)
        ? current.selectedChecks.filter((id) => id !== checkId)
        : [...current.selectedChecks, checkId];
      return { ...current, selectedChecks: selected };
    });
  }

  async function submit(): Promise<void> {
    if (!dataRoot || missing.length > 0) return;
    setSubmitting(true);
    setError(null);
    try {
      const commands = buildHealthSuiteCommands(suite, form, suiteLocation);
      const command = commands[0];
      if (!command) throw new Error("No model health command was generated.");
      await startCrucibleCommand(dataRoot, command.args, command.label, command.config);
      navigate("/runs", { state: { statusFilter: "running" } });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section className="model-health-runner">
      <div className="preflight-header">
        <div>
          <h3>Check health</h3>
          <p>Run one promotion-readiness report, then drill into findings only when needed.</p>
        </div>
        <span className="goal-card-icon"><BrainCircuit size={15} /></span>
      </div>

      <div className="health-runner-block">
        <span className="health-runner-label">Assessment goal</span>
        <div className="health-suite-grid">
          {HEALTH_SUITES.map((item) => (
            <button
              key={item.id}
              className={`health-suite-card${suiteId === item.id ? " active" : ""}`}
              onClick={() => setSuiteId(item.id)}
              type="button"
            >
              <strong>{item.title}</strong>
              <span>{item.summary}</span>
              <span className="health-suite-checks">
                {item.id === "targeted" ? "Choose checks below" : checksForSuite(item, form).map((check) => check.label).join(" + ")}
              </span>
            </button>
          ))}
        </div>

        {suiteId === "targeted" && (
          <div className="health-check-selector">
            {HEALTH_CHECKS.map((check) => (
              <label className="health-check-option" key={check.id}>
                <input
                  checked={form.selectedChecks.includes(check.id)}
                  onChange={() => toggleCheck(check.id)}
                  type="checkbox"
                />
                <span>
                  <strong>{check.label}</strong>
                  <small>{check.category}{check.isExpensive ? " · longer run" : ""}</small>
                  <em>{check.signal}</em>
                </span>
              </label>
            ))}
          </div>
        )}
      </div>

      <div className="grid-2 health-runner-block">
        <span className="health-runner-label health-runner-label-full">Model and calibration</span>
        <FormField label="Model" required>
          <ModelSelect value={form.modelPath} onChange={(value) => update("modelPath", value)} />
        </FormField>
        <FormField label="Calibration Dataset" required={requiresDataset}>
          <DatasetSelect value={form.dataset} onChange={(value) => update("dataset", value)} />
        </FormField>
        <FormField label="Max Samples">
          <input
            type="number"
            min={1}
            value={form.maxSamples}
            onChange={(event) => update("maxSamples", event.currentTarget.value)}
          />
        </FormField>
        {requiresLabel && (
          <FormField label="Label Field" required>
            <input
              value={form.labelField}
              onChange={(event) => update("labelField", event.currentTarget.value)}
              placeholder="label, category, outcome"
            />
          </FormField>
        )}
        {supportsLayers && (
          <FormField label="Layer Indices">
            <input
              value={form.layerIndices}
              onChange={(event) => update("layerIndices", event.currentTarget.value)}
              placeholder="optional: 0,4,8 or 4-12"
            />
          </FormField>
        )}
      </div>

      {suiteLocation.isRemote && (
        <div className="info-banner">
          Remote model selected — health checks will run on cluster <strong>{suiteLocation.clusterName}</strong>
        </div>
      )}

      {(requiresProbe || requiresContrast) && (
        <div className="health-runner-block">
          <span className="health-runner-label">Behavior probe</span>
          {requiresProbe && (
            <FormField label="Probe Text" required>
              <textarea value={form.probeText} onChange={(event) => update("probeText", event.currentTarget.value)} rows={2} />
            </FormField>
          )}
          {requiresContrast && (
            <div className="grid-2">
              <FormField label="Clean Contrast" required>
                <textarea value={form.cleanText} onChange={(event) => update("cleanText", event.currentTarget.value)} rows={3} />
              </FormField>
              <FormField label="Corrupted Contrast" required>
                <textarea value={form.corruptedText} onChange={(event) => update("corruptedText", event.currentTarget.value)} rows={3} />
              </FormField>
            </div>
          )}
        </div>
      )}
      <FormField label="Base Model">
        <input
          value={form.baseModel}
          onChange={(event) => update("baseModel", event.currentTarget.value)}
          placeholder="optional for LoRA/QLoRA adapters"
        />
      </FormField>

      {missing.length > 0 && <div className="error-alert">Missing required fields: {missing.join(", ")}</div>}
      {error && <div className="error-alert">{error}</div>}

      <div className="flex-row">
        <button className="btn btn-primary btn-lg" disabled={missing.length > 0 || submitting} onClick={() => submit().catch(console.error)}>
          {submitting ? <Loader2 size={14} className="spin" /> : <PlayCircle size={14} />}
          {submitting ? "Starting report..." : "Run Health Check"}
        </button>
        <span className="text-muted text-sm">{selectedChecks.length} selected checks, one report</span>
      </div>
    </section>
  );
}

function normalizeBackend(value: string): BackendKind {
  if (value === "ssh" || value === "http-api" || value === "slurm") return value;
  return "slurm";
}
