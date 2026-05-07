import { buildDispatchSpec } from "../../api/commandArgs";
import type { BackendKind } from "../../types/jobs";
import { modelBasename } from "../../utils/jobLabels";

export type HealthSuiteId = "standard" | "deep" | "supervised" | "targeted";
export type HealthCheckId =
  | "weight-norms"
  | "activation-norms"
  | "gradient-norms"
  | "logit-lens"
  | "activation-pca"
  | "activation-patching"
  | "linear-probe"
  | "linear-probe-layers";

export interface HealthCheckDefinition {
  category: string;
  id: HealthCheckId;
  isExpensive: boolean;
  label: string;
  requiresContrast: boolean;
  requiresDataset: boolean;
  requiresLabel: boolean;
  requiresProbe: boolean;
  signal: string;
  supportsLayers: boolean;
}

export interface HealthSuiteDefinition {
  checks: HealthCheckId[];
  id: HealthSuiteId;
  summary: string;
  title: string;
}

export interface HealthSuiteFormState {
  baseModel: string;
  cleanText: string;
  corruptedText: string;
  dataset: string;
  labelField: string;
  layerIndices: string;
  maxSamples: string;
  modelPath: string;
  probeText: string;
  selectedChecks: HealthCheckId[];
}

export interface HealthSuiteLocation {
  clusterBackend: BackendKind;
  clusterName: string;
  isRemote: boolean;
}

export interface HealthSuiteCommand {
  args: string[];
  config: Record<string, unknown>;
  label: string;
}

export const DEFAULT_HEALTH_FORM: HealthSuiteFormState = {
  baseModel: "",
  cleanText: "The support agent should refund the customer because the item arrived broken.",
  corruptedText: "The support agent should refund the customer because the item worked as expected.",
  dataset: "",
  labelField: "",
  layerIndices: "",
  maxSamples: "300",
  modelPath: "",
  probeText: "The customer asked for a refund because",
  selectedChecks: ["weight-norms", "activation-norms"],
};

export const HEALTH_CHECKS: HealthCheckDefinition[] = [
  check("weight-norms", "Static Model", "Weight stability", "Scans layers for weight norm spikes or invalid values.", false, false, false, false, true, false),
  check("activation-norms", "Activations", "Activation stability", "Checks layer activation norms on calibration prompts.", true, false, false, false, true, false),
  check("gradient-norms", "Training Stability", "Gradient stability", "Backprops a calibration objective to find exploding layer gradients.", true, false, false, false, true, true),
  check("logit-lens", "Behavior", "Prediction trace", "Shows how next-token predictions change across model layers.", false, false, true, false, true, false),
  check("activation-pca", "Representations", "Representation map", "Projects activations for calibration samples to catch drift or shortcut clusters.", true, false, false, false, true, false),
  check("activation-patching", "Causal", "Causal contrast", "Tests which activations drive a behavior difference between two prompts.", false, false, false, true, true, false),
  check("linear-probe", "Supervised", "Label separability", "Checks whether one selected layer encodes a supervised label.", true, true, false, false, true, false),
  check("linear-probe-layers", "Supervised", "Layer-wise label probe", "Runs linear probes across layers to locate where a label becomes separable.", true, true, false, false, true, true),
];

export const HEALTH_SUITES: HealthSuiteDefinition[] = [
  {
    checks: ["weight-norms", "activation-norms", "activation-pca", "activation-patching"],
    id: "standard",
    summary: "Balanced release-readiness checks for static weights, activations, representations, and one behavior contrast.",
    title: "Standard Health Check",
  },
  {
    checks: ["weight-norms", "activation-norms", "gradient-norms", "activation-pca", "activation-patching", "logit-lens"],
    id: "deep",
    summary: "Adds gradient sensitivity and prediction tracing for higher-risk candidate models.",
    title: "Deep Stability Assessment",
  },
  {
    checks: ["weight-norms", "activation-norms", "linear-probe-layers", "activation-pca"],
    id: "supervised",
    summary: "Uses a labeled calibration dataset to locate where product labels become separable.",
    title: "Supervised Behavior Check",
  },
  {
    checks: [],
    id: "targeted",
    summary: "Select specific checks and optional layers for an agent- or user-directed investigation.",
    title: "Targeted Investigation",
  },
];

export function getHealthSuite(id: HealthSuiteId): HealthSuiteDefinition {
  return HEALTH_SUITES.find((suite) => suite.id === id) ?? HEALTH_SUITES[0];
}

export function checksForSuite(suite: HealthSuiteDefinition, form: HealthSuiteFormState): HealthCheckDefinition[] {
  const ids = suite.id === "targeted" ? form.selectedChecks : suite.checks;
  return ids.map((id) => HEALTH_CHECKS.find((checkItem) => checkItem.id === id)).filter((item): item is HealthCheckDefinition => item !== undefined);
}

export function buildHealthSuiteCommands(
  suite: HealthSuiteDefinition,
  form: HealthSuiteFormState,
  location: HealthSuiteLocation,
): HealthSuiteCommand[] {
  return [buildHealthSuiteCommand({ form, location, suite })];
}

function buildHealthSuiteCommand({
  form,
  location,
  suite,
}: {
  form: HealthSuiteFormState;
  location: HealthSuiteLocation;
  suite: HealthSuiteDefinition;
}): HealthSuiteCommand {
  const selectedChecks = checksForSuite(suite, form);
  const config = buildHealthSuiteConfig({ form, selectedChecks, suite });
  const label = `Model Health · ${suite.title} · ${modelBasename(form.modelPath)}`;
  if (location.isRemote) {
    const args = buildDispatchSpec(
      "model-health-check",
      buildRemoteMethodArgs({ form, selectedChecks, suite }),
      location.clusterBackend,
      { label, clusterName: location.clusterName, config },
    );
    return { args, config, label };
  }
  return { args: buildLocalArgs({ form, selectedChecks, suite }), config, label };
}

function buildHealthSuiteConfig({
  form,
  selectedChecks,
  suite,
}: {
  form: HealthSuiteFormState;
  selectedChecks: HealthCheckDefinition[];
  suite: HealthSuiteDefinition;
}): Record<string, unknown> {
  return {
    page: "interpretability",
    tab: "health",
    workflow: "model-health-check",
    suiteId: suite.id,
    suiteTitle: suite.title,
    healthChecks: selectedChecks.map((checkItem) => ({ id: checkItem.id, label: checkItem.label, why: checkItem.signal })),
    modelPath: form.modelPath,
    dataset: form.dataset,
    probeText: form.probeText,
    cleanText: form.cleanText,
    corruptedText: form.corruptedText,
    labelField: form.labelField,
    maxSamples: form.maxSamples,
    baseModel: form.baseModel,
    checks: selectedChecks.map((checkItem) => checkItem.id),
    layerIndices: form.layerIndices,
  };
}

function buildRemoteMethodArgs({
  form,
  selectedChecks,
  suite,
}: {
  form: HealthSuiteFormState;
  selectedChecks: HealthCheckDefinition[];
  suite: HealthSuiteDefinition;
}): Record<string, unknown> {
  const shared: Record<string, unknown> = {
    model_path: form.modelPath,
    dataset_name: form.dataset,
    suite: suite.id,
    checks: selectedChecks.map((checkItem) => checkItem.id).join(","),
    layer_indices: form.layerIndices,
    probe_text: form.probeText,
    clean_text: form.cleanText,
    corrupted_text: form.corruptedText,
    label_field: form.labelField,
    max_samples: parseMaxSamples(form.maxSamples),
    output_dir: "./outputs/model-health",
  };
  if (form.baseModel.trim()) shared.base_model = form.baseModel;
  return shared;
}

function buildLocalArgs({
  form,
  selectedChecks,
  suite,
}: {
  form: HealthSuiteFormState;
  selectedChecks: HealthCheckDefinition[];
  suite: HealthSuiteDefinition;
}): string[] {
  const args = ["model-health-check", "--model-path", form.modelPath, "--suite", suite.id, "--max-samples", String(parseMaxSamples(form.maxSamples)), "--output-dir", "./outputs/model-health"];
  if (form.dataset.trim()) args.push("--dataset", form.dataset);
  if (form.probeText.trim()) args.push("--probe-text", form.probeText);
  if (form.cleanText.trim()) args.push("--clean-text", form.cleanText);
  if (form.corruptedText.trim()) args.push("--corrupted-text", form.corruptedText);
  if (form.baseModel.trim()) args.push("--base-model", form.baseModel);
  if (form.labelField.trim()) args.push("--label-field", form.labelField);
  if (selectedChecks.length > 0) args.push("--checks", selectedChecks.map((checkItem) => checkItem.id).join(","));
  if (form.layerIndices.trim()) args.push("--layer-indices", form.layerIndices);
  return args;
}

function check(
  id: HealthCheckId,
  category: string,
  label: string,
  signal: string,
  requiresDataset: boolean,
  requiresLabel: boolean,
  requiresProbe: boolean,
  requiresContrast: boolean,
  supportsLayers: boolean,
  isExpensive: boolean,
): HealthCheckDefinition {
  return { category, id, isExpensive, label, requiresContrast, requiresDataset, requiresLabel, requiresProbe, signal, supportsLayers };
}

function parseMaxSamples(value: string): number {
  const parsed = Number.parseInt(value || "300", 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 300;
}
