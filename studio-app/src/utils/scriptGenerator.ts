/**
 * Generate and parse training scripts for the Code view tab.
 *
 * Scripts use the crucible_sdk public API so they're readable,
 * editable, and runnable as standalone Python.
 */

import type { TrainingMethod } from "../types/training";
import type { SharedTrainingConfig } from "../types/training";

const CONFIG_BEGIN = "# CRUCIBLE:BEGIN_CONFIG";
const CONFIG_END = "# CRUCIBLE:END_CONFIG";

interface ScriptConfig {
  method: TrainingMethod;
  shared: SharedTrainingConfig;
  extra: Record<string, string>;
  modelName: string;
}

/**
 * Generate a complete Python training script from form values.
 */
export function generateScript(config: ScriptConfig): string {
  const { method, shared, extra, modelName } = config;
  const modelId = extra["--base-model"] || extra["--base-model-path"] || extra["--policy-model-path"] || extra["--teacher-model-path"] || "";
  const dataPath = extra["--dataset"] || extra["--sft-data-path"] || extra["--dpo-data-path"] || extra["--lora-data-path"] || extra["--qlora-data-path"] || extra["--kto-data-path"] || extra["--orpo-data-path"] || extra["--grpo-data-path"] || extra["--rlvr-data-path"] || "";
  const lines: string[] = [CONFIG_BEGIN];
  lines.push(`model_name = "${modelName || "My-Model"}"`);
  lines.push(`model_id = "${modelId}"`);
  lines.push(`data_path = "${dataPath}"`);
  lines.push(`output_dir = ""                                        # leave empty to auto-derive from model_name, or set a custom path`);
  lines.push(`epochs = ${shared.epochs}`);
  lines.push(`learning_rate = ${shared.learningRate}`);
  lines.push(`batch_size = ${shared.batchSize}`);
  lines.push(`max_length = ${shared.maxTokenLength}`);
  lines.push(`precision = "${shared.precision}"`);

  // Method-specific config
  const methodClean = method.replace("-train", "").replace("-", "");
  if (methodClean === "lora" || methodClean === "qlora") {
    lines.push(`lora_rank = ${extra["--lora-rank"] || "8"}`);
    lines.push(`lora_alpha = ${extra["--lora-alpha"] || "16.0"}`);
    lines.push(`lora_dropout = ${extra["--lora-dropout"] || "0.0"}`);
  }
  if (methodClean === "qlora") {
    lines.push(`quantization_bits = ${extra["--quantization-bits"] || "4"}`);
    lines.push(`qlora_type = "${extra["--qlora-type"] || "nf4"}"`);
  }
  if (methodClean === "dpo" || methodClean === "orpo") {
    lines.push(`beta = ${extra["--beta"] || "0.1"}`);
  }

  lines.push(CONFIG_END);
  lines.push("");
  lines.push("import crucible_sdk as crucible");
  lines.push("");

  // Model loading
  if (methodClean === "lora") {
    lines.push("model, tokenizer = crucible.load_model(");
    lines.push('    model_id, method="lora",');
    lines.push("    lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,");
    lines.push(")");
  } else if (methodClean === "qlora") {
    lines.push("model, tokenizer = crucible.load_model(");
    lines.push('    model_id, method="qlora",');
    lines.push("    lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,");
    lines.push("    quantization_bits=quantization_bits, qlora_type=qlora_type,");
    lines.push(")");
  } else {
    lines.push(`model, tokenizer = crucible.load_model(model_id, method="${methodClean}")`);
  }

  // Dataset loading
  const dataFormat = getDataFormat(methodClean);
  lines.push(`dataset = crucible.load_dataset(data_path, format="${dataFormat}")`);
  lines.push("");

  // Training call
  lines.push("result = crucible.train(");
  lines.push("    model=model,");
  lines.push("    tokenizer=tokenizer,");
  lines.push("    dataset=dataset,");
  lines.push(`    method="${methodClean}",`);
  lines.push("    epochs=epochs,");
  lines.push("    learning_rate=learning_rate,");
  lines.push("    batch_size=batch_size,");
  lines.push("    max_length=max_length,");
  lines.push("    precision=precision,");
  lines.push("    output_dir=output_dir,");
  lines.push("    model_name=model_name,");
  if (methodClean === "dpo" || methodClean === "orpo") {
    lines.push("    beta=beta,");
  }
  lines.push(")");
  lines.push("");

  return lines.join("\n");
}

function getDataFormat(method: string): string {
  if (method === "dpo" || method === "orpo") return "dpo";
  if (method === "kto") return "kto";
  if (method === "domainadapt" || method === "distill" || method === "train") return "text";
  if (method === "grpo" || method === "rlvr") return "prompts";
  return "sft";
}

/**
 * Parse config values from a script's CRUCIBLE config section.
 * Returns a dict that can update the form state.
 */
export function parseScriptConfig(script: string): Record<string, string> {
  const config: Record<string, string> = {};
  let inConfig = false;
  for (const line of script.split("\n")) {
    const trimmed = line.trim();
    if (trimmed === CONFIG_BEGIN) { inConfig = true; continue; }
    if (trimmed === CONFIG_END) break;
    if (!inConfig || trimmed.startsWith("#") || !trimmed) continue;
    const match = trimmed.match(/^(\w+)\s*=\s*(.+?)(?:\s*#.*)?$/);
    if (match) {
      const [, name, rawValue] = match;
      // Strip quotes
      let value = rawValue.trim();
      if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
        value = value.slice(1, -1);
      }
      config[name] = value;
    }
  }
  return config;
}

/**
 * Map parsed config values back to form shared config + extra fields.
 */
export function configToFormState(
  parsed: Record<string, string>,
  method: TrainingMethod,
): { shared: Partial<SharedTrainingConfig>; extra: Record<string, string> } {
  const shared: Partial<SharedTrainingConfig> = {};
  const extra: Record<string, string> = {};

  if (parsed.epochs) shared.epochs = parsed.epochs;
  if (parsed.learning_rate) shared.learningRate = parsed.learning_rate;
  if (parsed.batch_size) shared.batchSize = parsed.batch_size;
  if (parsed.max_length) shared.maxTokenLength = parsed.max_length;
  if (parsed.precision) shared.precision = parsed.precision;
  if (parsed.output_dir) shared.outputDir = parsed.output_dir;

  // Model ID → the right extra field
  if (parsed.model_id) {
    if (method === "rlhf-train") {
      extra["--policy-model-path"] = parsed.model_id;
    } else if (method === "distill") {
      extra["--teacher-model-path"] = parsed.model_id;
    } else if (method === "lora-train" || method === "qlora-train" || method === "domain-adapt") {
      extra["--base-model-path"] = parsed.model_id;
    } else {
      extra["--base-model"] = parsed.model_id;
    }
  }

  // Method-specific
  if (parsed.lora_rank) extra["--lora-rank"] = parsed.lora_rank;
  if (parsed.lora_alpha) extra["--lora-alpha"] = parsed.lora_alpha;
  if (parsed.lora_dropout) extra["--lora-dropout"] = parsed.lora_dropout;
  if (parsed.quantization_bits) extra["--quantization-bits"] = parsed.quantization_bits;
  if (parsed.qlora_type) extra["--qlora-type"] = parsed.qlora_type;
  if (parsed.beta) extra["--beta"] = parsed.beta;

  return { shared, extra };
}
