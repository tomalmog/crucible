"""Studio AI agent with Crucible MCP tools.

Agentic loop: call Claude, execute tools, repeat. Conversation persisted to disk.
"""
from __future__ import annotations

import inspect
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

# ── Tool introspection ──────────────────────────────────────────────

_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

_TOOL_NAMES = [
    "list_datasets", "ingest_dataset", "delete_dataset", "push_dataset",
    "list_remote_datasets", "list_models", "register_model", "register_remote_model",
    "delete_model", "pull_model", "train", "submit_remote_training",
    "submit_remote_interp", "submit_remote_sweep",
    "list_jobs", "job_status", "job_logs", "job_result", "cancel_job", "delete_job",
    "run_benchmark", "submit_remote_eval", "chat", "run_interp",
    "export_model", "merge_models", "register_cluster", "validate_cluster",
    "remove_cluster", "list_clusters", "cluster_info", "hub_search_models",
    "hub_download_model", "hub_search_datasets", "hub_download_dataset", "run_sweep",
    "lora_merge", "curate_dataset", "generate_synthetic_data",
    "hardware_profile",
]


def _parse_arg_descriptions(docstring: str | None) -> dict[str, str]:
    """Extract per-parameter descriptions from an Args: docstring section."""
    if not docstring:
        return {}
    descs: dict[str, str] = {}
    in_args = False
    for line in docstring.split("\n"):
        stripped = line.strip()
        if stripped.startswith("Args:"):
            in_args = True
            continue
        if in_args:
            if stripped and not stripped[0].isspace() and ":" not in stripped:
                break
            parts = stripped.split(":", 1)
            if len(parts) == 2 and parts[0].strip().isidentifier():
                descs[parts[0].strip()] = parts[1].strip()
    return descs


def _build_tool_definitions() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Introspect MCP server functions to build Claude API tool schemas."""
    from serve import mcp_server

    tools: list[dict[str, Any]] = []
    registry: dict[str, Any] = {}

    for name in _TOOL_NAMES:
        fn = getattr(mcp_server, name, None)
        if fn is None:
            continue
        sig = inspect.signature(fn)
        doc = inspect.getdoc(fn) or ""
        arg_descs = _parse_arg_descriptions(doc)
        description = doc.split("\n\n")[0].split("\nArgs:")[0].strip()

        properties: dict[str, Any] = {}
        required: list[str] = []
        for pname, param in sig.parameters.items():
            ptype = _TYPE_MAP.get(param.annotation, "string")
            prop: dict[str, Any] = {"type": ptype}
            if pname in arg_descs:
                prop["description"] = arg_descs[pname]
            properties[pname] = prop
            if param.default is inspect.Parameter.empty:
                required.append(pname)

        tools.append({
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        })
        registry[name] = fn

    return tools, registry


_TOOLS: list[dict[str, Any]] | None = None
_REGISTRY: dict[str, Any] | None = None


def _get_tools() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    global _TOOLS, _REGISTRY
    if _TOOLS is None:
        _TOOLS, _REGISTRY = _build_tool_definitions()
    return _TOOLS, _REGISTRY


def execute_tool(name: str, tool_input: dict[str, Any]) -> str:
    """Execute a Crucible tool by name and return the result string."""
    import io
    import sys

    _, registry = _get_tools()
    fn = registry.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        # Redirect stdout so tool progress output (e.g. training events)
        # doesn't corrupt the agent-chat JSON response on stdout.
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            result = str(fn(**tool_input))
        finally:
            sys.stdout = old_stdout
        return result
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


# ── System prompt ───────────────────────────────────────────────────

_AGENT_BEHAVIOR = """
You are the Crucible Studio AI assistant. You help users train, evaluate,
and manage ML models through the Crucible platform.

CRITICAL: Do NOT use tools unless the user explicitly asks for data or
an action. If the user says "hello", "hey", or any greeting, just
respond conversationally. NEVER call a tool on a greeting.

Rules:
- Only use tools when the user asks a question that requires real data
  (e.g. "what models do I have?") or asks you to perform an action
  (e.g. "train a model", "run an eval").
- When using tools, check state first (list_models, list_datasets, list_jobs)
  before making changes.
- For multi-step requests, briefly state the plan, then execute it.
- Prefer action over back-and-forth. Use defaults and keep going unless a
  missing input would make the workflow unsafe or impossible.
- Explain what you're doing and why.
- When writing files, use the agent workspace directory by default.
- Return results in a clear, readable format.
- Be concise. Don't repeat tool output verbatim — summarize it.
- Never fabricate information. If you don't know something, say so.
- Never pretend to execute actions you didn't actually perform via tools.

## Using Default Values

When the user doesn't specify a value, use sensible defaults instead of asking:
- `output_dir`: default `"./outputs/<method>-<model>"` (e.g. "./outputs/lora-gpt2")
- `model_name`: derive from the output dir or method name
- `epochs`: 3
- `batch_size`: 16
- `learning_rate`: use the method default (1e-3 for basic, 2e-5 for SFT/DPO/KTO/ORPO, 2e-4 for LoRA/QLoRA)
- `max_token_length`: 512
Only ask the user for values when they are truly required and have no default
(e.g. which dataset to use, which model to fine-tune).

## Model Name Resolution

The "Available models" list in Studio State shows model **names**, but training
tools require model **paths**. The names alone are NOT valid for fields like
`base_model_path`, `policy_model_path`, etc.
- Always call `list_models()` to resolve a model name to its actual path before
  passing it to any training, eval, or interp tool.
- HuggingFace model IDs (e.g. "gpt2", "meta-llama/Llama-2-7b") can be used
  directly — they don't need resolution.

## Autonomous Workflow Rules

When the user gives you an end goal, try to complete the workflow instead of
stopping after one tool call.

- If they provide remote cluster credentials, register the cluster, validate
  it, then continue with dataset push / job submission.
- If they ask for a model "on HuggingFace", use `hub_search_models` to find a
  good candidate before training. Use `hub_download_model` only when a local
  download is actually useful; remote training can use HuggingFace model IDs
  directly.
- If they ask for data and do not name a local dataset, search HuggingFace
  datasets with `hub_search_datasets`, then download one with
  `hub_download_dataset` or explain why no suitable dataset was found.
- For remote training requests, prefer the remote workflow once a cluster is
  available: verify dataset presence, push the dataset if needed, submit the
  job, navigate to `/jobs`, and create a pending chain for follow-up eval or
  interpretability work when appropriate.
- When submitting remote GPU work, prefer a concrete `gpu_type` from
  `list_clusters()` when the cluster advertises available GPU types instead of
  leaving the request generic.
- If the user asks for a better model in a domain like medical conversations,
  choose a plausible base model, dataset, and training method instead of
  asking the user to assemble the pipeline manually.

## Mechanistic Interpretability Workflows

Use only the available interpretability entry points:
`run_interp`, `submit_remote_interp`, and the methods supported by those tools:
`logit-lens`, `activation-pca`, `activation-patch`, `linear-probe`,
`sae-train`, `sae-analyze`, `steer-compute`, and `steer-apply`.
Do not claim access to attribution, circuit discovery, causal tracing, neuron
labeling, dashboards, visualizers, or automated report generators unless a tool
result explicitly provides them.

Before running tools:
- Resolve registered model names with `list_models()` unless the user gave a
  HuggingFace model ID or explicit filesystem path.
- For remote interp, use a remote-accessible model path from `list_models()`.
  Do not pass a bare Studio model name or assume a local-only path exists on the
  cluster.
- Check `list_datasets()` before local dataset-based tools. For remote
  dataset-based tools, choose a cluster with `list_clusters()`, then verify the
  dataset with `list_remote_datasets()` or push it with `push_dataset()`.

Intent-to-tool map with prerequisites:
- "what is it predicting" / "why this token": `logit-lens`; requires
  `input_text`, optional `top_k` and `layer_indices`.
- "are examples clustered" / "dataset geometry": `activation-pca`; requires
  `dataset_name`, optional `color_field`, `granularity`, `layer_index`.
- "where is this label represented": `linear-probe`; requires `dataset_name`
  and `label_field` with at least two label values.
- "what layer causes this behavior": `activation-patch`; requires
  `clean_text`, `corrupted_text`, `target_token_index`, and `metric`.
- "find sparse features": `sae-train`; requires `dataset_name`; returns
  `sae_path` for `sae-analyze`.
- "inspect sparse features on this prompt": `sae-analyze`; requires the exact
  `sae_path` from a prior `sae-train` result and `input_text`; optional
  `dataset_name` adds returned feature associations.
- "change/suppress/increase behavior": `steer-compute`; requires
  `positive_text`/`negative_text` or `positive_dataset`/`negative_dataset`;
  returns `steering_vector_path` for `steer-apply`.
- "apply a steering vector": `steer-apply`; requires the exact
  `steering_vector_path` from `steer-compute`, `input_text`, coefficient, and
  max token count.

Chaining rules:
- For local dependent steps, call one tool, parse its JSON result, then call the
  follow-up with the exact returned path. Never invent `sae_path` or
  `steering_vector_path` from an output directory.
- For remote dependent steps, submit only the first remote job. Put the
  dependent steps in `<pending_chain>` and say they must call `job_result()` for
  the completed job, extract the returned artifact path, then run the follow-up.
- Good remote pending-chain steps are explicit, e.g. "Call `job_result()` for
  the completed SAE training job, extract `sae_path`, then run `sae-analyze`
  on model path X with prompt Y" or "extract `steering_vector_path`, then run
  `steer-apply` with prompt Y and coefficient Z."
- On a chain continuation message, first call `job_result(<job_id>)` when the
  next step needs `sae_path`, `steering_vector_path`, or any remote result
  field. If the result is failed or the artifact key is missing, stop and report
  the gap instead of guessing.

Concrete recipes:
- Prompt triage: `logit-lens` on 2-3 prompts; compare returned
  `layers[].predictions`.
- Representation separation: `activation-pca`; if a label field exists, follow
  with `linear-probe`.
- Minimal causal contrast: `activation-patch`; rank layers by returned
  recovery, but describe them as candidates for that contrast only.
- Feature discovery: `sae-train` -> parse `sae_path` -> `sae-analyze` on a
  specific prompt.
- Steering intervention: `steer-compute` -> parse `steering_vector_path` ->
  `steer-apply` and compare `original_text` with `steered_text`.

Report guardrails:
- Logit lens shows layer-wise vocabulary projections, not a causal explanation.
- PCA shows a projection of activation geometry, not proof of discrete clusters.
- Probe accuracy is decodability evidence, not proof the model uses the feature.
- Patching recovery identifies candidate causal layers for one contrast, not a
  full circuit.
- SAE feature concepts are only as strong as returned activations and
  associated texts; do not invent labels when the result does not include them.
- Steering output is an intervention result, not evidence of safe or reliable
  control beyond the tested prompt and coefficient.

Navigate to `/interpretability` when setting up an interp run. Navigate to
`/jobs` after submitting local or remote interp work so the user can inspect
the artifact.

## Training Script Interaction
When the user has the Code tab open in the training wizard, you can see and
edit their training script (shown in "Current Training Script" below).
- To EXPLAIN the script: analyze it and describe what each section does.
- To MODIFY the script: return the COMPLETE updated script in <script_update> tags.
  Always return the full script, not just changed lines.
  Preserve the CRUCIBLE config markers (BEGIN_CONFIG / END_CONFIG).
- Only modify the script when the user asks. Do not modify it unprompted.
- If the user asks you to edit the script but the Code tab is NOT open
  (no "Current Training Script" section below), tell them to switch to the
  Code tab in the training wizard so you can see and edit the script.

## Job Chaining

When the user asks you to perform multiple steps that involve remote jobs (e.g.
"train model A, then LoRA fine-tune it, then benchmark it"), submit only the
FIRST job. Then declare the remaining steps in a <pending_chain> tag so they
run automatically after the job completes.

Usage: include <pending_chain>...</pending_chain> in your response AFTER
submitting a remote job. List each remaining step on its own line.
- Only use this when you've submitted a remote job AND have clearly defined
  follow-up steps that depend on the job's output.
- Each step should be a complete, self-contained instruction that you can
  execute when re-activated (include model names, dataset names, parameters).
- Only include one <pending_chain> tag per response.
- Do NOT include the step you just submitted — only REMAINING steps.

Example:
<pending_chain>
LoRA fine-tune the output model on dataset 'instruct-v2' with rank=16, learning_rate=2e-4, call it "my-lora"
Run MMLU and HellaSwag benchmarks on the LoRA-tuned model
</pending_chain>

## Page Navigation
You can navigate the user to any Studio page by including a <navigate_to> tag in your
response. The user will see the page change and a badge in the chat confirming it.

Valid routes:
/build (Build),
/dashboard (Dashboard), /training (Training), /benchmarks (Eval),
/interpretability (Interpretability), /datasets (Datasets), /models (Models),
/eval-tasks (Benchmarks), /chat (Chat),
/hub (Hub), /export (Export), /jobs (Jobs), /clusters (Clusters),
/resources (Resources), /docs (Docs), /settings (Settings)

Usage: include <navigate_to>/route</navigate_to> anywhere in your response.
- Navigate when the user asks to go somewhere ("show me my models", "go to jobs").
- Navigate after completing actions where the result lives on another page
  (e.g. after submitting a training job, navigate to /jobs).
- Always include useful text alongside the navigation — don't just navigate silently.
- Only use exact routes from the list above. Never invent routes.
- Only include one <navigate_to> tag per response.

## Build Workspace Directives
On the main /build page, you can shape the workspace with control tags.
- Use `<workspace_mode>auto</workspace_mode>`, `focus`, `compare`, or `board`
  to suggest the layout.
- Use `<workspace_cards>` with one card ID per line to choose which cards to
  highlight or show first.
- Preferred card IDs: `latest`, `latest_training`, `latest_eval`,
  `previous_eval`, `latest_interp`, `live_trace`, `pending_chain`, `context`.
- Short aliases also work: `artifact` = latest, `trace` = live_trace.
- These tags are stripped from the visible chat text.
"""

_VALID_ROUTES: dict[str, str] = {
    "/build": "Build",
    "/dashboard": "Dashboard",
    "/training": "Training",
    "/benchmarks": "Eval",
    "/interpretability": "Interpretability",
    "/datasets": "Datasets",
    "/models": "Models",
    "/eval-tasks": "Benchmarks",
    "/chat": "Chat",
    "/hub": "Hub",
    "/export": "Export",
    "/jobs": "Jobs",
    "/clusters": "Clusters",
    "/resources": "Resources",
    "/docs": "Docs",
    "/settings": "Settings",
}

_CONTROL_TAGS = (
    "script_update",
    "navigate_to",
    "pending_chain",
    "workspace_mode",
    "workspace_cards",
)


def _control_tag_pattern(tag: str) -> re.Pattern[str]:
    return re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)


def _normalize_assistant_text(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _pop_control_tag(text: str, tag: str) -> tuple[str | None, str]:
    pattern = _control_tag_pattern(tag)
    match = pattern.search(text)
    value = match.group(1).strip() if match else None
    return value, _normalize_assistant_text(pattern.sub("", text))


def _strip_control_tags(text: str) -> str:
    cleaned = text
    for tag in _CONTROL_TAGS:
        cleaned = _control_tag_pattern(tag).sub("", cleaned)
    return _normalize_assistant_text(cleaned)


def _parse_workspace_mode(raw_mode: str | None) -> str | None:
    if raw_mode is None:
        return None
    mode = " ".join(raw_mode.split())
    return mode or None


def _parse_workspace_cards(raw_cards: str | None) -> list[str] | None:
    if raw_cards is None:
        return None
    cards: list[str] = []
    for piece in re.split(r"[\n,]", raw_cards):
        card = re.sub(r"^[*-]\s*", "", piece.strip())
        if not card or card in cards:
            continue
        cards.append(card)
    return cards or None


def _build_system_prompt(app_context: dict[str, Any], data_root: str) -> str:
    from serve.mcp_server import mcp
    workspace = str(Path(data_root) / "agent" / "workspace")
    os.makedirs(workspace, exist_ok=True)

    context_lines = [
        f"- Data root: {data_root}",
        f"- Agent workspace: {workspace}",
    ]
    if app_context.get("currentPage"):
        context_lines.append(f"- Current page: {app_context['currentPage']}")
    if app_context.get("selectedModel"):
        context_lines.append(f"- Selected model: {app_context['selectedModel']}")
    if app_context.get("selectedDataset"):
        context_lines.append(f"- Selected dataset: {app_context['selectedDataset']}")
    if app_context.get("modelNames"):
        context_lines.append(f"- Available models: {', '.join(app_context['modelNames'])}")
    if app_context.get("modelPaths"):
        paths = app_context["modelPaths"]
        path_lines = [f"  {name}: {path}" for name, path in paths.items()]
        context_lines.append(
            "- Model paths (use these in tool calls, not names):\n"
            + "\n".join(path_lines),
        )
    if app_context.get("datasetNames"):
        context_lines.append(f"- Available datasets: {', '.join(app_context['datasetNames'])}")

    script_section = ""
    if app_context.get("script"):
        script_ctx = app_context["script"]
        method = script_ctx.get("trainingMethod", "unknown")
        script = script_ctx.get("trainingScript", "")
        script_section = (
            "\n## Current Training Script\n\n"
            f"The user is viewing a {method} training script in the code editor.\n"
            "You can read, explain, and modify this script.\n"
            "To modify it, return the COMPLETE updated script wrapped in "
            "<script_update> tags. The updated script will replace the current "
            "editor content.\n\n"
            f"```python\n{script}\n```\n"
        )

    return (
        (mcp.instructions or "") + "\n\n"
        "## Current Studio State\n\n"
        + "\n".join(context_lines) + "\n\n"
        + _AGENT_BEHAVIOR
        + script_section
    )


# ── Conversation persistence ────────────────────────────────────────

def _load_conversation_raw(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data.get("messages", [])
    except (json.JSONDecodeError, KeyError):
        return []


def _load_conversation(path: Path) -> list[dict[str, Any]]:
    return [_model_message(m) for m in _load_conversation_raw(path)]


def _model_message(message: dict[str, Any]) -> dict[str, Any]:
    """Return only provider-supported message fields."""
    return {
        "role": message.get("role", ""),
        "content": message.get("content", ""),
    }


def _save_conversation(path: Path, messages: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"messages": messages}, indent=2, default=str))


def _trim_conversation(
    messages: list[dict[str, Any]],
    max_chars: int = 400_000,
) -> list[dict[str, Any]]:
    """Trim oldest turns if conversation is too long (~100k tokens)."""
    total = len(json.dumps(messages))
    if total <= max_chars:
        return messages
    while len(messages) > 2 and len(json.dumps(messages)) > max_chars:
        messages.pop(0)
    return messages


# ── Agentic loop ────────────────────────────────────────────────────

_MAX_TOOL_LOOPS = 15
_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
_DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
_DEFAULT_OLLAMA_MODEL = "llama3.1"
_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

_MEDICAL_DEMO_EXPLICIT_TERMS = (
    "medical assistant",
    "medical safety triage",
    "medical safety triage training",
)
_MEDICAL_DEMO_MODEL_TERMS = (
    "medical assistant",
)
_MEDICAL_DEMO_FAILURE_TERMS = (
    "failing",
    "failed",
    "fails",
    "underperforming",
    "not doing well",
    "poor",
    "bad",
)
_MEDICAL_DEMO_IMPROVEMENT_TERMS = (
    "improve",
    "fix",
    "fine tune",
    "finetune",
    "train",
    "make better",
)
_MEDICAL_DEMO_EVAL_TERMS = (
    "benchmark",
    "benchmarks",
    "eval",
    "evals",
    "evaluation",
    "relevant",
    "safety",
    "triage",
)
_MEDICAL_BENCHMARK = "medical_safety_triage"
_MEDICAL_DATASET = "medical_safety_triage_training"
_MEDICAL_TUNED_MODEL_PREFIX = "medical-assistant-safety-tuned"


def is_medical_safety_demo_request(user_message: str) -> bool:
    """Return true when the request matches the medical improvement workflow."""
    normalized = _normalize_request_text(user_message)
    if _contains_all_phrases(normalized, _MEDICAL_DEMO_EXPLICIT_TERMS):
        return True
    return (
        _contains_any_phrase(normalized, _MEDICAL_DEMO_MODEL_TERMS)
        and _contains_any_phrase(normalized, _MEDICAL_DEMO_FAILURE_TERMS)
        and _contains_any_phrase(normalized, _MEDICAL_DEMO_IMPROVEMENT_TERMS)
        and _contains_any_phrase(normalized, _MEDICAL_DEMO_EVAL_TERMS)
    )


def _normalize_request_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def _contains_all_phrases(value: str, phrases: tuple[str, ...]) -> bool:
    return all(phrase in value for phrase in phrases)


def _contains_any_phrase(value: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in value for phrase in phrases)


def _maybe_run_medical_safety_demo(
    conversation_path: Path,
    user_message: str,
    data_root: str,
    event_sink: Any | None,
) -> dict[str, Any] | None:
    """Run the live medical safety workflow when the user asks for it."""
    if not is_medical_safety_demo_request(user_message):
        return None

    live_result = _run_live_medical_safety_workflow(Path(data_root), event_sink)
    if "error" in live_result:
        return live_result

    before = live_result["baseline_artifact"]
    after = live_result["tuned_artifact"]

    baseline_message = {
        "role": "assistant",
        "content": _content_blocks(
            "I selected `medical_safety_triage` as the relevant benchmark "
            "and ran it against `medical-assistant` before training.",
        ),
        "artifact": before,
    }
    tuned_message = {
        "role": "assistant",
        "content": _content_blocks(
            "Fine-tuning completed on `medical_safety_triage_training`, "
            "then I ran the same benchmark again for a direct before/after comparison.",
        ),
        "artifact": after,
        "workspaceDirective": _medical_compare_directive(),
    }
    messages = _load_conversation_raw(conversation_path)
    messages.extend([
        {"role": "user", "content": user_message},
        baseline_message,
        tuned_message,
    ])
    _save_conversation(conversation_path, messages)

    return {
        "role": "assistant",
        "content": (
            "I ran the medical safety loop live and surfaced the before/after "
            "comparison in the chat."
        ),
        "tools_used": ["run_benchmark", "train", "run_benchmark"],
        "artifact_messages": [
            _display_message(baseline_message),
            _display_message(tuned_message),
        ],
        "workspace_mode": "compare",
        "workspace_cards": ["previous_eval", "latest_eval", "context"],
    }


def _run_live_medical_safety_workflow(
    data_root: Path,
    event_sink: Any | None,
) -> dict[str, Any]:
    """Execute benchmark, training, and follow-up benchmark through real tools."""
    from serve.agent_events import emit_event

    emit_event(event_sink, "status", text="Resolving medical-assistant from registry")
    model_path = _resolve_registered_model_path(data_root, "medical-assistant")
    baseline_result = _execute_live_tool(
        event_sink,
        name="run_benchmark",
        tool_input={
            "model_path": model_path,
            "benchmarks": _MEDICAL_BENCHMARK,
            "max_samples": 0,
        },
        input_summary=f"model=medical-assistant benchmark={_MEDICAL_BENCHMARK}",
    )
    if "error" in baseline_result:
        return _live_workflow_error("Baseline evaluation failed", baseline_result)

    suffix = uuid.uuid4().hex[:8]
    tuned_model_name = f"{_MEDICAL_TUNED_MODEL_PREFIX}-{suffix}"
    output_dir = f"tmp/medical_safety_triage_live_{suffix}"
    train_result = _execute_live_tool(
        event_sink,
        name="train",
        tool_input={
            "method": "sft",
            "method_args": json.dumps({
                "dataset_name": _MEDICAL_DATASET,
                "sft_data_path": "data/medical_safety_triage_training.jsonl",
                "base_model": model_path,
                "output_dir": output_dir,
                "epochs": 3,
                "batch_size": 16,
                "learning_rate": 5e-4,
                "max_token_length": 512,
                "model_name": tuned_model_name,
            }),
        },
        input_summary=f"dataset={_MEDICAL_DATASET} method=sft",
    )
    if "error" in train_result:
        return _live_workflow_error("Fine-tuning failed", train_result)

    tuned_model_path = str(train_result.get("model_path", ""))
    if not tuned_model_path:
        return _live_workflow_error(
            "Fine-tuning did not return a model path",
            train_result,
        )
    tuned_result = _execute_live_tool(
        event_sink,
        name="run_benchmark",
        tool_input={
            "model_path": tuned_model_path,
            "benchmarks": _MEDICAL_BENCHMARK,
            "max_samples": 0,
        },
        input_summary=f"model={tuned_model_name} benchmark={_MEDICAL_BENCHMARK}",
    )
    if "error" in tuned_result:
        return _live_workflow_error("Follow-up evaluation failed", tuned_result)

    return {
        "baseline_artifact": _eval_artifact_from_result(
            baseline_result,
            title="medical-assistant baseline",
        ),
        "tuned_artifact": _eval_artifact_from_result(
            tuned_result,
            title=f"{tuned_model_name} SFT",
        ),
    }


def _execute_live_tool(
    event_sink: Any | None,
    name: str,
    tool_input: dict[str, Any],
    input_summary: str,
) -> dict[str, Any]:
    """Execute one MCP tool and emit the same trace events as the agent loop."""
    from serve.agent_events import emit_event, summarize_tool_output

    emit_event(
        event_sink,
        "tool_call",
        tool_name=name,
        input_summary=input_summary,
    )
    result_text = execute_tool(name, tool_input)
    emit_event(
        event_sink,
        "tool_result",
        tool_name=name,
        output_summary=summarize_tool_output(result_text),
    )
    return _parse_tool_result(result_text)


def _parse_tool_result(result_text: str) -> dict[str, Any]:
    parsed = _try_parse_json_object(result_text)
    if parsed is None:
        return {"error": result_text}
    return parsed


def _try_parse_json_object(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _live_workflow_error(message: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": f"{message}: {result.get('error', result)}",
        "tools_used": ["run_benchmark", "train", "run_benchmark"],
        "error": f"{message}: {result.get('error', result)}",
    }


def _resolve_registered_model_path(data_root: Path, model_name: str) -> str:
    from store.model_registry import ModelRegistry

    entry = ModelRegistry(data_root).get_model(model_name)
    return entry.model_path or model_name


def _eval_artifact_from_result(
    result: dict[str, Any],
    title: str,
) -> dict[str, Any]:
    benchmarks = _eval_benchmarks_from_result(result)
    return {
        "kind": "eval",
        "jobId": str(result.get("job_id", "")),
        "title": title,
        "cluster": "local",
        "averageScore": float(result.get("average_score", 0.0)),
        "benchmarkCount": len(benchmarks),
        "topBenchmarks": sorted(
            [{"name": b["name"], "score": b["score"]} for b in benchmarks],
            key=lambda item: float(item["score"]),
            reverse=True,
        )[:3],
        "benchmarks": sorted(
            benchmarks,
            key=lambda item: float(item["score"]),
            reverse=True,
        ),
    }


def _eval_benchmarks_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = result.get("benchmarks", [])
    if not isinstance(rows, list):
        return []
    benchmarks: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        total = row.get("total", row.get("num_examples", row.get("numExamples", 0)))
        benchmarks.append({
            "name": str(row.get("name", "")),
            "score": float(row.get("score", 0.0)),
            "correct": int(row.get("correct", 0) or 0),
            "numExamples": int(total or 0),
        })
    return benchmarks


def _content_blocks(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def _display_message(message: dict[str, Any]) -> dict[str, Any]:
    text = "\n".join(
        block.get("text", "")
        for block in message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()
    display = {"role": "assistant", "content": text}
    _copy_display_metadata(message, display)
    return display


def _medical_compare_directive() -> dict[str, object]:
    return {
        "mode": "compare",
        "cards": ["previous_eval", "latest_eval", "context"],
    }


def run_agent_turn(
    conversation_path: Path,
    user_message: str,
    app_context: dict[str, Any],
    api_key: str,
    data_root: str,
    provider: str = "anthropic",
    model: str = "",
    ollama_url: str = "",
    event_sink: Any | None = None,
) -> dict[str, Any]:
    """Run a single agent conversation turn with tool use."""
    from serve.agent_backends import call_anthropic, call_gemini, call_ollama, call_openai
    from serve.agent_events import emit_event, summarize_tool_input, summarize_tool_output
    from serve.mcp_server import _ensure_backends
    _ensure_backends()

    demo_result = _maybe_run_medical_safety_demo(
        conversation_path, user_message, data_root, event_sink,
    )
    if demo_result is not None:
        return demo_result

    tools, _ = _get_tools()
    system = _build_system_prompt(app_context, data_root)
    messages = _load_conversation(conversation_path)
    messages.append({"role": "user", "content": user_message})
    messages = _trim_conversation(messages)
    tools_used: list[str] = []
    emit_event(event_sink, "status", text="Analyzing request")

    for _ in range(_MAX_TOOL_LOOPS):
        if provider == "ollama":
            effective_model = model or _DEFAULT_OLLAMA_MODEL
            effective_url = ollama_url or _DEFAULT_OLLAMA_URL
            response = call_ollama(effective_url, effective_model, system, messages, tools)
        elif provider == "openai":
            effective_model = model or _DEFAULT_OPENAI_MODEL
            response = call_openai(api_key, effective_model, system, messages, tools)
        elif provider == "gemini":
            effective_model = model or _DEFAULT_GEMINI_MODEL
            response = call_gemini(effective_model, system, messages, tools, api_key=api_key)
        else:
            effective_model = model or _DEFAULT_ANTHROPIC_MODEL
            response = call_anthropic(api_key, effective_model, system, messages, tools)

        messages.append({"role": "assistant", "content": response.content_blocks})
        response_text = "\n".join(
            block.get("text", "")
            for block in response.content_blocks
            if block.get("type") == "text" and block.get("text")
        ).strip()
        if response_text and response.stop_reason == "tool_use":
            emit_event(event_sink, "assistant_note", text=response_text)

        if response.stop_reason != "tool_use":
            break

        tool_results: list[dict[str, Any]] = []
        for block in response.content_blocks:
            if block.get("type") != "tool_use":
                continue
            tools_used.append(block["name"])
            emit_event(
                event_sink,
                "tool_call",
                tool_name=block["name"],
                input_summary=summarize_tool_input(block.get("input", {})),
            )
            result_str = execute_tool(block["name"], block.get("input", {}))
            emit_event(
                event_sink,
                "tool_result",
                tool_name=block["name"],
                output_summary=summarize_tool_output(result_str),
            )
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.get("id", ""),
                "content": result_str,
            })
        messages.append({"role": "user", "content": tool_results})

    text_parts = [
        b.get("text", "") for b in response.content_blocks if b.get("type") == "text"
    ]
    _save_conversation(conversation_path, messages)

    full_text = "\n".join(text_parts)

    # Extract script_update if the agent returned one
    script_update, full_text = _pop_control_tag(full_text, "script_update")

    # Extract navigate_to if the agent returned one
    navigate_to = None
    route, full_text = _pop_control_tag(full_text, "navigate_to")
    if route:
        if route in _VALID_ROUTES:
            navigate_to = route
            emit_event(event_sink, "navigation", route=route)

    # Extract pending_chain if the agent declared follow-up steps
    pending_chain = None
    chain_text, full_text = _pop_control_tag(full_text, "pending_chain")
    if chain_text:
        # Find the job_id from tool results in this turn
        chain_job_id = _extract_submitted_job_id(messages)
        if chain_job_id and chain_text:
            steps = [s.strip() for s in chain_text.splitlines() if s.strip()]
            _save_pending_chain(
                conversation_path.parent, chain_job_id, steps, user_message,
            )
            emit_event(event_sink, "pending_chain", job_id=chain_job_id, steps=steps)
            pending_chain = {"job_id": chain_job_id, "steps": steps}

    raw_workspace_mode, full_text = _pop_control_tag(full_text, "workspace_mode")
    workspace_mode = _parse_workspace_mode(raw_workspace_mode)
    raw_workspace_cards, full_text = _pop_control_tag(full_text, "workspace_cards")
    workspace_cards = _parse_workspace_cards(raw_workspace_cards)

    result: dict[str, Any] = {
        "role": "assistant",
        "content": full_text,
        "tools_used": tools_used,
    }
    if script_update:
        result["script_update"] = script_update
    if navigate_to:
        result["navigate_to"] = navigate_to
    if pending_chain:
        result["pending_chain"] = pending_chain
    if workspace_mode:
        result["workspace_mode"] = workspace_mode
    if workspace_cards:
        result["workspace_cards"] = workspace_cards
    return result


def load_conversation_for_display(conversation_path: Path) -> list[dict[str, Any]]:
    """Load conversation and convert to display format (role + content text)."""
    messages = _load_conversation_raw(conversation_path)
    display: list[dict[str, Any]] = []
    for msg in messages:
        if msg["role"] == "user":
            content = msg["content"]
            if isinstance(content, str):
                display.append({"role": "user", "content": content})
            # Skip tool_result user messages (they're not user-typed)
        elif msg["role"] == "assistant":
            content = msg["content"]
            if isinstance(content, str):
                cleaned_content = _strip_control_tags(content)
                if cleaned_content:
                    entry = {"role": "assistant", "content": cleaned_content}
                    _copy_display_metadata(msg, entry)
                    display.append(entry)
            elif isinstance(content, list):
                text_parts = []
                tool_names = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            tool_names.append(block.get("name", ""))
                cleaned_content = _strip_control_tags("\n".join(text_parts))
                if cleaned_content:
                    entry: dict[str, Any] = {
                        "role": "assistant",
                        "content": cleaned_content,
                    }
                    if tool_names:
                        entry["tools_used"] = tool_names
                    _copy_display_metadata(msg, entry)
                    display.append(entry)
    return display


def _copy_display_metadata(source: dict[str, Any], target: dict[str, Any]) -> None:
    artifact = source.get("artifact")
    if isinstance(artifact, dict):
        target["artifact"] = artifact
    directive = source.get("workspaceDirective")
    if isinstance(directive, dict):
        target["workspaceDirective"] = directive


# ── Pending chain helpers ─────────────────────────────────────────

_CHAIN_FILE = "pending_chain.json"


def _save_pending_chain(
    agent_dir: Path,
    job_id: str,
    steps: list[str],
    original_request: str,
) -> None:
    from datetime import datetime, timezone
    data = {
        "waiting_on_job_id": job_id,
        "remaining_steps": steps,
        "original_request": original_request,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = agent_dir / _CHAIN_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load_pending_chain(agent_dir: Path) -> dict[str, Any] | None:
    path = agent_dir / _CHAIN_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def delete_pending_chain(agent_dir: Path) -> None:
    path = agent_dir / _CHAIN_FILE
    if path.exists():
        path.unlink()


def _extract_submitted_job_id(messages: list[dict[str, Any]]) -> str | None:
    """Find the job_id from the most recent tool result in the conversation."""
    for msg in reversed(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            # Tool result blocks
            result_content = block.get("content", "")
            if isinstance(result_content, str) and '"job_id"' in result_content:
                try:
                    parsed = json.loads(result_content)
                    if "job_id" in parsed:
                        return parsed["job_id"]
                except (json.JSONDecodeError, TypeError):
                    pass
    return None
