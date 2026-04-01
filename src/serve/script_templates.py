"""Training script template generation and config parsing.

Generates standalone Python scripts from form values, and parses
config sections back into form values. Scripts use the crucible_sdk
public API so they're both readable and runnable.
"""

from __future__ import annotations

import re
from typing import Any

CONFIG_BEGIN = "# CRUCIBLE:BEGIN_CONFIG"
CONFIG_END = "# CRUCIBLE:END_CONFIG"

# ── Shared fields across all methods ─────────────────────────────────

_SHARED_FIELDS = (
    "model_id", "data_path", "output_dir", "epochs", "learning_rate",
    "batch_size", "max_length", "precision", "validation_split", "seed",
)


def generate_script(method: str, config: dict[str, Any]) -> str:
    """Generate a complete training script from form config values.

    Args:
        method: Training method ID (sft, lora-train, dpo-train, etc.).
        config: Dict of config values from the form.

    Returns:
        Complete Python script as a string.
    """
    clean = method.replace("-train", "").replace("-", "")
    generator = _GENERATORS.get(clean, _generate_sft)
    return generator(config)


def parse_script_config(script: str) -> dict[str, Any]:
    """Extract config values from a script's CRUCIBLE config section.

    Parses simple variable assignments between the BEGIN/END markers.

    Args:
        script: Full Python script content.

    Returns:
        Dict of variable name → value.
    """
    config: dict[str, Any] = {}
    in_config = False
    for line in script.splitlines():
        stripped = line.strip()
        if stripped == CONFIG_BEGIN:
            in_config = True
            continue
        if stripped == CONFIG_END:
            break
        if not in_config:
            continue
        if stripped.startswith("#") or not stripped:
            continue
        match = re.match(r"^(\w+)\s*=\s*(.+?)(?:\s*#.*)?$", stripped)
        if match:
            name, value_str = match.group(1), match.group(2).strip()
            config[name] = _parse_value(value_str)
    return config


def _parse_value(s: str) -> Any:
    """Parse a Python literal value from a string."""
    s = s.strip()
    if s in ("True", "true"):
        return True
    if s in ("False", "false"):
        return False
    if s == "None":
        return None
    # String
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    # Tuple of strings
    if s.startswith("(") and s.endswith(")"):
        inner = s[1:-1]
        items = [i.strip().strip("'\"") for i in inner.split(",") if i.strip()]
        return tuple(items)
    # Number
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s


def _q(v: Any) -> str:
    """Quote a value for Python source."""
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, tuple):
        inner = ", ".join(f'"{i}"' for i in v)
        return f"({inner},)"
    if v is None:
        return "None"
    return str(v)


def _config_block(fields: list[tuple[str, Any, str]]) -> str:
    """Build the config section with aligned comments."""
    lines = [CONFIG_BEGIN]
    max_assign = max(len(f"{name} = {_q(val)}") for name, val, _ in fields)
    for name, val, comment in fields:
        assign = f"{name} = {_q(val)}"
        if comment:
            pad = " " * (max_assign - len(assign) + 4)
            lines.append(f"{assign}{pad}# {comment}")
        else:
            lines.append(assign)
    lines.append(CONFIG_END)
    return "\n".join(lines)


# ── SFT ──────────────────────────────────────────────────────────────

def _generate_sft(config: dict[str, Any]) -> str:
    cfg = _config_block([
        ("model_id", config.get("model_id", "gpt2"), "Base model (HuggingFace ID or local path)"),
        ("data_path", config.get("data_path", "./data.jsonl"), "JSONL with prompt/response pairs"),
        ("output_dir", config.get("output_dir", "./outputs/sft"), "Where to save the trained model"),
        ("epochs", config.get("epochs", 3), "Number of training epochs"),
        ("learning_rate", config.get("learning_rate", 2e-5), ""),
        ("batch_size", config.get("batch_size", 16), ""),
        ("max_length", config.get("max_length", 512), "Max sequence length"),
        ("precision", config.get("precision", "auto"), "auto, fp32, fp16, or bf16"),
    ])
    return f'''{cfg}

import crucible_sdk as crucible

model, tokenizer = crucible.load_model(model_id, method="sft")
dataset = crucible.load_dataset(data_path, format="sft")

result = crucible.train(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    method="sft",
    epochs=epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    max_length=max_length,
    precision=precision,
    output_dir=output_dir,
)
'''


# ── LoRA ─────────────────────────────────────────────────────────────

def _generate_lora(config: dict[str, Any]) -> str:
    cfg = _config_block([
        ("model_id", config.get("model_id", "gpt2"), "Base model"),
        ("data_path", config.get("data_path", "./data.jsonl"), "JSONL with prompt/response pairs"),
        ("output_dir", config.get("output_dir", "./outputs/lora"), ""),
        ("epochs", config.get("epochs", 3), ""),
        ("learning_rate", config.get("learning_rate", 2e-4), ""),
        ("batch_size", config.get("batch_size", 16), ""),
        ("max_length", config.get("max_length", 512), ""),
        ("precision", config.get("precision", "auto"), ""),
        ("lora_rank", config.get("lora_rank", 8), "LoRA rank"),
        ("lora_alpha", config.get("lora_alpha", 16.0), "LoRA alpha scaling"),
        ("lora_dropout", config.get("lora_dropout", 0.0), "LoRA dropout"),
    ])
    return f'''{cfg}

import crucible_sdk as crucible

model, tokenizer = crucible.load_model(
    model_id, method="lora",
    lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
)
dataset = crucible.load_dataset(data_path, format="sft")

result = crucible.train(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    method="lora",
    epochs=epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    max_length=max_length,
    precision=precision,
    output_dir=output_dir,
)
'''


# ── QLoRA ────────────────────────────────────────────────────────────

def _generate_qlora(config: dict[str, Any]) -> str:
    cfg = _config_block([
        ("model_id", config.get("model_id", "gpt2"), "Base model"),
        ("data_path", config.get("data_path", "./data.jsonl"), ""),
        ("output_dir", config.get("output_dir", "./outputs/qlora"), ""),
        ("epochs", config.get("epochs", 3), ""),
        ("learning_rate", config.get("learning_rate", 2e-4), ""),
        ("batch_size", config.get("batch_size", 16), ""),
        ("max_length", config.get("max_length", 512), ""),
        ("precision", config.get("precision", "auto"), ""),
        ("lora_rank", config.get("lora_rank", 8), ""),
        ("lora_alpha", config.get("lora_alpha", 16.0), ""),
        ("lora_dropout", config.get("lora_dropout", 0.0), ""),
        ("quantization_bits", config.get("quantization_bits", 4), "4 or 8 bit"),
        ("qlora_type", config.get("qlora_type", "nf4"), "nf4 or fp4"),
    ])
    return f'''{cfg}

import crucible_sdk as crucible

model, tokenizer = crucible.load_model(
    model_id, method="qlora",
    lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
    quantization_bits=quantization_bits, qlora_type=qlora_type,
)
dataset = crucible.load_dataset(data_path, format="sft")

result = crucible.train(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    method="qlora",
    epochs=epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    max_length=max_length,
    precision=precision,
    output_dir=output_dir,
)
'''


# ── DPO ──────────────────────────────────────────────────────────────

def _generate_dpo(config: dict[str, Any]) -> str:
    cfg = _config_block([
        ("model_id", config.get("model_id", "gpt2"), "Base model"),
        ("data_path", config.get("data_path", "./data.jsonl"), "JSONL with prompt/chosen/rejected"),
        ("output_dir", config.get("output_dir", "./outputs/dpo"), ""),
        ("epochs", config.get("epochs", 3), ""),
        ("learning_rate", config.get("learning_rate", 5e-5), ""),
        ("batch_size", config.get("batch_size", 16), ""),
        ("max_length", config.get("max_length", 512), ""),
        ("precision", config.get("precision", "auto"), ""),
        ("beta", config.get("beta", 0.1), "DPO beta (preference strength)"),
    ])
    return f'''{cfg}

import crucible_sdk as crucible

model, tokenizer = crucible.load_model(model_id, method="dpo")
dataset = crucible.load_dataset(data_path, format="dpo")

result = crucible.train(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    method="dpo",
    epochs=epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    max_length=max_length,
    precision=precision,
    output_dir=output_dir,
    beta=beta,
)
'''


# ── KTO ──────────────────────────────────────────────────────────────

def _generate_kto(config: dict[str, Any]) -> str:
    cfg = _config_block([
        ("model_id", config.get("model_id", "gpt2"), "Base model"),
        ("data_path", config.get("data_path", "./data.jsonl"), "JSONL with prompt/response/is_desirable"),
        ("output_dir", config.get("output_dir", "./outputs/kto"), ""),
        ("epochs", config.get("epochs", 3), ""),
        ("learning_rate", config.get("learning_rate", 5e-5), ""),
        ("batch_size", config.get("batch_size", 16), ""),
        ("max_length", config.get("max_length", 512), ""),
        ("precision", config.get("precision", "auto"), ""),
    ])
    return f'''{cfg}

import crucible_sdk as crucible

model, tokenizer = crucible.load_model(model_id, method="kto")
dataset = crucible.load_dataset(data_path, format="kto")

result = crucible.train(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    method="kto",
    epochs=epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    max_length=max_length,
    precision=precision,
    output_dir=output_dir,
)
'''


# ── ORPO ─────────────────────────────────────────────────────────────

def _generate_orpo(config: dict[str, Any]) -> str:
    cfg = _config_block([
        ("model_id", config.get("model_id", "gpt2"), "Base model"),
        ("data_path", config.get("data_path", "./data.jsonl"), "JSONL with prompt/chosen/rejected"),
        ("output_dir", config.get("output_dir", "./outputs/orpo"), ""),
        ("epochs", config.get("epochs", 3), ""),
        ("learning_rate", config.get("learning_rate", 5e-5), ""),
        ("batch_size", config.get("batch_size", 16), ""),
        ("max_length", config.get("max_length", 512), ""),
        ("precision", config.get("precision", "auto"), ""),
        ("beta", config.get("beta", 0.1), "ORPO beta"),
    ])
    return f'''{cfg}

import crucible_sdk as crucible

model, tokenizer = crucible.load_model(model_id, method="orpo")
dataset = crucible.load_dataset(data_path, format="dpo")

result = crucible.train(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    method="orpo",
    epochs=epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    max_length=max_length,
    precision=precision,
    output_dir=output_dir,
    beta=beta,
)
'''


# ── Domain Adaptation ────────────────────────────────────────────────

def _generate_domainadapt(config: dict[str, Any]) -> str:
    cfg = _config_block([
        ("model_id", config.get("model_id", "gpt2"), "Base model for continued pretraining"),
        ("data_path", config.get("data_path", "./data.jsonl"), "JSONL with text field"),
        ("output_dir", config.get("output_dir", "./outputs/domain-adapt"), ""),
        ("epochs", config.get("epochs", 3), ""),
        ("learning_rate", config.get("learning_rate", 5e-5), ""),
        ("batch_size", config.get("batch_size", 16), ""),
        ("max_length", config.get("max_length", 512), ""),
        ("precision", config.get("precision", "auto"), ""),
    ])
    return f'''{cfg}

import crucible_sdk as crucible

model, tokenizer = crucible.load_model(model_id, method="domain-adapt")
dataset = crucible.load_dataset(data_path, format="text")

result = crucible.train(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    method="domain-adapt",
    epochs=epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    max_length=max_length,
    precision=precision,
    output_dir=output_dir,
)
'''


# ── Generator dispatch ───────────────────────────────────────────────

_GENERATORS: dict[str, Any] = {
    "sft": _generate_sft,
    "lora": _generate_lora,
    "qlora": _generate_qlora,
    "dpo": _generate_dpo,
    "kto": _generate_kto,
    "orpo": _generate_orpo,
    "domainadapt": _generate_domainadapt,
    # GRPO, RLVR, RLHF, distill, multimodal, basic train
    # use SFT template with method swapped (same script structure)
    "grpo": _generate_sft,
    "rlvr": _generate_sft,
    "rlhf": _generate_sft,
    "distill": _generate_sft,
    "multimodal": _generate_sft,
    "train": _generate_sft,
}
