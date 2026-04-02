"""Crucible public SDK — simple interface for training scripts.

This module provides the functions that user-editable training scripts
call. Each function wraps Crucible's internal machinery behind a clean
API that researchers can understand and replace piece by piece.

Usage in a training script:

    import crucible
    model, tokenizer = crucible.load_model("gpt2")
    dataset = crucible.load_dataset("sft-mini", format="sft")
    result = crucible.train(model, tokenizer, dataset, method="sft", epochs=3)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_model(
    model_id: str,
    method: str = "sft",
    *,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    quantization_bits: int = 4,
    qlora_type: str = "nf4",
    precision: str = "auto",
) -> tuple[Any, Any]:
    """Load a model and tokenizer, ready for training.

    Handles HuggingFace model IDs, local .pt paths, and applies
    method-specific setup (LoRA adapters, quantization, etc.).

    Args:
        model_id: HuggingFace model ID (e.g. 'gpt2') or path to .pt file.
        method: Training method — determines what setup is applied.
        lora_rank: LoRA rank (for lora/qlora methods).
        lora_alpha: LoRA alpha scaling (for lora/qlora methods).
        lora_dropout: LoRA dropout (for lora/qlora methods).
        lora_target_modules: Modules to apply LoRA to.
        quantization_bits: Bits for quantization (qlora only).
        qlora_type: Quantization type (qlora only).
        precision: Precision mode (auto, fp32, fp16, bf16).

    Returns:
        Tuple of (model, tokenizer).
    """
    import torch
    from serve.hf_model_loader import is_huggingface_model_id

    if is_huggingface_model_id(model_id):
        return _load_hf_model(
            model_id, method,
            lora_rank=lora_rank, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout, lora_target_modules=lora_target_modules,
            quantization_bits=quantization_bits, qlora_type=qlora_type,
            precision=precision,
        )
    return _load_crucible_model(model_id)


def _load_hf_model(
    model_id: str,
    method: str,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Load a HuggingFace model with method-specific setup."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs: dict[str, Any] = {}
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = "cpu"

    # QLoRA: load with 4-bit quantization
    if method in ("qlora", "qlora-train"):
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=kwargs.get("qlora_type", "nf4"),
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA/QLoRA: inject adapters
    if method in ("lora", "lora-train", "qlora", "qlora-train"):
        from peft import LoraConfig, get_peft_model
        target_mods: list[str] | str = list(kwargs.get("lora_target_modules", ("q_proj", "v_proj")))
        model_names = {name for name, _ in model.named_modules()}
        if not any(any(t in n for t in target_mods) for n in model_names):
            target_mods = "all-linear"
        lora_config = LoraConfig(
            r=kwargs.get("lora_rank", 8),
            lora_alpha=kwargs.get("lora_alpha", 16.0),
            lora_dropout=kwargs.get("lora_dropout", 0.0),
            target_modules=target_mods,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    return model, tokenizer


def _load_crucible_model(model_path: str) -> tuple[Any, Any]:
    """Load a Crucible .pt model and tokenizer."""
    import torch
    from serve.device_selection import resolve_execution_device
    from serve.chat_option_resolver import resolve_chat_training_options
    from serve.architecture_loader import load_training_model
    from serve.model_weights import load_initial_weights, read_model_state_dict
    from serve.training_metadata import load_tokenizer
    from core.chat_types import ChatOptions

    device = resolve_execution_device(torch)
    model_state = read_model_state_dict(torch, model_path, device)
    chat_opts = ChatOptions(model_path=model_path, prompt="")
    training_options = resolve_chat_training_options(chat_opts, model_state)

    tokenizer = load_tokenizer(model_path)
    if tokenizer is None:
        from core.errors import CrucibleServeError
        raise CrucibleServeError(f"No tokenizer found at {model_path}.")

    vocab_size = len(tokenizer.vocabulary)
    model = load_training_model(torch, training_options, vocab_size)
    model = model.to(device)
    load_initial_weights(torch, model, model_path, device)

    return model, tokenizer


def load_dataset(
    source: str,
    format: str = "sft",
) -> Any:
    """Load a dataset for training.

    Args:
        source: Path to JSONL/Parquet file, or name of an ingested dataset.
        format: Data format — determines how the file is parsed.
            'sft': expects {"prompt": ..., "response": ...}
            'dpo': expects {"prompt": ..., "chosen": ..., "rejected": ...}
            'kto': expects {"prompt": ..., "response": ..., "is_desirable": bool}
            'text': expects {"text": ...}
            'prompts': expects {"prompt": ...}

    Returns:
        HuggingFace Dataset.
    """
    from datasets import Dataset
    from serve.data_file_reader import read_data_rows

    # If it's an ingested dataset name, resolve to the records file
    resolved = _resolve_dataset_path(source)
    rows = list(read_data_rows(resolved))

    if format == "sft":
        texts = []
        for row in rows:
            # Accept multiple field layouts:
            # 1. {text: "..."} — already flattened (ingested datasets)
            # 2. {prompt: "...", response: "..."} — structured
            # 3. {instruction: "...", output: "..."} — Alpaca-style
            # 4. {input: "...", output: "..."} — generic
            if "text" in row:
                texts.append(str(row["text"]))
            elif "prompt" in row and "response" in row:
                system = row.get("system_prompt", "")
                text = f"{system}\n\n" if system else ""
                text += f"{row['prompt']}\n{row['response']}"
                texts.append(text)
            elif "instruction" in row:
                inp = row.get("input", "")
                out = row.get("output", "")
                text = f"{row['instruction']}\n{inp}\n{out}" if inp else f"{row['instruction']}\n{out}"
                texts.append(text)
            elif "content" in row:
                texts.append(str(row["content"]))
            else:
                raise ValueError(
                    f"Row {len(texts)} has no recognized SFT fields. "
                    f"Expected 'text', 'prompt'+'response', or 'instruction'+'output'. "
                    f"Got keys: {list(row.keys())[:5]}"
                )
        return Dataset.from_dict({"text": texts})

    if format == "dpo":
        # Try to recover structured fields from ingested records
        rows = _try_recover_source_rows(rows, source, ("prompt", "chosen", "rejected"))
        for i, row in enumerate(rows):
            for field in ("prompt", "chosen", "rejected"):
                if field not in row:
                    raise ValueError(
                        f"Row {i} missing required field '{field}' for DPO format. "
                        f"Got keys: {list(row.keys())[:5]}. "
                        f"Ingested datasets flatten fields — use the original JSONL file directly."
                    )
        return Dataset.from_dict({
            "prompt": [r["prompt"] for r in rows],
            "chosen": [r["chosen"] for r in rows],
            "rejected": [r["rejected"] for r in rows],
        })

    if format == "kto":
        rows = _try_recover_source_rows(rows, source, ("prompt", "response"))
        for i, row in enumerate(rows):
            for field in ("prompt", "response"):
                if field not in row:
                    raise ValueError(
                        f"Row {i} missing required field '{field}' for KTO format. "
                        f"Got keys: {list(row.keys())[:5]}. "
                        f"Ingested datasets flatten fields — use the original JSONL file directly."
                    )
        return Dataset.from_dict({
            "prompt": [r["prompt"] for r in rows],
            "completion": [r["response"] for r in rows],
            "label": [r.get("is_desirable", True) for r in rows],
        })

    if format == "text":
        return Dataset.from_dict({
            "text": [r.get("text", "") for r in rows],
        })

    if format == "prompts":
        return Dataset.from_dict({
            "text": [r.get("prompt", "") for r in rows],
        })

    raise ValueError(f"Unknown format: {format}")


def _try_recover_source_rows(
    rows: list[dict[str, object]],
    source: str,
    required_fields: tuple[str, ...],
) -> list[dict[str, object]]:
    """If rows lack required fields (ingested format), try recovering them.

    Strategy 1: Check metadata.extra_fields for the structured fields
    (preserved during ingest since the flattening keeps originals as extras).
    Strategy 2: Try loading the original source file from metadata.source_uri.
    """
    if not rows or all(f in rows[0] for f in required_fields):
        return rows  # Already has the fields we need

    # Strategy 1: Recover from metadata.extra_fields
    meta = rows[0].get("metadata", {})
    if isinstance(meta, dict):
        extras = meta.get("extra_fields", {})
        if isinstance(extras, dict) and all(f in extras for f in required_fields):
            # All required fields are in extra_fields — reconstruct rows
            recovered = []
            for row in rows:
                m = row.get("metadata", {})
                ef = m.get("extra_fields", {}) if isinstance(m, dict) else {}
                if isinstance(ef, dict) and all(f in ef for f in required_fields):
                    recovered.append(dict(ef))
                else:
                    break
            if len(recovered) == len(rows):
                return recovered

    # Strategy 2: Try loading the original source file
    if isinstance(meta, dict):
        source_uri = str(meta.get("source_uri", ""))
        if ":" in source_uri:
            source_file = source_uri.rsplit(":", 1)[0]
        else:
            source_file = source_uri
        if source_file and Path(source_file).exists():
            from serve.data_file_reader import read_data_rows
            recovered = list(read_data_rows(source_file))
            if recovered and all(f in recovered[0] for f in required_fields):
                return recovered
    return rows


def _resolve_dataset_path(source: str) -> str:
    """Resolve a dataset name or path to a file path."""
    path = Path(source)
    if path.exists():
        return str(path)
    # Try as an ingested dataset name
    from core.config import CrucibleConfig
    config = CrucibleConfig.from_env()
    dataset_dir = config.data_root / "datasets" / source
    records = dataset_dir / "records.jsonl"
    if records.exists():
        return str(records)
    raise FileNotFoundError(f"Dataset '{source}' not found as file or ingested dataset.")


def train(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    method: str = "sft",
    *,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    max_length: int = 512,
    precision: str = "auto",
    output_dir: str = "./outputs/train",
    model_name: str = "",
    weight_decay: float = 0.0,
    validation_split: float = 0.2,
    seed: int = 42,
    beta: float = 0.1,
    **extra: Any,
) -> dict[str, Any]:
    """Train a model and save results.

    Works with any model/tokenizer pair — HuggingFace, Crucible, or custom.
    Dispatches to the appropriate trl Trainer based on the method.

    Args:
        model: The model to train.
        tokenizer: The tokenizer.
        dataset: HuggingFace Dataset.
        method: Training method (sft, lora, qlora, dpo, kto, orpo, etc.).
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        batch_size: Per-device batch size.
        max_length: Maximum sequence length.
        precision: Precision mode (auto, fp32, fp16, bf16).
        output_dir: Where to save outputs.
        weight_decay: Weight decay for optimizer.
        validation_split: Fraction of data for validation.
        seed: Random seed.
        beta: Beta parameter (for DPO/KTO/ORPO).
        **extra: Additional method-specific arguments.

    Returns:
        Dict with model_path, history_path, epochs_completed.
    """
    import torch

    # Derive output dir from model_name if not explicitly set
    if not output_dir or output_dir == "./outputs/train":
        if model_name:
            output_dir = f"./outputs/{model_name.replace(' ', '_')}"
        else:
            output_dir = "./outputs/train"
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Detect if this is a HuggingFace model (trl path) or Crucible .pt (custom loop)
    is_hf = _is_hf_tokenizer(tokenizer)

    if is_hf:
        return _train_with_trl(
            model, tokenizer, dataset, method, out,
            epochs=epochs, learning_rate=learning_rate, batch_size=batch_size,
            max_length=max_length, precision=precision, weight_decay=weight_decay,
            validation_split=validation_split, seed=seed, beta=beta,
            model_name=model_name, **extra,
        )
    else:
        return _train_crucible(
            model, tokenizer, dataset, method, out,
            epochs=epochs, learning_rate=learning_rate, batch_size=batch_size,
            max_length=max_length, model_name=model_name,
            validation_split=validation_split, seed=seed,
        )


def _is_hf_tokenizer(tokenizer: Any) -> bool:
    """Check if the tokenizer is a HuggingFace PreTrainedTokenizer."""
    try:
        from transformers import PreTrainedTokenizerBase
        return isinstance(tokenizer, PreTrainedTokenizerBase)
    except ImportError:
        return hasattr(tokenizer, "save_pretrained") and hasattr(tokenizer, "pad_token")


def _train_with_trl(
    model: Any, tokenizer: Any, dataset: Any, method: str, out: Path,
    *, epochs: int, learning_rate: float, batch_size: int, max_length: int,
    precision: str, weight_decay: float, validation_split: float, seed: int,
    beta: float, model_name: str, **extra: Any,
) -> dict[str, Any]:
    """Train using trl Trainers (for HuggingFace models)."""
    import torch
    import trl

    split = dataset.train_test_split(test_size=validation_split, seed=seed)
    use_bf16 = precision in ("bf16", "auto") and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = precision == "fp16"
    base_args = {
        "output_dir": str(out),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 5,
        "seed": seed,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "max_grad_norm": 1.0,
        "report_to": "none",
        "dataloader_pin_memory": False,
        "max_length": max_length,
    }
    trainer = _create_trainer(
        method, model, tokenizer, split, base_args, beta=beta, **extra,
    )
    print(f"Training ({method}) with trl...", flush=True)
    trainer.train()
    print("Training complete.", flush=True)
    return save_result(model, tokenizer, str(out), epochs, model_name=model_name)


def _train_crucible(
    model: Any, tokenizer: Any, dataset: Any, method: str, out: Path,
    *, epochs: int, learning_rate: float, batch_size: int, max_length: int,
    model_name: str, validation_split: float, seed: int,
) -> dict[str, Any]:
    """Train using the Crucible custom training loop (for .pt models)."""
    import torch

    # Use CUDA if available, otherwise CPU. MPS has incomplete op coverage
    # that causes internal assertion failures with some tensor operations.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    # Build token sequences from the text column
    texts = dataset["text"] if "text" in dataset.column_names else [str(r) for r in dataset]

    # Simple tokenization using the Crucible ChatTokenizer
    all_ids = []
    for text in texts:
        ids = tokenizer.encode(text, max_length)
        if len(ids) >= 2:  # Need at least 2 tokens for input/target
            all_ids.append(ids)

    # Train/val split
    import random
    rng = random.Random(seed)
    indices = list(range(len(all_ids)))
    rng.shuffle(indices)
    split_idx = max(1, int(len(indices) * (1 - validation_split)))
    train_ids = [all_ids[i] for i in indices[:split_idx]]
    val_ids = [all_ids[i] for i in indices[split_idx:]]

    # Build batches
    def _make_batches(id_list: list[list[int]], bs: int) -> list[tuple[Any, Any]]:
        batches = []
        for start in range(0, len(id_list), bs):
            chunk = id_list[start:start + bs]
            max_len = max(len(s) for s in chunk)
            padded = [s + [0] * (max_len - len(s)) for s in chunk]
            t = torch.tensor(padded, dtype=torch.long, device=device)
            batches.append((t[:, :-1], t[:, 1:]))
        return batches

    train_batches = _make_batches(train_ids, batch_size)
    val_batches = _make_batches(val_ids, batch_size) if val_ids else []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    print(f"Training ({method}) Crucible model — {epochs} epochs, {len(train_batches)} batches...", flush=True)
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for inputs, targets in train_batches:
            optimizer.zero_grad()
            logits = model(inputs)
            if hasattr(logits, "logits"):
                logits = logits.logits
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  Epoch {epoch}: NaN/Inf loss detected, skipping batch", flush=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / max(len(train_batches), 1)
        print(f"  Epoch {epoch}/{epochs} — loss: {avg:.4f}", flush=True)

    print("Training complete.", flush=True)
    return save_result(model, tokenizer, str(out), epochs, model_name=model_name)


def _create_trainer(
    method: str,
    model: Any,
    tokenizer: Any,
    split: Any,
    base_args: dict[str, Any],
    **kwargs: Any,
) -> Any:
    """Create the appropriate trl Trainer for the method."""
    import trl

    method = method.replace("-train", "").replace("-", "")

    if method in ("sft", "lora", "qlora", "domainadapt"):
        config = trl.SFTConfig(**base_args)
        peft_config = None
        # For lora/qlora, the model is already wrapped by load_model
        return trl.SFTTrainer(
            model=model, args=config,
            train_dataset=split["train"], eval_dataset=split["test"],
            processing_class=tokenizer, peft_config=peft_config,
        )

    if method == "dpo":
        args = {k: v for k, v in base_args.items() if k != "max_length"}
        args["max_length"] = base_args["max_length"]
        args["beta"] = kwargs.get("beta", 0.1)
        config = trl.DPOConfig(**args)
        return trl.DPOTrainer(
            model=model, args=config,
            train_dataset=split["train"], eval_dataset=split["test"],
            processing_class=tokenizer,
        )

    if method == "kto":
        args = {k: v for k, v in base_args.items() if k != "max_length"}
        args["max_length"] = base_args["max_length"]
        config = trl.KTOConfig(**args)
        return trl.KTOTrainer(
            model=model, args=config,
            train_dataset=split["train"], eval_dataset=split["test"],
            processing_class=tokenizer,
        )

    if method == "orpo":
        args = {k: v for k, v in base_args.items() if k != "max_length"}
        args["max_length"] = base_args["max_length"]
        args["beta"] = kwargs.get("beta", 0.1)
        config_cls = getattr(trl, "ORPOConfig", None) or trl.DPOConfig
        trainer_cls = getattr(trl, "ORPOTrainer", None) or trl.DPOTrainer
        config = config_cls(**args)
        return trainer_cls(
            model=model, args=config,
            train_dataset=split["train"], eval_dataset=split["test"],
            processing_class=tokenizer,
        )

    # Fallback: SFTTrainer for unknown methods
    config = trl.SFTConfig(**base_args)
    return trl.SFTTrainer(
        model=model, args=config,
        train_dataset=split["train"], eval_dataset=split["test"],
        processing_class=tokenizer,
    )


def save_result(
    model: Any,
    tokenizer: Any,
    output_dir: str,
    epochs: int = 0,
    model_name: str = "",
) -> dict[str, Any]:
    """Save a trained model and register it.

    Args:
        model: The trained model.
        tokenizer: The tokenizer.
        output_dir: Where to save.
        epochs: Number of epochs completed.

    Returns:
        Dict with model_path and metadata.
    """
    import torch
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Unwrap PEFT if needed
    save_model = model
    inner = getattr(model, "base_model", None)
    if inner is not None and hasattr(inner, "merge_and_unload"):
        save_model = model.merge_and_unload()

    # Save HF format
    hf_dir = out / "hf_model"
    if hasattr(save_model, "save_pretrained"):
        save_model.save_pretrained(str(hf_dir))
        tokenizer.save_pretrained(str(hf_dir))

    # Save model.pt for Crucible compatibility
    model_path = out / "model.pt"
    state = save_model.state_dict() if hasattr(save_model, "state_dict") else {}
    torch.save(state, str(model_path))

    # Save tokenizer alongside model.pt — use training_metadata for all types
    try:
        from serve.training_metadata import save_tokenizer_vocabulary
        save_tokenizer_vocabulary(out, tokenizer)
    except Exception:
        # Fallback for HF tokenizers
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(str(out))

    # Register in model registry
    try:
        from store.model_registry import ModelRegistry
        from core.config import CrucibleConfig
        config = CrucibleConfig.from_env()
        registry = ModelRegistry(config.data_root)
        name = model_name or out.name
        registry.register_model(name, str(model_path))
    except Exception:
        pass

    print(f"model_path={model_path}", flush=True)
    return {
        "model_path": str(model_path),
        "output_dir": str(out),
        "epochs_completed": epochs,
    }
