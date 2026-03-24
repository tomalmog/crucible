"""Linear probe runner: train linear classifiers on frozen activations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from core.linear_probe_types import LinearProbeOptions
from serve.activation_extractor import discover_transformer_layers
from serve.interp_activation_collector import collect_activations


def run_linear_probe(
    options: LinearProbeOptions, records: list[Any],
) -> dict[str, Any]:
    """Train linear probes on activations and return accuracy per layer."""
    model, tokenizer = _load_model_and_tokenizer(options)
    model.eval()

    all_layers = discover_transformer_layers(model)

    if options.layer_index == -2:
        layer_indices = list(range(len(all_layers)))
    elif options.layer_index >= 0:
        layer_indices = [options.layer_index]
    else:
        layer_indices = [len(all_layers) - 1]

    layer_results: list[dict[str, Any]] = []
    for layer_idx in layer_indices:
        target_layer = all_layers[layer_idx]
        X, labels, _ = collect_activations(
            model, tokenizer, records, target_layer,
            options.max_samples, granularity="sample",
            label_field=options.label_field,
        )
        X = X.to(dtype=torch.float32)
        result = _probe_layer(
            X, labels, layer_idx, target_layer,
            options.epochs, options.learning_rate,
        )
        layer_results.append(result)

    output: dict[str, Any] = {"layers": layer_results}

    out_dir = Path(options.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "linear_probe.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    return output


def _probe_layer(
    X: Tensor, labels: list[str],
    layer_idx: int, layer_name: str,
    epochs: int, lr: float,
) -> dict[str, Any]:
    """Train a single linear probe and return results."""
    class_names = sorted(set(labels))
    if len(class_names) < 2:
        return {
            "layer_index": layer_idx,
            "layer_name": layer_name,
            "accuracy": 0.0,
            "num_classes": len(class_names),
            "class_names": class_names,
            "confusion_matrix": [],
            "error": "Need at least 2 classes for probing",
        }

    label_to_idx = {name: i for i, name in enumerate(class_names)}
    y = torch.tensor([label_to_idx[l] for l in labels], dtype=torch.long)
    num_classes = len(class_names)

    # 80/20 train/val split
    n = len(y)
    perm = torch.randperm(n)
    split = max(1, int(n * 0.8))
    train_idx, val_idx = perm[:split], perm[split:]

    if len(val_idx) == 0:
        val_idx = train_idx[-1:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    accuracy, confusion = _train_probe(
        X_train, y_train, X_val, y_val, num_classes, epochs, lr,
    )

    return {
        "layer_index": layer_idx,
        "layer_name": layer_name,
        "accuracy": round(accuracy, 4),
        "num_classes": num_classes,
        "class_names": class_names,
        "confusion_matrix": confusion,
    }


def _train_probe(
    X_train: Tensor, y_train: Tensor,
    X_val: Tensor, y_val: Tensor,
    num_classes: int, epochs: int, lr: float,
) -> tuple[float, list[list[int]]]:
    """Train a linear probe and return (accuracy, confusion_matrix)."""
    hidden_dim = X_train.shape[1]
    probe = nn.Linear(hidden_dim, num_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        probe.train()
        optimizer.zero_grad()
        logits = probe(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        val_logits = probe(X_val)
        preds = val_logits.argmax(dim=1)
        accuracy = float((preds == y_val).float().mean())

    # Confusion matrix
    confusion = [[0] * num_classes for _ in range(num_classes)]
    for true, pred in zip(y_val.tolist(), preds.tolist()):
        confusion[true][pred] += 1

    return accuracy, confusion


def _load_model_and_tokenizer(options: LinearProbeOptions) -> tuple[Any, Any]:
    """Load model + tokenizer for probe analysis."""
    from serve.interp_model_loader import load_interp_model
    model_path = options.base_model or options.model_path
    return load_interp_model(model_path)
