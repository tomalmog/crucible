"""Shared activation collection for interpretability tools."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from serve.activation_extractor import ActivationExtractor
from serve.interp_data_utils import extract_texts, get_label


def collect_activations(
    model: Any,
    tokenizer: Any,
    records: list[Any],
    target_layer: str,
    max_samples: int,
    granularity: str = "sample",
    label_field: str = "",
) -> tuple[Tensor, list[str], list[str]]:
    """Extract activations for records at a target layer.

    Returns (stacked_vectors, labels, text_snippets).
    """
    texts = extract_texts(records, max_samples)
    if not texts:
        from core.errors import CrucibleError
        raise CrucibleError(
            f"No usable text found in {len(records)} dataset records. "
            "Ensure the dataset has 'text', 'prompt', 'content', or 'input' fields."
        )

    device = next(model.parameters()).device
    vectors: list[Tensor] = []
    labels: list[str] = []
    snippets: list[str] = []

    with torch.no_grad():
        with ActivationExtractor(model, [target_layer]) as extractor:
            for i, text in enumerate(texts):
                extractor.clear()
                ids = tokenizer.encode(
                    text, return_tensors="pt", truncation=True, max_length=512,
                )
                ids = ids.to(device)
                model(ids)
                acts = extractor.get_activations()[target_layer]

                color = get_label(records, i, label_field)

                if granularity == "token":
                    token_strs = tokenizer.convert_ids_to_tokens(ids[0].tolist())
                    for t in range(acts.shape[1]):
                        vectors.append(acts[0, t].cpu())
                        labels.append(color)
                        snippets.append(
                            token_strs[t] if t < len(token_strs) else "",
                        )
                else:
                    pooled = acts[0].mean(dim=0).cpu()
                    vectors.append(pooled)
                    labels.append(color)
                    snippet = text[:120].rstrip()
                    if len(text) > 120:
                        snippet += "..."
                    snippets.append(snippet)

    stacked = torch.stack(vectors).float()
    return stacked, labels, snippets
