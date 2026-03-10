"""Reward model training from human preference data.

This module trains a reward model from JSONL preference data (prompt/chosen/rejected)
so that the trained model assigns higher scores to chosen responses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.errors import CrucibleRlhfError
from core.rlhf_types import RlhfOptions
from core.types import DataRecord
from serve.reward_model import build_reward_model_from_base
from serve.training_setup import fit_training_tokenizer


def train_reward_model(
    torch_module: Any,
    base_model: Any,
    options: RlhfOptions,
    device: Any,
) -> Any:
    """Train a reward model from preference data.

    Loads preference pairs and trains a classifier so that
    reward(prompt+chosen) > reward(prompt+rejected).

    Args:
        torch_module: Imported torch module.
        base_model: Base language model for feature extraction.
        options: RLHF training options with reward config.
        device: Torch device for training.

    Returns:
        Trained reward model.

    Raises:
        CrucibleRlhfError: If preference data is missing or invalid.
    """
    preference_path = options.reward_config.preference_data_path
    if preference_path is None:
        raise CrucibleRlhfError(
            "Reward model training requires --preference-data-path. "
            "Provide a JSONL file with prompt/chosen/rejected triples."
        )
    pairs = _load_preference_pairs(preference_path)
    reward_model = build_reward_model_from_base(
        torch_module, base_model, options.hidden_dim,
    )
    reward_model = reward_model.to(device)
    _run_reward_training_loop(
        torch_module, reward_model, pairs, options, device,
    )
    return reward_model


def _load_preference_pairs(data_path: str) -> list[dict[str, str]]:
    """Load preference pairs from JSONL file.

    Args:
        data_path: Path to JSONL file.

    Returns:
        List of preference pair dicts with prompt/chosen/rejected.

    Raises:
        CrucibleRlhfError: If file is missing or contains invalid rows.
    """
    resolved_path = Path(data_path).expanduser().resolve()
    if not resolved_path.exists():
        raise CrucibleRlhfError(
            f"Preference data file not found at {resolved_path}. "
            "Provide a valid --preference-data-path."
        )
    try:
        lines = resolved_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise CrucibleRlhfError(
            f"Failed to read preference data at {resolved_path}: {error}."
        ) from error
    pairs: list[dict[str, str]] = []
    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        pair = _parse_preference_line(stripped, line_num, resolved_path)
        pairs.append(pair)
    if not pairs:
        raise CrucibleRlhfError(
            f"Preference data at {resolved_path} contains no valid pairs."
        )
    return pairs


def _parse_preference_line(
    line: str, line_num: int, file_path: Path,
) -> dict[str, str]:
    """Parse and validate one JSONL preference line."""
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError as error:
        raise CrucibleRlhfError(
            f"Preference data line {line_num} in {file_path}: "
            f"invalid JSON ({error.msg})."
        ) from error
    if not isinstance(parsed, dict):
        raise CrucibleRlhfError(
            f"Preference data line {line_num}: expected JSON object."
        )
    for field in ("prompt", "chosen", "rejected"):
        value = parsed.get(field)
        if not isinstance(value, str) or not value.strip():
            raise CrucibleRlhfError(
                f"Preference data line {line_num}: "
                f"missing or empty '{field}' field."
            )
    return {
        "prompt": str(parsed["prompt"]).strip(),
        "chosen": str(parsed["chosen"]).strip(),
        "rejected": str(parsed["rejected"]).strip(),
    }


def _run_reward_training_loop(
    torch_module: Any,
    reward_model: Any,
    pairs: list[dict[str, str]],
    options: RlhfOptions,
    device: Any,
) -> None:
    """Run the reward model training loop.

    Trains the reward model so chosen responses score higher than rejected.

    Args:
        torch_module: Imported torch module.
        reward_model: Reward model to train.
        pairs: Preference pair dicts.
        options: RLHF training options.
        device: Torch device.
    """
    reward_model.train()
    optimizer = torch_module.optim.Adam(
        reward_model.parameters(), lr=options.learning_rate,
    )
    loss_fn = torch_module.nn.MarginRankingLoss(margin=1.0)
    batch_size = options.batch_size
    for epoch in range(options.epochs):
        epoch_loss = 0.0
        batch_count = 0
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            chosen_scores, rejected_scores = _score_batch(
                torch_module, reward_model, batch, device,
            )
            target = torch_module.ones(
                chosen_scores.size(0), device=device,
            )
            loss = loss_fn(chosen_scores, rejected_scores, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        avg_loss = epoch_loss / max(batch_count, 1)
        print(f"Reward model epoch {epoch + 1}/{options.epochs} loss={avg_loss:.4f}")
    reward_model.eval()


def _score_batch(
    torch_module: Any,
    reward_model: Any,
    batch: list[dict[str, str]],
    device: Any,
) -> tuple[Any, Any]:
    """Score chosen and rejected sequences for a batch.

    Args:
        torch_module: Imported torch module.
        reward_model: Reward model producing scalar scores.
        batch: List of preference pair dicts.
        device: Torch device.

    Returns:
        Tuple of (chosen_scores, rejected_scores) tensors.
    """
    chosen_ids = _encode_texts(
        torch_module, [p["prompt"] + " " + p["chosen"] for p in batch], device,
    )
    rejected_ids = _encode_texts(
        torch_module, [p["prompt"] + " " + p["rejected"] for p in batch], device,
    )
    chosen_scores = reward_model(chosen_ids)
    rejected_scores = reward_model(rejected_ids)
    return chosen_scores, rejected_scores


def _encode_texts(
    torch_module: Any,
    texts: list[str],
    device: Any,
) -> Any:
    """Simple character-level encoding for reward model training.

    Args:
        torch_module: Imported torch module.
        texts: List of text strings to encode.
        device: Torch device.

    Returns:
        Tensor of token ids [batch, max_len].
    """
    max_len = 64
    batch_ids = []
    for text in texts:
        ids = [ord(c) % 256 for c in text[:max_len]]
        ids = ids + [0] * (max_len - len(ids))
        batch_ids.append(ids)
    return torch_module.tensor(batch_ids, dtype=torch_module.long, device=device)
