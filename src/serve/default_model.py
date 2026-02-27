"""Default PyTorch model architecture for language modeling.

This module defines a configurable transformer-style causal language model
used when no custom architecture is supplied by the user.
"""

from __future__ import annotations

import math
from typing import Any, cast

from core.types import TrainingOptions


def build_default_model(
    torch_module: Any,
    vocab_size: int,
    options: TrainingOptions,
) -> Any:
    """Build the default causal language model.

    Args:
        torch_module: Imported torch module.
        vocab_size: Vocabulary size.
        options: User-configured training options.

    Returns:
        torch.nn.Module default model.
    """
    model_class = _build_default_model_class(torch_module, vocab_size, options)
    return model_class()


def _build_default_model_class(
    torch_module: Any,
    vocab_size: int,
    options: TrainingOptions,
) -> type:
    """Create the default model class bound to runtime torch/options."""
    torch_nn = torch_module.nn
    module_base = cast(type, torch_module.nn.Module)

    functional = torch_module.nn.functional

    class ForgeMultiHeadAttention(module_base):  # type: ignore[misc,valid-type]
        """Multi-head attention using F.scaled_dot_product_attention."""

        def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
            super().__init__()
            self.nhead = nhead
            self.head_dim = d_model // nhead
            self.qkv_proj = torch_nn.Linear(d_model, 3 * d_model)
            self.out_proj = torch_nn.Linear(d_model, d_model)
            self.dropout = dropout

        def forward(self, x: Any, is_causal: bool = True) -> Any:
            batch, seq_len, _ = x.shape
            qkv = self.qkv_proj(x)
            qkv = qkv.reshape(batch, seq_len, 3, self.nhead, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            drop = self.dropout if self.training else 0.0
            attn_out = functional.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal, dropout_p=drop,
            )
            attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, -1)
            return self.out_proj(attn_out)

    class ForgeTransformerBlock(module_base):  # type: ignore[misc,valid-type]
        """Pre-norm transformer block with SDPA-based attention."""

        def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float) -> None:
            super().__init__()
            self.norm1 = torch_nn.LayerNorm(d_model)
            self.attn = ForgeMultiHeadAttention(d_model, nhead, dropout)
            self.norm2 = torch_nn.LayerNorm(d_model)
            self.ffn = torch_nn.Sequential(
                torch_nn.Linear(d_model, dim_ff),
                torch_nn.GELU(),
                torch_nn.Dropout(dropout),
                torch_nn.Linear(dim_ff, d_model),
                torch_nn.Dropout(dropout),
            )

        def forward(self, x: Any) -> Any:
            x = x + self.attn(self.norm1(x))
            x = x + self.ffn(self.norm2(x))
            return x

    class DefaultCausalModel(module_base):  # type: ignore[misc,valid-type]
        """Embedding + transformer blocks with SDPA attention."""

        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch_nn.Embedding(vocab_size, options.hidden_dim)
            if options.position_embedding_type == "learned":
                self.position_embedding = torch_nn.Embedding(
                    options.max_token_length,
                    options.hidden_dim,
                )
            else:
                self.register_buffer(
                    "sinusoidal_position_encoding",
                    _build_sinusoidal_position_encoding(
                        torch_module,
                        options.max_token_length,
                        options.hidden_dim,
                    ),
                    persistent=False,
                )
            self.blocks = torch_nn.ModuleList([
                ForgeTransformerBlock(
                    d_model=options.hidden_dim,
                    nhead=options.attention_heads,
                    dim_ff=options.mlp_hidden_dim,
                    dropout=options.dropout,
                )
                for _ in range(options.num_layers)
            ])
            self.final_norm = torch_nn.LayerNorm(options.hidden_dim)
            self.output = _build_projection_head(
                torch_module=torch_module,
                input_dim=options.hidden_dim,
                vocab_size=vocab_size,
                hidden_dim=options.mlp_hidden_dim,
                layer_count=options.mlp_layers,
                dropout=options.dropout,
            )

        def forward(self, inputs: Any) -> Any:
            embedded = self.embedding(inputs)
            positions = _position_ids(torch_module, inputs)
            positional = _resolve_position_embeddings(self, positions)
            hidden_states = embedded + positional
            for block in self.blocks:
                hidden_states = block(hidden_states)
            hidden_states = self.final_norm(hidden_states)
            logits = self.output(hidden_states)
            return logits

    return DefaultCausalModel


def _build_projection_head(
    torch_module: Any,
    input_dim: int,
    vocab_size: int,
    hidden_dim: int,
    layer_count: int,
    dropout: float,
) -> Any:
    """Build MLP projection from hidden states to vocabulary logits."""
    torch_nn = torch_module.nn
    if layer_count <= 1:
        return torch_nn.Linear(input_dim, vocab_size)
    layers: list[Any] = []
    current_dim = input_dim
    for _ in range(layer_count - 1):
        layers.append(torch_nn.Linear(current_dim, hidden_dim))
        layers.append(torch_nn.GELU())
        if dropout > 0:
            layers.append(torch_nn.Dropout(dropout))
        current_dim = hidden_dim
    layers.append(torch_nn.Linear(current_dim, vocab_size))
    return torch_nn.Sequential(*layers)


def _position_ids(torch_module: Any, inputs: Any) -> Any:
    """Create per-token position ids for a batch."""
    batch_size = int(inputs.shape[0])
    sequence_length = int(inputs.shape[1])
    base_positions = torch_module.arange(sequence_length, device=inputs.device)
    return base_positions.unsqueeze(0).expand(batch_size, sequence_length)


def _resolve_position_embeddings(model: Any, positions: Any) -> Any:
    """Resolve positional embeddings for learned or sinusoidal mode."""
    position_embedding = getattr(model, "position_embedding", None)
    if position_embedding is not None:
        return position_embedding(positions)
    sinusoidal = getattr(model, "sinusoidal_position_encoding")
    return sinusoidal.index_select(0, positions.reshape(-1)).reshape(
        positions.shape[0],
        positions.shape[1],
        sinusoidal.shape[1],
    )


def _build_sinusoidal_position_encoding(
    torch_module: Any,
    max_token_length: int,
    hidden_dim: int,
) -> Any:
    """Build static sinusoidal positional embeddings."""
    positions = torch_module.arange(max_token_length, dtype=torch_module.float32).unsqueeze(1)
    even_indexes = torch_module.arange(0, hidden_dim, 2, dtype=torch_module.float32)
    scale = -math.log(10000.0) / max(hidden_dim, 1)
    div_term = torch_module.exp(even_indexes * scale)
    encoding = torch_module.zeros((max_token_length, hidden_dim), dtype=torch_module.float32)
    encoding[:, 0::2] = torch_module.sin(positions * div_term)
    odd_width = int(encoding[:, 1::2].shape[1])
    if odd_width > 0:
        encoding[:, 1::2] = torch_module.cos(positions * div_term[:odd_width])
    return encoding
