"""HuggingFace tokenizer wrappers for Crucible.

Wraps both the ``tokenizers`` library and ``transformers.AutoTokenizer``
to provide the same encode/decode/vocabulary interface (ChatTokenizer
protocol) that Crucible training and chat runners expect.
"""

from __future__ import annotations

from typing import Any

from core.errors import CrucibleDependencyError, CrucibleServeError


class HuggingFaceTokenizer:
    """Wrapper around a HuggingFace ``tokenizers.Tokenizer`` instance.

    Exposes the same ``encode``, ``decode``, and ``vocabulary`` interface
    used by Crucible chat runners so it can substitute for VocabularyTokenizer.
    """

    def __init__(self, tokenizer_instance: Any) -> None:
        self._tokenizer: Any = tokenizer_instance
        self.vocabulary: dict[str, int] = dict(tokenizer_instance.get_vocab())

    def encode(self, text: str, max_token_length: int) -> list[int]:
        """Encode text to token ids using the HuggingFace tokenizer.

        Args:
            text: Input text.
            max_token_length: Maximum token count.

        Returns:
            Encoded token ids, truncated to max_token_length.
        """
        encoding = self._tokenizer.encode(text)
        ids: list[int] = list(encoding.ids)
        if len(ids) > max_token_length:
            return ids[:max_token_length]
        return ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids back into text.

        Args:
            token_ids: Token ids to decode.

        Returns:
            Decoded text string.
        """
        return str(self._tokenizer.decode(token_ids))


class AutoTokenizerAdapter:
    """Wraps a ``transformers.AutoTokenizer`` to satisfy ChatTokenizer.

    Unlike ``HuggingFaceTokenizer`` (which wraps the low-level ``tokenizers``
    library), this wraps the full ``transformers`` tokenizer — preserving BPE
    subword encoding, special-token handling, and proper decode.
    """

    def __init__(self, hf_tokenizer: Any) -> None:
        self._tokenizer: Any = hf_tokenizer
        self.vocabulary: dict[str, int] = dict(hf_tokenizer.get_vocab())

    def encode(self, text: str, max_token_length: int) -> list[int]:
        ids: list[int] = self._tokenizer.encode(text)
        if len(ids) > max_token_length:
            return ids[:max_token_length]
        return ids

    def decode(self, token_ids: list[int]) -> str:
        return str(self._tokenizer.decode(token_ids))


def load_huggingface_tokenizer(tokenizer_path: str) -> HuggingFaceTokenizer:
    """Load a HuggingFace tokenizer from a tokenizer.json file.

    Args:
        tokenizer_path: Path to a HuggingFace tokenizer.json file.

    Returns:
        Wrapped tokenizer instance.

    Raises:
        CrucibleDependencyError: If the ``tokenizers`` library is not installed.
        CrucibleServeError: If the tokenizer file cannot be loaded.
    """
    tokenizers_module = _import_tokenizers()
    try:
        instance = tokenizers_module.Tokenizer.from_file(tokenizer_path)
    except Exception as error:
        raise CrucibleServeError(
            f"Failed to load HuggingFace tokenizer from {tokenizer_path}: {error}. "
            "Verify the file is a valid tokenizer.json."
        ) from error
    return HuggingFaceTokenizer(instance)


def _import_tokenizers() -> Any:
    """Import the tokenizers library with a clear error on failure."""
    try:
        import tokenizers
    except ImportError as error:
        raise CrucibleDependencyError(
            "HuggingFace tokenizer support requires the tokenizers library. "
            "Install with: pip install tokenizers"
        ) from error
    return tokenizers
