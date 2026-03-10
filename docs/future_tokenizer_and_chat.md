# Future Considerations: Tokenization and Chat Inference

## Training Tokenizer Improvements

Crucible currently uses a whitespace-splitting tokenizer (`VocabularyTokenizer`) for training. This works for proof-of-concept models but limits quality for real workloads.

### BPE / SentencePiece for Training

- **What**: Replace or supplement the whitespace tokenizer with byte-pair encoding (BPE) or SentencePiece during the training pipeline.
- **Why**: Subword tokenization handles unseen words, morphology, punctuation, and multilingual text far better. Models trained with BPE produce more coherent output and have smaller effective vocabulary sizes.
- **Approach**: The `ChatTokenizer` protocol added in the current work makes this straightforward — any new tokenizer just needs to satisfy `encode`, `decode`, and `vocabulary`. The training pipeline would need to save the trained tokenizer alongside the model (BPE merge rules, not just the vocabulary).
- **Consideration**: BPE training itself is a non-trivial step. The `tokenizers` library (already an optional dependency) can train BPE/WordPiece/Unigram tokenizers from text data efficiently.

### Configurable Tokenization Method

- **What**: Let users choose the tokenization method when training (`--tokenizer-type whitespace|bpe|sentencepiece`).
- **Why**: Different use cases benefit from different tokenizers. Character-level models, word-level models, and subword models all have trade-offs in vocabulary size, sequence length, and generalization.
- **Consideration**: This should be a training-time setting persisted in the training config so chat/inference automatically uses the matching tokenizer.

## Chat Inference Improvements

### Broader External Model Support

- **What**: The ONNX chat runner now handles `attention_mask` and `position_ids` inputs dynamically. Further inputs may be needed for other model architectures (e.g., `token_type_ids` for BERT-style models, `past_key_values` for cached generation).
- **Approach**: The `_build_input_feed` function in `onnx_chat_runner.py` can be extended per-input-name as new model types are encountered.

### KV-Cache / Past Key Values for Faster Generation

- **What**: Autoregressive generation currently recomputes the full sequence on every token. Models that support KV-caching export past key/value tensors as outputs and accept them as inputs on subsequent steps.
- **Why**: Dramatically faster generation — O(n) per token instead of O(n^2) for the full sequence.
- **Consideration**: This requires detecting KV-cache outputs in the ONNX graph and threading them through the generation loop.

### PyTorch Chat with External Models

- **What**: The PyTorch chat runner (`chat_runner.py`) currently only loads Crucible-trained models. Supporting HuggingFace `.pt`/`.bin` checkpoints would broaden usability.
- **Why**: Users may want to fine-tune or evaluate models from the HuggingFace ecosystem.
- **Consideration**: Architecture detection and weight mapping are the main challenges. The `architecture_loader.py` module would need extension.

### Streaming Token Output

- **What**: Currently the full response is generated before being returned. Streaming tokens as they're generated would improve the Studio chat UX.
- **Why**: Users see output immediately rather than waiting for the full sequence to complete.
- **Approach**: The generation loop in both chat runners could yield tokens incrementally. The Studio app would need to poll for partial output or use a streaming protocol.

## Dependency Notes

- The `tokenizers` library is now an optional dependency (`pip install -e .[tokenizers]`). It is only imported when a HuggingFace tokenizer.json is detected — Crucible-trained models work without it.
- If BPE training support is added to the training pipeline, `tokenizers` may become a core dependency or part of the `serve` extra.
