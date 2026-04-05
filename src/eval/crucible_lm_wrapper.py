"""lm-evaluation-harness wrapper for Crucible .pt checkpoint models.

Bridges Crucible's model loading into the ``lm_eval.api.model.LM`` interface
so that ``simple_evaluate()`` can score Crucible checkpoints against any
lm-eval-harness task.  HuggingFace models should use lm-eval's own ``HFLM``
class instead.
"""

from __future__ import annotations

from typing import Any

from lm_eval.api.model import LM


class CrucibleLM(LM):
    """Wrap a Crucible .pt checkpoint for lm-eval-harness evaluation."""

    def __init__(self, model_path: str) -> None:
        super().__init__()
        from eval.benchmarks._model_loader import load_eval_model

        self._eval_model = load_eval_model(model_path)

    # ------------------------------------------------------------------
    # Properties expected by the harness
    # ------------------------------------------------------------------

    @property
    def device(self) -> Any:
        return self._eval_model.device

    # ------------------------------------------------------------------
    # LM interface
    # ------------------------------------------------------------------

    def loglikelihood(self, requests: list) -> list[tuple[float, bool]]:
        """Score each (context, continuation) pair.

        Returns (total_log_likelihood, is_greedy) per request.
        ``compute_completion_loss`` returns *average* cross-entropy so we
        multiply by the number of completion tokens to get the total
        log-likelihood that lm-eval-harness expects.
        """
        from eval.benchmarks._model_loader import (
            compute_completion_loss,
            compute_logits,
        )

        results: list[tuple[float, bool]] = []
        for request in requests:
            context, continuation = request.args
            tokenizer = self._eval_model.tokenizer
            max_len = self._eval_model.max_token_length

            # Token counts for converting average → total log-likelihood
            prompt_ids = tokenizer.encode(context, max_len)
            full_ids = tokenizer.encode(context + continuation, max_len)
            num_completion_tokens = len(full_ids) - len(prompt_ids)

            loss = compute_completion_loss(self._eval_model, context, continuation)
            ll = -loss * num_completion_tokens

            # Greedy check: is the first continuation token the argmax?
            logits = compute_logits(self._eval_model, context)
            cont_ids = tokenizer.encode(continuation, max_len)
            is_greedy = bool(
                cont_ids and int(logits.argmax().item()) == cont_ids[0]
            )

            results.append((ll, is_greedy))
        return results

    def loglikelihood_rolling(self, requests: list) -> list[float]:
        """Score full text sequences.

        Returns total log-likelihood per request.
        ``compute_sequence_loss`` returns *average* cross-entropy so we
        multiply by (num_tokens - 1) — the number of predicted positions.
        """
        from eval.benchmarks._model_loader import compute_sequence_loss

        results: list[float] = []
        for request in requests:
            (text,) = request.args
            tokenizer = self._eval_model.tokenizer
            max_len = self._eval_model.max_token_length

            ids = tokenizer.encode(text, max_len)
            num_predicted = max(len(ids) - 1, 0)

            loss = compute_sequence_loss(self._eval_model, text)
            results.append(-loss * num_predicted)
        return results

    def generate_until(self, requests: list) -> list[str]:
        """Generate text, stopping at any of the specified stop strings."""
        from eval.benchmarks._model_loader import generate_text

        results: list[str] = []
        for request in requests:
            context, until_dict = request.args
            max_tokens = (
                until_dict.get("max_gen_toks", 256)
                if isinstance(until_dict, dict)
                else 256
            )
            generated = generate_text(
                self._eval_model, context, max_new_tokens=max_tokens,
            )
            # Truncate at first occurrence of any stop string
            if isinstance(until_dict, dict):
                for stop in until_dict.get("until", []):
                    idx = generated.find(stop)
                    if idx >= 0:
                        generated = generated[:idx]
            results.append(generated)
        return results
