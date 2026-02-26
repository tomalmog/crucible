# CLAUDE.md — Lessons & Guardrails

> Hard-won rules from real mistakes in this codebase.
> Every item here exists because ignoring it caused a bug, wasted time, or produced bad code.

---

## Defaults Must Stay in Sync

When a value has a default in Python AND in the UI, both must be updated together.
The CLI and the Studio UI are two frontends to the same backend — they must agree.

- Python defaults live in `src/core/constants.py` and the relevant `*_types.py` dataclass.
- UI defaults live in `studio-app/src/types/training.ts` (`getDefaultConfigForMethod`).
- **If you change a default in one place, grep for the old value and update every occurrence.**
- Fine-tuning methods (DPO, LoRA, RLHF, SFT, distillation) need much lower learning rates
  than training from scratch. Never use the generic `DEFAULT_TRAIN_LEARNING_RATE` for them.

---

## Don't Reinvent — Use Libraries

Before writing any non-trivial utility, check if a well-maintained library already does it.
Writing your own version means more code to maintain, more bugs, and worse results.

**Real examples of this mistake:**
- Writing a custom markdown renderer instead of using `react-markdown` + `remark-gfm`.
- Building custom progress bar components when the design system already has one.

**The rule:** If a mature, small-dependency library exists for the task, use it.
Only write it yourself if: (a) the library pulls in a massive dependency tree,
(b) you need <20 lines of code, or (c) the library doesn't fit the exact use case.

---

## Autograd Graph Must Be Preserved

Any tensor operation in a training loop that will flow into `loss.backward()` must be
differentiable. PyTorch silently breaks the gradient graph in several common patterns:

- **Element-wise Python loops** assigning into `torch.zeros()` — use `torch.gather()` instead.
- **In-place operations** on leaf tensors without `requires_grad` — the grad_fn is lost.
- **Detaching tensors** accidentally (`.item()`, `.detach()`, `.numpy()` mid-graph).

If you write training code, verify: does `loss.requires_grad` return `True` before calling `.backward()`?

---

## Training Loops Need Safety Rails

Every training loop must have:

1. **Gradient clipping** — `clip_grad_norm_(model.parameters(), max_norm=1.0)` between
   `.backward()` and `.step()`. Without this, a single bad batch can cascade to NaN.
2. **NaN detection** — check `math.isnan(loss.item())` after each batch and raise a
   clear error before the model is silently corrupted.
3. **Flushed structured output** — use `emit_progress()` from `serve/training_progress.py`
   with `flush=True`, not raw `print()`. This ensures real-time streaming to the UI.

---

## Test What You Change

When modifying infrastructure (like switching from `_LOGGER` to `emit_progress()`),
grep for tests that mock/monkeypatch the old interface and update them.
A passing test suite with a broken test is worse than a failing one — it gives false confidence.

---

## Frontend-Backend Contract

The Studio UI calls the Python CLI via Tauri subprocess. The contract is:

- **CLI flags** are the API. If you add/rename/remove a CLI flag, update `commandArgs.ts`.
- **stdout** is the data channel. Training progress is JSON-per-line on stdout, parsed by
  `parseTrainingProgress()` in `TrainingRunMonitor.tsx`.
- **stderr** is for logs and errors. The UI parses `ForgeXxxError:` patterns for friendly messages.

Never change stdout format without updating the frontend parser.

---

## Code Style Reminders

- Follow AGENTS.md strictly — it's the law of this codebase.
- Max 300 lines/file, 50 lines/function, 4 params/function.
- Use existing design system classes (`panel`, `progress-bar`, `badge`, `console`, etc.)
  before adding inline styles.
- Prefer editing existing files over creating new ones.
- Search the codebase before writing anything new.
