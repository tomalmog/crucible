# AGENTS.md — Crucible Codebase Standards

> This file governs all AI-assisted code generation in this repository.
> Every agent (Copilot, Claude, Cursor, etc.) MUST follow these rules.
> If a rule conflicts with "getting it done fast," the rule wins.

---

## Core Philosophy

This codebase will be read by humans, debugged by humans, and maintained for years. Write code that a new engineer can understand on their first day. Clever code is bad code. Clear code is fast code.

**The Three Questions — ask before writing anything:**
1. Does this function/module/component already exist somewhere in the codebase? (Search first.)
2. Will a developer understand this in 6 months without context?
3. If this breaks, can someone trace the error to the root cause in under 5 minutes?

If the answer to any of these is "no," rewrite it.

---

## Tech Stack

### Python (CLI / SDK / Training)
- **Python:** >=3.11
- **Type checking:** `mypy --strict`
- **Linting:** `ruff` (line-length 100)
- **Testing:** `pytest` (80% minimum coverage, 95% target)
- **ML:** PyTorch 2.6, Lance 1.2 (vector storage), PyArrow
- **Types:** Frozen dataclasses everywhere — no raw dicts for domain objects

### Frontend (Studio Desktop App)
- **Desktop:** Tauri 2 (Rust backend, WebView frontend)
- **Framework:** React 19, TypeScript strict mode
- **Routing:** react-router 7
- **Build:** Vite 7
- **Styling:** Plain CSS with design tokens (CSS custom properties) — no Tailwind, no CSS-in-JS
- **Icons:** lucide-react
- **Markdown:** react-markdown + remark-gfm
- **State:** React hooks + Context API (no external state library)

### Rust (Tauri Commands)
- Shuttle JSON between TypeScript and disk. No business logic in Rust.
- Use `serde_json::Value` for shapes owned by TypeScript.
- No `.unwrap()` in production — use `?` or explicit error handling.

---

## Project Structure

```
crucible/
├── src/                        # Python — CLI, SDK, core library
│   ├── core/                   #   Shared types, errors, constants, config
│   ├── cli/                    #   CLI commands (one file per command)
│   ├── serve/                  #   Training runners, losses, strategies, evaluation, remote job submission
│   ├── store/                  #   Dataset SDK, model registry, persistence
│   ├── ingest/                 #   Data ingestion pipeline
│   ├── transforms/             #   Dedup, quality scoring, language detection
│   ├── eval/                   #   Evaluation runners
│   ├── deploy/                 #   Deployment logic
│   ├── safety/                 #   Safety evaluation and gating
│   └── crucible.py              #   CrucibleClient SDK entry point
├── studio-app/                 # Frontend — Tauri desktop app
│   ├── src/
│   │   ├── api/                #   Tauri invoke wrappers + CLI arg builders
│   │   ├── components/         #   Reusable UI (shared/, sidebar/)
│   │   ├── context/            #   React Context providers
│   │   ├── hooks/              #   Custom hooks (one hook per file)
│   │   ├── pages/              #   Page components organized by domain
│   │   ├── theme/              #   CSS design system (variables, components, layout)
│   │   ├── types/              #   Shared TypeScript type definitions
│   │   ├── App.tsx             #   Root component
│   │   ├── router.tsx          #   Route definitions
│   │   └── main.tsx            #   Entry point
│   └── src-tauri/
│       └── src/
│           ├── commands/       #   Rust Tauri command handlers
│           ├── models.rs       #   Rust data models for Tauri
│           └── lib.rs          #   Command registration
├── tests/                      # Python tests
│   ├── unit/                   #   Mirror src/ structure exactly
│   ├── integration/            #   Cross-module tests
│   └── fixtures/               #   Shared test data — NEVER generate inline
├── docs/                       # Architecture docs, training method reference
└── benchmarks/                 # Performance regression tests
```

**Rules:**
- Every Python directory has an `__init__.py` with a module docstring.
- Every Rust directory has a `mod.rs` that documents what the module does.
- No file exceeds **300 lines**. If it does, split it.
- No function exceeds **50 lines**. Extract helpers.
- No function has more than **4 parameters**. Use a config/options object.
- No component/hook exceeds **80 lines of logic** (excluding JSX/types).

---

## The DRY Contract

**Before writing ANY new function, hook, component, or type:**

1. Search the codebase for similar functionality.
2. Check `src/core/` (Python) or `components/shared/` (TypeScript) for shared code.
3. If something similar exists, extend or refactor it. Do NOT create a parallel version.

**If you find yourself writing any of these, STOP:**
- A second config parser or path resolver
- A second progress emitter or logging setup
- A second Tauri invoke wrapper for the same command
- A second CSS class for the same visual pattern
- Any file named `utils.ts`, `helpers.ts`, `misc.ts`, `common.ts`, `utils.py`, `helpers.py`
- Any function prefixed with `custom_`, `my_`, `new_`, or `v2_`

**Shared Python code lives in `src/core/`. Shared UI code lives in `components/shared/`. Not in the module that happened to need it first.**

---

## Naming Conventions

### Files
- **Python:** `snake_case.py` — name describes what it does.
  - Good: `dpo_loss.py`, `training_checkpoint.py`, `dataset_sdk.py`
  - Bad: `utils.py`, `helpers.py`, `misc.py`
- **TypeScript components:** `PascalCase.tsx` — name describes what it renders.
  - Good: `TrainingWizard.tsx`, `MetricCard.tsx`, `PageHeader.tsx`
  - Bad: `Card.tsx`, `Component.tsx`, `View.tsx`
- **TypeScript hooks:** `useCamelCase.ts` — name describes what state/behavior it provides.
  - Good: `useTrainingConfig.ts`, `useCrucibleCommand.ts`
  - Bad: `useData.ts`, `useStuff.ts`
- **TypeScript types:** `camelCase.ts` in `types/` for shared types.
  - Good: `training.ts`, `deploy.ts`, `models.ts`
- **CSS:** semantic class names in `theme/components.css`. No BEM, just descriptive.
  - Good: `.metric-card`, `.wizard-step`, `.console-tall`

### Code
- **Python functions:** `verb_noun()` in `snake_case`.
  - Good: `compute_quality_score()`, `emit_progress()`, `build_training_args()`
  - Banned: `process()`, `handle()`, `do_thing()`, `run()`, `manage()`
- **TypeScript functions:** `verbNoun()` in `camelCase`.
  - Good: `buildTrainingArgs()`, `parseTrainingProgress()`, `loadTrainingConfig()`
- **React components:** `PascalCase` noun describing the UI element.
- **Constants:** `UPPER_SNAKE_CASE`, defined in `src/core/constants.py` (Python) or `types/*.ts` (TypeScript). Never hardcode inline.
- **Booleans:** prefix with `is`, `has`, `can`, `should`.
- **No single-letter variables** outside of array methods and loop counters.
- **No abbreviations** unless universally understood: `gpu`, `cpu`, `url`, `pii`, `lr`, `ws`.

---

## TypeScript Rules

**All code is strictly typed. No shortcuts.**

- `strict: true` is enforced in `tsconfig.json`. It stays that way.
- No `any`. No `as any`. No `// @ts-ignore`. No `// @ts-expect-error` without a linked issue.
- Use `interface` for object shapes. Use `type` for unions and aliases.
- Use `Record<K, V>` over `{ [key: string]: V }`.
- No type assertions (`as Type`) unless you can prove the cast is safe with a comment.
- No enums. Use string literal union types or `as const` objects.
- Every function has explicit parameter types and return types.

---

## React & Component Rules

### Component Structure
Every component follows this order:
1. Imports
2. Types/interfaces (if component-specific)
3. Constants (if component-specific)
4. Component function (named export, no default export — only `main.tsx` uses default)

### Hooks
- Custom hooks extract ALL non-trivial logic from components.
- A component's job is: call hooks, derive display values, return JSX. That's it.
- Every `useEffect` must have a comment explaining WHAT it reacts to and WHY.
- Every `useEffect` cleanup function must be complete — no leaked intervals, subscriptions, or timers.
- `useRef` for mutable values that don't trigger re-renders. `useState` for values that do.
- Never read `.current` of a ref inside JSX — it won't re-render when it changes.

### State
- Keep state as close to where it's used as possible.
- Lift state only when two or more siblings need it.
- Context is for app-wide state (`CrucibleContext`, `CommandContext`). Not for passing props two levels down.
- Never store derived state. Compute it from source state in render or `useMemo`.

### Memoization
- `useMemo` for expensive computations or reference-stable objects passed as deps.
- `useCallback` for functions passed as props or used in dependency arrays.
- Do NOT memoize trivially cheap operations. The overhead is worse than recomputing.

---

## Styling Rules (CSS Design Tokens)

The design system lives in `studio-app/src/theme/`:
- `variables.css` — color palette, fonts, spacing, transitions as CSS custom properties
- `components.css` — pre-built component classes
- `layout.css` — app shell grid, sidebar
- `reset.css` — browser normalization

**Rules:**
- Use the existing design system classes before creating new ones. The vocabulary includes:
  `panel`, `btn`, `btn-primary`, `btn-ghost`, `btn-sm`, `btn-lg`, `badge`, `console`,
  `metric-card`, `stats-grid`, `progress-bar`, `tab-list`, `tab-item`, `form-section`,
  `empty-state`, `action-group`, `wizard-header`, `wizard-step`, `path-input`, `error-text`.
- Use semantic color tokens: `var(--text)`, `var(--text-secondary)`, `var(--bg-surface)`,
  `var(--border)`, `var(--error)`, `var(--success)`. Never hardcode hex colors.
- Inline `style` objects are acceptable ONLY for dynamic values (e.g., computed widths).
  For static styling, add a class to `components.css`.
- New classes go in `components.css`, not in component files.

---

## Python Rules

### Type Safety
- `mypy --strict` must pass. No `Any`. No untyped function signatures.
- Use frozen `@dataclass` for all domain objects. No raw dicts for structured data.
- Every function has complete type annotations (params and return).

### Error Handling
- Custom exceptions live in `src/core/errors.py`, grouped by domain:
  `CrucibleIngestError`, `CrucibleStoreError`, `CrucibleTransformError`, etc.
- Never catch bare `Exception` unless you re-raise with context.
- Never silently swallow errors. Every `except` block must log or re-raise.
- Error messages must include: what happened, what input caused it, what the user should do.

### Configuration
- All config flows through `CrucibleConfig` in `src/core/config.py`.
- Environment variables are read in ONE place. No `os.getenv()` scattered through business logic.
- Every config value has a default, a type, and a docstring.
- Secrets are never logged, never in error messages, never in stack traces.

### Logging
- Use structured logging (`structlog`). No f-string print statements for debugging.
- Training progress goes through `emit_progress()` from `serve/training_progress.py` with `flush=True`.
- Log levels mean something: DEBUG for internal state, INFO for operations completing, WARNING for recoverable issues, ERROR for failures.

### PyTorch / Training Code

**Autograd graph must be preserved.** Any tensor operation that flows into `loss.backward()` must be differentiable. PyTorch silently breaks the gradient graph in common patterns:
- Element-wise Python loops assigning into `torch.zeros()` — use `torch.gather()` instead.
- In-place operations on leaf tensors without `requires_grad` — the grad_fn is lost.
- Detaching tensors accidentally (`.item()`, `.detach()`, `.numpy()` mid-graph).
- If you write training code, verify: does `loss.requires_grad` return `True` before `.backward()`?

**Every training loop must have:**
1. **Gradient clipping** — `clip_grad_norm_(model.parameters(), max_norm=1.0)` between `.backward()` and `.step()`. Without this, a single bad batch cascades to NaN.
2. **NaN detection** — check `math.isnan(loss.item())` after each batch and raise a clear error before the model is silently corrupted.
3. **Flushed structured output** — use `emit_progress()` with `flush=True`, not raw `print()`.

---

## Frontend-Backend Contract

The Studio UI calls the Python CLI via Tauri subprocess. The contract is:

- **CLI flags are the API.** If you add/rename/remove a CLI flag, update `commandArgs.ts`.
- **stdout is the data channel.** Training progress is JSON-per-line on stdout, parsed by `parseTrainingProgress()` in `TrainingRunMonitor.tsx`.
- **stderr is for logs and errors.** The UI parses `CrucibleXxxError:` patterns for friendly messages.
- **Defaults must stay in sync.** Python defaults (`constants.py`, `*_types.py`) and UI defaults (`training.ts`, `getDefaultConfigForMethod`) must agree. If you change one, grep for the old value and update every occurrence. Fine-tuning methods (DPO, LoRA, RLHF, SFT, distillation) need much lower learning rates than training from scratch — never use the generic `DEFAULT_TRAIN_LEARNING_RATE` for them.

Never change stdout format without updating the frontend parser.

---

## Performance & Scalability

- No O(n^2) algorithms on datasets. If you're nesting loops over data, find a better approach.
- Streaming over loading: never load an entire dataset into memory if you can process it as a stream.
- Batch I/O operations. Never make N network calls when 1 batch call works.
- Cap all arrays with `.slice()` to prevent unbounded growth.
- `useMemo` for expensive computations (chart paths, filtered lists). But measure before optimizing — if you add a `useMemo`, explain what re-render it prevents.
- Interval timers use the coarsest frequency that provides acceptable UX.

---

## Testing Requirements

**No PR merges without tests for new functionality.**

### Coverage Rules
- Every public function has at least one unit test.
- Every error path has a test that triggers it.
- Bug fixes include a regression test BEFORE the fix (red-green-refactor).

### Test Structure
```python
def test_<what>_<condition>_<expected>():
    """One sentence: what this test verifies."""
    # Arrange — set up inputs
    # Act — call the function
    # Assert — check ONE thing
```

- Test names describe behavior, not implementation.
- One assert per test. Multiple asserts = multiple tests.
- No test depends on another test. Every test runs in isolation.
- Test data lives in `tests/fixtures/`. Never generate fake data inline.
- Mock external services. Never hit real infrastructure in unit tests.
- **Test what you change.** When modifying infrastructure (e.g., switching from `_LOGGER` to `emit_progress()`), grep for tests that mock the old interface and update them. A passing test suite with a broken test is worse than a failing one.

---

## Anti-Patterns — Instant Rejection

| Pattern | Why It's Bad | Do This Instead |
|---|---|---|
| `any` type / `Any` type | Defeats TypeScript/mypy entirely | Use proper types or `unknown` + narrowing |
| `as Type` without justification | Masks bugs | Narrow with type guards |
| `// @ts-ignore` | Hides type errors | Fix the type error |
| `utils.py` / `helpers.ts` | Junk drawer that grows forever | Name the file after its purpose |
| Bare `except: pass` or `catch {}` | Swallows errors silently | Catch specific exceptions, log them |
| `console.log` / `print()` for debugging | Noise in production | Use structured logging or remove |
| Copy-pasted code blocks | Maintenance nightmare | Extract a shared function |
| Global mutable state | Untestable, race conditions | Use React state/context or pass explicitly |
| Inline magic numbers | Unreadable, fragile | Extract to named constants |
| `useEffect` without cleanup | Memory leaks, leaked timers | Return cleanup function |
| `useEffect` with missing deps | Stale closures, subtle bugs | Exhaustive deps, use refs for stable values |
| Nested ternaries >2 deep | Unreadable | Extract to a function or early returns |
| Boolean params that change behavior | Confusing API | Use separate functions or union types |
| Inline `style` for static values | Inconsistent, unsearchable | Add a class to `components.css` |
| Hardcoded hex colors in JSX | Breaks theming | Use `var(--token)` CSS custom properties |
| Commented-out code | Dead weight | Delete it, git has history |
| Components >300 lines | Unreadable, untestable | Extract hooks and sub-components |
| Hardcoded file paths | Breaks across environments | Use config or path resolution |
| `time.sleep()` in production | Fragile timing | Use proper async/retry patterns |
| `.unwrap()` in Rust | Crashes on error | Use `?` or explicit error handling |

---

## AI Agent-Specific Rules

1. **Search before writing.** Before creating any new file, function, hook, or component, search the existing codebase. Duplicates are rejected.

2. **Follow existing patterns.** Look at how similar things are done in the codebase. Match the style exactly. Consistency beats preference.

3. **No placeholder code.** No `// TODO: implement`, no `raise NotImplementedError`, no stub functions. Every function works or it doesn't exist.

4. **No demo-quality code.** This is production. No `// This is a simplified version`, no shortcuts. Ship-quality or nothing.

5. **Don't refactor what you weren't asked to touch.** If you're fixing a bug in `serve/`, don't reorganize `store/`. Keep scope tight.

6. **Explain non-obvious decisions.** If you chose an approach that isn't the obvious first choice, leave a brief comment explaining why.

7. **Verify types mentally before outputting.** Are all types annotated? All imports used? All effects cleaned up? All arrays capped?

8. **When in doubt, ask.** Ambiguous requirements get clarified, not guessed. A wrong implementation costs more than a question.

9. **Leave the codebase better than you found it.** If you notice a small adjacent issue (missing type, unclear name), fix it if the scope is small. Otherwise, note it.

10. **Use existing libraries.** Before writing any non-trivial utility, check if a well-maintained library already does it. Only write it yourself if the library pulls in a massive dependency tree, you need <20 lines, or it doesn't fit the exact use case.

---

## Enforcement

- `mypy --strict` must pass with zero errors.
- `ruff` must pass with zero warnings.
- `tsc --strict` must pass with zero errors.
- `pytest` with coverage threshold (80% minimum).
- No file over 300 lines merges.
- No `any`/`Any` types merge without an exception.
- No commented-out code merges.

---

*Last updated: March 2026*
*This document is enforced, not aspirational. If the code doesn't meet these standards, it doesn't ship.*
