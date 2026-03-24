# AGENTS.md — Crucible Codebase Standards

> Every agent (Copilot, Claude, Cursor, etc.) MUST follow these rules.
> If a rule conflicts with "getting it done fast," the rule wins.

---

## Core Philosophy

This codebase will be read by humans, debugged by humans, and maintained for years. Clear code is fast code. Clever code is bad code.

**Before writing anything, ask:**
1. Does this already exist somewhere in the codebase? (Search first.)
2. Will a developer understand this in 6 months without context?
3. If this breaks, can someone trace the root cause in under 5 minutes?

---

## Object-Oriented Design Principles

These are not optional. Every module, class, and function must reflect them.

### Single Responsibility
Every file, class, and function does ONE thing. If you can't describe what it does in one sentence without "and," split it. A training runner runs training. A model registry manages model entries. A CLI command parses args and delegates. No god objects.

### Encapsulation
Hide implementation details. Expose behavior through clean interfaces. Internal state is private — callers interact through methods, not by reaching into internals. If a module changes how it stores data internally, no other module should break.

### Abstraction
Callers should not need to know HOW something works, only WHAT it does. `classify_model(path)` returns `(is_hf, model_id)` — the caller doesn't care whether it checks training_config.json, inspects state dict keys, or queries a registry. Push complexity down, keep interfaces simple.

### Polymorphism
Use shared interfaces for things that behave the same way. `TrackingAdapter` works for W&B and TensorBoard. `InterpTokenizer` wraps HuggingFace and Crucible tokenizers behind one interface. When adding a new variant of something, extend the existing interface — don't bolt on special cases with `if/elif` chains.

### Don't Repeat Yourself
Before writing ANY new function, hook, component, or type:
1. Search the codebase for similar functionality.
2. Check `src/core/` (Python) or `components/shared/` (TypeScript) for shared code.
3. If something similar exists, extend or refactor it. Do NOT create a parallel version.

Shared Python code lives in `src/core/`. Shared UI code lives in `components/shared/`. Never in the module that happened to need it first. No files named `utils`, `helpers`, `misc`, or `common`.

---

## Tech Stack

### Python (CLI / SDK / Training)
- **Python** >=3.11, **mypy --strict**, **ruff** (line-length 100), **pytest** (80% min coverage)
- **ML:** PyTorch 2.6, Lance (vector storage), PyArrow
- Frozen dataclasses for all domain objects — no raw dicts for structured data

### Frontend (Studio Desktop App)
- **Tauri 2** (Rust backend, WebView frontend)
- **React 19**, TypeScript strict, react-router 7, Vite
- **Styling:** Plain CSS with design tokens — no Tailwind, no CSS-in-JS
- **State:** React hooks + Context API — no external state library

### Rust (Tauri Commands)
- Shuttle JSON between TypeScript and Python. No business logic in Rust.
- No `.unwrap()` — use `?` or explicit error handling.

---

## Project Structure

```
crucible/
├── src/                        # Python — CLI, SDK, core library
│   ├── core/                   #   Shared types, errors, constants, config
│   ├── cli/                    #   CLI commands (one file per command)
│   ├── serve/                  #   Runners, losses, strategies, remote jobs
│   ├── store/                  #   Dataset SDK, model registry, persistence
│   ├── ingest/                 #   Data ingestion pipeline
│   ├── transforms/             #   Dedup, quality scoring, language detection
│   ├── eval/                   #   Evaluation runners + benchmarks
│   └── crucible.py             #   CrucibleClient SDK entry point
├── studio-app/                 # Frontend — Tauri desktop app
│   ├── src/
│   │   ├── api/                #   Tauri invoke wrappers + CLI arg builders
│   │   ├── components/         #   Reusable UI (shared/, sidebar/)
│   │   ├── context/            #   React Context providers
│   │   ├── pages/              #   Page components organized by domain
│   │   ├── theme/              #   CSS design system
│   │   └── types/              #   Shared TypeScript type definitions
│   └── src-tauri/src/commands/ #   Rust Tauri command handlers
├── tests/                      # Python tests (mirror src/ structure)
└── docs/                       # Architecture docs
```

**Size limits:**
- No file over **300 lines**. No function over **50 lines**. No function with more than **4 parameters** (use an options object).

---

## Naming Conventions

- **Python files:** `snake_case.py` — name describes what it does, not `utils.py`
- **Python functions:** `verb_noun()` — `compute_quality_score()`, not `process()` or `handle()`
- **TypeScript components:** `PascalCase.tsx` — `TrainingWizard.tsx`, not `View.tsx`
- **TypeScript hooks:** `useCamelCase.ts` — `useTrainingConfig.ts`, not `useData.ts`
- **Constants:** `UPPER_SNAKE_CASE` in `src/core/constants.py` or `types/*.ts`
- **Booleans:** prefix with `is`, `has`, `can`, `should`
- **CSS:** semantic class names in `theme/components.css` — `.metric-card`, `.wizard-step`

---

## TypeScript Rules

- `strict: true` is enforced. No `any`, no `as any`, no `@ts-ignore`.
- `interface` for object shapes, `type` for unions/aliases. No enums — use `as const`.
- Every function has explicit parameter types and return types.
- No type assertions (`as Type`) without a comment proving safety.

---

## React Rules

- Components: imports → types → constants → named export function. No default exports.
- Custom hooks extract non-trivial logic. Components call hooks, derive values, return JSX.
- Every `useEffect` has a comment explaining WHAT it reacts to and WHY, and a complete cleanup function.
- State lives as close to usage as possible. Context is for app-wide state only.
- Never store derived state — compute it in render or `useMemo`.

---

## CSS Design Tokens

Use semantic tokens: `var(--text)`, `var(--bg-surface)`, `var(--border)`, `var(--error)`, `var(--success)`. Never hardcode hex colors. Use existing classes from `components.css` before creating new ones. New classes go in `components.css`, not in component files.

---

## Python Rules

### Type Safety
- `mypy --strict` must pass. Frozen `@dataclass` for domain objects. Complete type annotations on every function.

### Error Handling
- Custom exceptions in `src/core/errors.py`, grouped by domain.
- Never catch bare `Exception` unless re-raising with context. Never swallow errors.
- Error messages: what happened + what input caused it + what the user should do.

### Configuration
- All config through `CrucibleConfig` in `src/core/config.py`. No scattered `os.getenv()`.

### Training Code
- **Autograd graph must be preserved.** No in-place ops on leaf tensors, no `.item()`/`.detach()`/`.numpy()` mid-graph.
- Every training loop: gradient clipping, NaN detection, flushed output via `emit_progress()`.

---

## Frontend-Backend Contract

The Studio UI calls the Python CLI via Tauri subprocess:
- **CLI flags are the API.** Add/rename a flag → update `commandArgs.ts`.
- **stdout** is JSON-per-line data. **stderr** is logs/errors.
- **Defaults must stay in sync** between Python (`constants.py`, `*_types.py`) and UI (`training.ts`). Change one → grep and update all.

---

## Testing

### Unit Tests
- Every public function has at least one test. Every error path has a test.
- Test names: `test_<what>_<condition>_<expected>()`. One assert per test.
- Tests run in isolation. Test data in `tests/fixtures/`, never generated inline.

### CLI-First Exhaustive Testing (MANDATORY for model/dataset features)

Type checks and unit tests are not enough. Any feature touching models or datasets must be tested by running it through the actual CLI against every model in the registry.

**Procedure:**
1. List all models in the registry — get every name, local path, and remote path.
2. Run the CLI command for EACH model using its remote path (tests the full resolution pipeline).
3. Verify the output — don't just check exit code 0. Read the JSON result, confirm file sizes are reasonable, confirm output files exist on disk.
4. Test error cases: non-existent paths, missing dependencies, invalid arguments.
5. Test parameter variations (e.g. F16 vs F32, different opset versions).
6. If any model fails, fix the issue and re-test ALL models — fixes often break other code paths.

**Don't declare a feature done until every model combination produces correct output from the CLI.**

---

## AI Agent Rules

1. **Search before writing.** Duplicates are rejected.
2. **Follow existing patterns.** Consistency beats preference.
3. **No placeholder code.** No `TODO: implement`, no stubs. Every function works or doesn't exist.
4. **No demo-quality code.** This is production. Ship-quality or nothing.
5. **Don't refactor what you weren't asked to touch.** Keep scope tight.
6. **When in doubt, ask.** A wrong implementation costs more than a question.
7. **Use existing libraries.** Don't rewrite what a well-maintained package already does.

---

## Enforcement

- `mypy --strict` — zero errors
- `ruff` — zero warnings
- `tsc --strict` — zero errors
- `pytest` — 80% minimum coverage
- No `any`/`Any` types, no commented-out code, no files over 300 lines

---

*Last updated: March 2026*
