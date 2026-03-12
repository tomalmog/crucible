# Crucible Studio UI Overhaul — Design

## Goal

Redesign the Studio desktop app with a clean, clinical aesthetic (Linear/Vercel/Raycast) — collapsible sidebar, single-column list-to-detail navigation, standardized components, tighter spacing.

## Decisions

- **Aesthetic:** Clean & Clinical — subtle rose tint, tight spacing, 1px borders, 4-6px radii
- **Sidebar:** Collapsible — 180px expanded (icon+text) ↔ 48px collapsed (icon-only), persisted in localStorage
- **List pages:** Single-column full-width list → click row → full-width detail view with back button

## Aesthetic Rules

- DM Sans body font, JetBrains Mono for data/code
- 12-13px body text, weight 500 headings (not size increases)
- Accent color only on: active nav, primary buttons, status badges, focus rings
- 1px borders, 4-6px radii everywhere (no 8px rounded cards)
- 20px page padding, 12px panel padding, 6px gaps
- No shadows, no gradients, no decorative elements

## Sidebar

**Expanded (180px):** Brand icon + "Crucible", icon+label nav items, collapse chevron at bottom. No section labels ("Workspace"/"Tools" removed).

**Collapsed (48px):** Centered icons, tooltip on hover, expand chevron at bottom. Instant toggle (no animation).

State persisted in localStorage.

## Page Layout

Every page:
```
Page Title                    [Actions]
─────────────────────────────────────────
[Optional: TabBar or FilterBar]

[Content: list, form, or detail view]
```

## Standardized Components

| Component | Replaces | Description |
|-----------|----------|-------------|
| `ListRow` | `RegistryRow`, `JobRow`, `RemoteJobRow` | Unified clickable row: name, metadata chips, chevron |
| `DetailPage` | two-column right pane | Back button + title + TabBar + content |
| `TabBar` | `.tab-list`/`.tab-item` | Pill/segment style tabs (not underline) |
| `FilterBar` | triple tab-list on Jobs | Compact inline dropdown filters |
| `Panel` | `.panel` (refined) | Tighter padding (12px), 4px radius |
| `MetricCard` | existing (refined) | Label + monospace value |
| `Badge` | existing (refined) | Small pill, semantic color |
| `EmptyState` | existing | Icon + text |
| `PageHeader` | existing (kept) | Title + right-aligned actions |

## Page Changes

**Training:** Method picker grid stays. TabBar replaces header toggle buttons. Tighter card padding.

**Datasets:** Full-width ListRow items (name, count, source, date). Click → DetailPage (Overview, Samples, Ingest, Filter tabs). No Local/Remote split — unified list with location badge.

**Models:** Same as Datasets. Detail tabs: Overview, Merge.

**Jobs:** Full-width ListRow with status badge, method, duration, inline progress bar. FilterBar replaces triple tab-lists (Status, Location, Type dropdowns). Click → full-page JobResultDetail.

**Hub:** Card grid stays for search results (suits browsable content). Clean up card density. Detail view becomes full-page DetailPage.

**Chat / A/B Compare:** Structural changes minimal. Apply new spacing and components.

**Settings:** Stack of Panels. Already clean.

**Clusters:** List → detail pattern.

**Benchmarks / Docs:** Apply new spacing and typography only.

## Removals

- Sidebar section labels ("Workspace", "Tools")
- Two-column master-detail on Datasets/Models
- Underline-style tabs → pill tabs
- Triple filter tab-lists on Jobs → dropdown FilterBar
- 220px sidebar → 180px/48px collapsible
- 8px border radius → 4-6px
