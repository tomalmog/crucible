# UI Overhaul Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign Crucible Studio with a clean/clinical aesthetic — collapsible sidebar, single-column list-to-detail navigation, pill tabs, standardized components.

**Architecture:** Pure frontend changes — CSS theme files + React components. No Python backend changes. The design system in `variables.css`, `reset.css`, `layout.css`, `components.css` gets rewritten for tighter spacing and clinical feel. Shared components get new unified versions (`ListRow`, `TabBar`, `DetailPage`). Page components get refactored to use single-column list→detail pattern instead of two-column master-detail.

**Tech Stack:** React 19, TypeScript, CSS custom properties, lucide-react icons, react-router NavLink

**Design doc:** `docs/plans/2026-03-11-ui-overhaul-design.md`

---

## Task 1: Update CSS Theme — variables.css + reset.css

Tighten the design tokens for the clinical aesthetic.

**Files:**
- Modify: `studio-app/src/theme/variables.css`
- Modify: `studio-app/src/theme/reset.css`

**Step 1: Update variables.css**

In `variables.css`, change these values in `:root`:

```css
/* Radii — tighter */
--radius-sm: 3px;
--radius-md: 4px;
--radius-lg: 6px;

/* Layout — narrower sidebar */
--sidebar-width: 180px;
--sidebar-collapsed-width: 48px;
```

**Step 2: Update reset.css heading sizes**

In `reset.css`, update heading styles to use weight over size (clinical feel):

```css
h1 { font-size: 1.125rem; }
h2 { font-size: 0.875rem; }
h3 { font-size: 0.8125rem; font-weight: 500; }
h4 { font-size: 0.75rem; font-weight: 500; color: var(--text-secondary); }
```

**Step 3: Verify app compiles and renders**

Run: `cd studio-app && npx tsc --noEmit`
Expected: No errors

**Step 4: Commit**

```bash
git add studio-app/src/theme/variables.css studio-app/src/theme/reset.css
git commit -m "tighten design tokens: radii, sidebar width, heading sizes"
```

---

## Task 2: Update CSS Theme — layout.css

Tighten page padding, update sidebar styles for collapsible support, remove two-column layout.

**Files:**
- Modify: `studio-app/src/theme/layout.css`

**Step 1: Update page-content padding**

Change `.page-content` padding from `24px 28px` to `20px`:

```css
.page-content {
    overflow-y: auto;
    overflow-x: hidden;
    padding: 20px;
}
```

**Step 2: Update sidebar styles**

Update `.app-shell` to use the CSS variable:

```css
.app-shell {
    display: grid;
    grid-template-columns: var(--sidebar-width) 1fr;
    height: 100vh;
    overflow: hidden;
}
```

Add collapsed sidebar class:

```css
.app-shell.sidebar-collapsed {
    grid-template-columns: var(--sidebar-collapsed-width) 1fr;
}
```

Update sidebar to support collapsed state:

```css
.app-sidebar {
    background: var(--bg-surface);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
    width: var(--sidebar-width);
    transition: width 0ms;
}

.sidebar-collapsed .app-sidebar {
    width: var(--sidebar-collapsed-width);
}
```

Update `.sidebar-brand`:

```css
.sidebar-brand {
    padding: 12px 10px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
```

Add collapse button style:

```css
.sidebar-collapse-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 6px;
    border: none;
    background: transparent;
    color: var(--text-tertiary);
    cursor: pointer;
    border-radius: var(--radius-sm);
}

.sidebar-collapse-btn:hover {
    background: var(--bg-hover);
    color: var(--text-secondary);
}
```

Add tooltip style for collapsed sidebar:

```css
.nav-item-tooltip {
    display: none;
    position: absolute;
    left: calc(var(--sidebar-collapsed-width) + 4px);
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 3px 8px;
    font-size: 0.6875rem;
    color: var(--text);
    white-space: nowrap;
    z-index: 50;
    pointer-events: none;
}

.sidebar-collapsed .nav-item {
    justify-content: center;
    padding: 8px;
    position: relative;
}

.sidebar-collapsed .nav-item:hover .nav-item-tooltip {
    display: block;
}

.sidebar-collapsed .nav-item-label {
    display: none;
}

.sidebar-collapsed .sidebar-brand h2 span {
    display: none;
}

.sidebar-collapsed .sidebar-section-label {
    display: none;
}
```

**Step 3: Remove `.two-column` layout class**

Delete the `.two-column` CSS rule (it will no longer be used once pages switch to list→detail):

```css
/* DELETE THIS: */
.two-column {
    display: grid;
    grid-template-columns: 260px 1fr;
    gap: 16px;
    min-height: 0;
}
```

Also remove the `@media (max-width: 900px)` rule for `.two-column`.

**Step 4: Update page-header margins**

```css
.page-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
}

.page-header h1 {
    font-size: 1.125rem;
    font-weight: 500;
    letter-spacing: -0.025em;
}
```

**Step 5: Remove sidebar section label "Workspace"/"Tools"**

Already handled via CSS — the section labels will be removed from the React component in Task 4.

**Step 6: Verify and commit**

Run: `cd studio-app && npx tsc --noEmit`

```bash
git add studio-app/src/theme/layout.css
git commit -m "update layout: tighter padding, collapsible sidebar support, remove two-column"
```

---

## Task 3: Update CSS Theme — components.css

Update component styles for the clinical aesthetic: tighter panels, pill tabs, refined buttons.

**Files:**
- Modify: `studio-app/src/theme/components.css`

**Step 1: Tighten panel padding**

```css
.panel {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 12px;
}

.panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 10px;
}
```

**Step 2: Replace underline tabs with pill/segment tabs**

Replace the existing `.tab-list` and `.tab-item` styles:

```css
.tab-list {
    display: flex;
    gap: 2px;
    padding: 2px;
    background: var(--bg-elevated);
    border-radius: var(--radius-md);
    border: 1px solid var(--border-light);
    margin-bottom: 16px;
    width: fit-content;
}

.tab-item {
    padding: 4px 10px;
    background: transparent;
    border: none;
    border-radius: var(--radius-sm);
    color: var(--text-tertiary);
    font-size: 0.6875rem;
    font-weight: 500;
    cursor: pointer;
    transition: color var(--transition-fast), background var(--transition-fast);
}

.tab-item:hover {
    color: var(--text-secondary);
}

.tab-item.active {
    background: var(--bg-base);
    color: var(--text);
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
```

**Step 3: Add ListRow styles**

Add new CSS for the unified list row component:

```css
/* ─── List Row ─── */

.list-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border-light);
    cursor: pointer;
    transition: background var(--transition-fast);
}

.list-row:last-child {
    border-bottom: none;
}

.list-row:hover {
    background: var(--bg-hover);
}

.list-row-name {
    font-size: 0.8125rem;
    font-weight: 500;
    color: var(--text);
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.list-row-meta {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-left: auto;
    flex-shrink: 0;
}

.list-row-meta span {
    font-size: 0.6875rem;
    color: var(--text-tertiary);
}

.list-row-chevron {
    color: var(--text-tertiary);
    flex-shrink: 0;
}

.list-row-actions {
    display: flex;
    gap: 2px;
    flex-shrink: 0;
}
```

**Step 4: Add DetailPage styles**

```css
/* ─── Detail Page ─── */

.detail-back {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.75rem;
    color: var(--text-tertiary);
    cursor: pointer;
    background: none;
    border: none;
    padding: 0;
    margin-bottom: 12px;
    transition: color var(--transition-fast);
}

.detail-back:hover {
    color: var(--text);
}
```

**Step 5: Add FilterBar styles**

```css
/* ─── Filter Bar ─── */

.filter-bar {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 12px;
}

.filter-bar select {
    width: auto;
    min-width: 100px;
    padding: 4px 8px;
    font-size: 0.6875rem;
}
```

**Step 6: Refine button sizes**

Update `.btn` font size:

```css
.btn {
    /* ... keep existing properties ... */
    font-size: 0.6875rem;
    /* ... */
}
```

**Step 7: Remove `.jobs-filters` and `.job-divider` styles**

Delete the `.jobs-filters` and `.job-divider` CSS rules — they will be replaced by the FilterBar.

**Step 8: Verify and commit**

Run: `cd studio-app && npx tsc --noEmit`

```bash
git add studio-app/src/theme/components.css
git commit -m "update components: pill tabs, list-row, detail-page, filter-bar, tighter panels"
```

---

## Task 4: Collapsible Sidebar Component

Rewrite the sidebar to support expanded/collapsed mode with localStorage persistence.

**Files:**
- Modify: `studio-app/src/components/sidebar/AppSidebar.tsx`
- Modify: `studio-app/src/components/sidebar/SidebarNavItem.tsx`
- Modify: `studio-app/src/App.tsx`

**Step 1: Update SidebarNavItem to support collapsed mode**

File: `studio-app/src/components/sidebar/SidebarNavItem.tsx`

```tsx
import { ReactNode } from "react";
import { NavLink } from "react-router";

interface SidebarNavItemProps {
  to: string;
  icon: ReactNode;
  label: string;
}

export function SidebarNavItem({ to, icon, label }: SidebarNavItemProps) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
    >
      <span className="nav-item-icon">{icon}</span>
      <span className="nav-item-label">{label}</span>
      <span className="nav-item-tooltip">{label}</span>
    </NavLink>
  );
}
```

**Step 2: Rewrite AppSidebar with collapse toggle**

File: `studio-app/src/components/sidebar/AppSidebar.tsx`

Remove section labels ("Workspace", "Tools"). Add collapse button. Read/write collapsed state from localStorage.

```tsx
import { useState } from "react";
import { SidebarNavItem } from "./SidebarNavItem";
import {
  Zap, Database, Box, MessageSquare, FlaskConical, Globe,
  Activity, Server, GitCompare, BookOpen, Settings,
  PanelLeftClose, PanelLeftOpen,
} from "lucide-react";

const SIDEBAR_KEY = "crucible_sidebar_collapsed";

function getInitialCollapsed(): boolean {
  return localStorage.getItem(SIDEBAR_KEY) === "true";
}

export function AppSidebar() {
  const [collapsed, setCollapsed] = useState(getInitialCollapsed);

  function toggleCollapsed() {
    const next = !collapsed;
    setCollapsed(next);
    localStorage.setItem(SIDEBAR_KEY, String(next));
    // Dispatch event so App.tsx can update the shell class
    window.dispatchEvent(new CustomEvent("sidebar-toggle", { detail: next }));
  }

  return (
    <aside className="app-sidebar">
      <div className="sidebar-brand">
        <h2>
          <span className="brand-icon">C</span>
          <span>Crucible</span>
        </h2>
      </div>

      <nav className="sidebar-nav">
        <SidebarNavItem to="/training" icon={<Zap size={16} />} label="Training" />
        <SidebarNavItem to="/datasets" icon={<Database size={16} />} label="Datasets" />
        <SidebarNavItem to="/models" icon={<Box size={16} />} label="Models" />
        <SidebarNavItem to="/chat" icon={<MessageSquare size={16} />} label="Chat" />
        <SidebarNavItem to="/benchmarks" icon={<FlaskConical size={16} />} label="Benchmarks" />
        <SidebarNavItem to="/hub" icon={<Globe size={16} />} label="Hub" />

        <div className="sidebar-divider" />

        <SidebarNavItem to="/jobs" icon={<Activity size={16} />} label="Jobs" />
        <SidebarNavItem to="/clusters" icon={<Server size={16} />} label="Clusters" />
        <SidebarNavItem to="/compare-chat" icon={<GitCompare size={16} />} label="A/B Compare" />
        <SidebarNavItem to="/docs" icon={<BookOpen size={16} />} label="Docs" />
        <SidebarNavItem to="/settings" icon={<Settings size={16} />} label="Settings" />
      </nav>

      <div className="sidebar-footer">
        <button className="sidebar-collapse-btn" onClick={toggleCollapsed} title={collapsed ? "Expand sidebar" : "Collapse sidebar"}>
          {collapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
        </button>
      </div>
    </aside>
  );
}
```

**Step 3: Add sidebar divider CSS**

In `layout.css`, add:

```css
.sidebar-divider {
    height: 1px;
    background: var(--border-light);
    margin: 6px 8px;
}
```

**Step 4: Update App.tsx to toggle shell class**

File: `studio-app/src/App.tsx`

```tsx
import { useEffect, useState } from "react";
import { Outlet } from "react-router";
import { CrucibleProvider } from "./context/CrucibleContext";
import { CommandProvider } from "./context/CommandContext";
import { AppSidebar } from "./components/sidebar/AppSidebar";
import "./theme/variables.css";
import "./theme/reset.css";
import "./theme/components.css";
import "./theme/layout.css";

const SIDEBAR_KEY = "crucible_sidebar_collapsed";

function App() {
  const [collapsed, setCollapsed] = useState(
    () => localStorage.getItem(SIDEBAR_KEY) === "true"
  );

  useEffect(() => {
    function onToggle(e: Event) {
      setCollapsed((e as CustomEvent).detail as boolean);
    }
    window.addEventListener("sidebar-toggle", onToggle);
    return () => window.removeEventListener("sidebar-toggle", onToggle);
  }, []);

  return (
    <CrucibleProvider>
      <CommandProvider>
        <main className={`app-shell${collapsed ? " sidebar-collapsed" : ""}`}>
          <AppSidebar />
          <div className="page-content">
            <Outlet />
          </div>
        </main>
      </CommandProvider>
    </CrucibleProvider>
  );
}

export default App;
```

**Step 5: Remove sidebar footer Settings link**

The current sidebar has a duplicate Settings link in the footer. Remove it — Settings is already in the main nav.

**Step 6: Verify and commit**

Run: `cd studio-app && npx tsc --noEmit`

Open the app and test:
- Sidebar shows expanded with icon + text
- Click collapse button → sidebar shrinks to icons only
- Hover icon in collapsed mode → tooltip shows label
- Reload page → collapsed state persists
- Nav items still route correctly in both modes

```bash
git add studio-app/src/components/sidebar/ studio-app/src/App.tsx studio-app/src/theme/layout.css
git commit -m "collapsible sidebar: toggle between icon+text and icon-only"
```

---

## Task 5: Create Shared ListRow Component

Create a unified list row component that replaces RegistryRow for use on Datasets, Models, Jobs, and Clusters pages.

**Files:**
- Create: `studio-app/src/components/shared/ListRow.tsx`

**Step 1: Create the ListRow component**

```tsx
import { ReactNode } from "react";
import { ChevronRight } from "lucide-react";

interface ListRowProps {
  /** Primary label displayed on the left */
  name: string;
  /** Optional secondary metadata elements displayed before the chevron */
  meta?: ReactNode;
  /** Optional action buttons (stop propagation internally) */
  actions?: ReactNode;
  /** Show right chevron (default true) */
  showChevron?: boolean;
  /** Click handler for the entire row */
  onClick?: () => void;
}

export function ListRow({ name, meta, actions, showChevron = true, onClick }: ListRowProps) {
  return (
    <div className="list-row" onClick={onClick}>
      <span className="list-row-name">{name}</span>
      {meta && <div className="list-row-meta">{meta}</div>}
      {actions && (
        <div className="list-row-actions" onClick={(e) => e.stopPropagation()}>
          {actions}
        </div>
      )}
      {showChevron && <ChevronRight size={14} className="list-row-chevron" />}
    </div>
  );
}
```

**Step 2: Verify compiles**

Run: `cd studio-app && npx tsc --noEmit`

**Step 3: Commit**

```bash
git add studio-app/src/components/shared/ListRow.tsx
git commit -m "add ListRow shared component for unified list items"
```

---

## Task 6: Create Shared TabBar Component

Create a React wrapper for the pill-style tab bar.

**Files:**
- Create: `studio-app/src/components/shared/TabBar.tsx`

**Step 1: Create TabBar component**

```tsx
interface TabBarProps<T extends string> {
  tabs: readonly T[];
  active: T;
  onChange: (tab: T) => void;
  /** Optional label formatter — defaults to capitalizing first letter */
  format?: (tab: T) => string;
}

export function TabBar<T extends string>({ tabs, active, onChange, format }: TabBarProps<T>) {
  const fmt = format ?? ((t: T) => t.charAt(0).toUpperCase() + t.slice(1));
  return (
    <div className="tab-list">
      {tabs.map((t) => (
        <button
          key={t}
          className={`tab-item${t === active ? " active" : ""}`}
          onClick={() => onChange(t)}
        >
          {fmt(t)}
        </button>
      ))}
    </div>
  );
}
```

**Step 2: Verify compiles**

Run: `cd studio-app && npx tsc --noEmit`

**Step 3: Commit**

```bash
git add studio-app/src/components/shared/TabBar.tsx
git commit -m "add TabBar shared component with pill styling"
```

---

## Task 7: Create Shared DetailPage Shell

Create a reusable detail page wrapper with back button.

**Files:**
- Create: `studio-app/src/components/shared/DetailPage.tsx`

**Step 1: Create DetailPage component**

```tsx
import { ReactNode } from "react";
import { ArrowLeft } from "lucide-react";

interface DetailPageProps {
  title: string;
  onBack: () => void;
  actions?: ReactNode;
  children: ReactNode;
}

export function DetailPage({ title, onBack, actions, children }: DetailPageProps) {
  return (
    <>
      <button className="detail-back" onClick={onBack}>
        <ArrowLeft size={14} /> Back
      </button>
      <div className="page-header">
        <h1>{title}</h1>
        {actions && <div className="page-header-actions">{actions}</div>}
      </div>
      {children}
    </>
  );
}
```

**Step 2: Verify compiles**

Run: `cd studio-app && npx tsc --noEmit`

**Step 3: Commit**

```bash
git add studio-app/src/components/shared/DetailPage.tsx
git commit -m "add DetailPage shared component with back button"
```

---

## Task 8: Refactor DatasetsPage — List→Detail

Convert Datasets from two-column master-detail to single-column list→detail.

**Files:**
- Modify: `studio-app/src/pages/datasets/DatasetsPage.tsx`
- Modify: `studio-app/src/pages/datasets/DatasetListPanel.tsx` (refactor to flat list)

**Step 1: Rewrite DatasetsPage**

The page now has two modes: `list` and `detail`. In list mode, it shows a full-width list of datasets. Click a dataset to switch to detail mode, which shows the full-width detail with a back button and tabs.

```tsx
import { useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { TabBar } from "../../components/shared/TabBar";
import { DetailPage } from "../../components/shared/DetailPage";
import { ListRow } from "../../components/shared/ListRow";
import { DatasetDashboard } from "./DatasetDashboard";
import { SampleInspector } from "./SampleInspector";
import { IngestForm } from "./IngestForm";
import { FilterForm } from "./FilterForm";
import { useCrucible } from "../../context/CrucibleContext";
import { EmptyState } from "../../components/shared/EmptyState";
import { Database } from "lucide-react";

type DetailTab = "overview" | "samples" | "ingest" | "filter";
const DETAIL_TABS = ["overview", "samples", "ingest", "filter"] as const;

export function DatasetsPage() {
  const { datasets, selectedDataset, setSelectedDataset, refreshDatasets } = useCrucible();
  const [showDetail, setShowDetail] = useState(false);
  const [tab, setTab] = useState<DetailTab>("overview");
  const [isRefreshing, setIsRefreshing] = useState(false);

  function handleSelect(ds: string) {
    setSelectedDataset(ds);
    setShowDetail(true);
    setTab("overview");
  }

  function handleBack() {
    setShowDetail(false);
  }

  async function handleRefresh(): Promise<void> {
    setIsRefreshing(true);
    await refreshDatasets();
    setIsRefreshing(false);
  }

  if (showDetail && selectedDataset) {
    return (
      <DetailPage title={selectedDataset} onBack={handleBack}>
        <TabBar tabs={DETAIL_TABS} active={tab} onChange={setTab} />
        {tab === "overview" && <DatasetDashboard />}
        {tab === "samples" && <SampleInspector />}
        {tab === "ingest" && <IngestForm />}
        {tab === "filter" && <FilterForm />}
      </DetailPage>
    );
  }

  return (
    <>
      <PageHeader title="Datasets">
        <button className="btn" onClick={() => handleRefresh().catch(console.error)} disabled={isRefreshing}>
          {isRefreshing ? "Refreshing..." : "Refresh"}
        </button>
      </PageHeader>

      {datasets.length === 0 ? (
        <EmptyState title="No datasets" description="Ingest data from the Training page or CLI." />
      ) : (
        <div className="panel panel-flush">
          {datasets.map((ds) => (
            <ListRow
              key={ds.name}
              name={ds.name}
              meta={<span>{ds.recordCount?.toLocaleString() ?? "—"} records</span>}
              onClick={() => handleSelect(ds.name)}
            />
          ))}
        </div>
      )}
    </>
  );
}
```

NOTE: The exact shape of the `datasets` array depends on what `useCrucible()` provides. The implementer should check `CrucibleContext.tsx` to see what dataset fields are available (likely `name` and `recordCount` or similar). Adjust the `ListRow` `meta` prop accordingly. If the datasets array is just strings, use `ds` directly as the name with no meta.

**Step 2: Keep DatasetListPanel for potential reuse but it's no longer imported by DatasetsPage**

Don't delete `DatasetListPanel.tsx` yet — it may be needed for the remote dataset transfer functionality. But `DatasetsPage.tsx` no longer imports it.

**Step 3: Verify compiles**

Run: `cd studio-app && npx tsc --noEmit`

If there are type errors due to dataset shape, check `CrucibleContext.tsx` for the actual dataset type and adjust.

**Step 4: Test in browser**

- Open Datasets page → see full-width list of datasets
- Click a dataset → detail view with back button and tabs
- Click back → returns to list
- Tabs (Overview, Samples, Ingest, Filter) all render correctly

**Step 5: Commit**

```bash
git add studio-app/src/pages/datasets/DatasetsPage.tsx
git commit -m "refactor datasets: single-column list to detail navigation"
```

---

## Task 9: Refactor ModelsPage — List→Detail

Same pattern as Datasets. Convert from two-column to list→detail.

**Files:**
- Modify: `studio-app/src/pages/models/ModelsPage.tsx`

**Step 1: Rewrite ModelsPage**

Follow the same pattern as DatasetsPage: list mode shows all models as full-width `ListRow` items, click navigates to detail mode with `DetailPage` shell + `TabBar`.

The implementer should check:
- `useCrucible()` for the `models` array shape and `selectedModel`/`setSelectedModel`
- What the `ModelOverview` and `ModelMergeForm` components expect as props
- The `ModelEntry` type from `types/`

The detail tabs are: `overview` and `merge`.

Show model name, size (formatted), and any version count as metadata in `ListRow`.

**Step 2: Remove the ModelListPanel import**

Don't delete the file yet (it handles remote operations) but DatasetsPage/ModelsPage no longer use it.

**Step 3: Verify, test, commit**

Run: `cd studio-app && npx tsc --noEmit`

Test: Models list → click model → detail with back button → Overview/Merge tabs work.

```bash
git add studio-app/src/pages/models/ModelsPage.tsx
git commit -m "refactor models: single-column list to detail navigation"
```

---

## Task 10: Refactor JobsPage — FilterBar + ListRow

Replace the triple tab-list filter system with a FilterBar using dropdowns. Use ListRow for job items.

**Files:**
- Modify: `studio-app/src/pages/jobs/JobsPage.tsx`

**Step 1: Replace filter UI**

Replace the 3 `tab-list` blocks with a single `filter-bar`:

```tsx
<div className="filter-bar">
  <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value as StatusFilter)}>
    <option value="all">All Status</option>
    <option value="running">Running</option>
    <option value="completed">Completed</option>
    <option value="failed">Failed</option>
  </select>
  <select value={locationFilter} onChange={(e) => setLocationFilter(e.target.value as LocationFilter)}>
    <option value="all">All Locations</option>
    <option value="local">Local</option>
    <option value="remote">Remote</option>
  </select>
  <select value={typeFilter} onChange={(e) => setTypeFilter(e.target.value as TaskTypeFilter)}>
    <option value="all">All Types</option>
    <option value="training">Training</option>
    <option value="eval">Eval</option>
    <option value="sweep">Sweep</option>
  </select>
</div>
```

**Step 2: Keep JobRow and RemoteJobRow**

The job rows are complex enough (progress bars, expandable logs, inline editing) that they should keep their own components rather than switching to `ListRow`. But apply the new CSS styles for consistency — ensure they use the `list-row` pattern for the collapsed state.

**Step 3: Make job detail views use DetailPage**

When `viewingJob` or `viewingRemoteJob` is set, wrap the result in `DetailPage`:

```tsx
if (viewingJob) {
  return (
    <DetailPage title={viewingJob.label || viewingJob.command} onBack={() => setViewingJob(null)}>
      <JobResultDetail job={viewingJob} onBack={() => setViewingJob(null)} />
    </DetailPage>
  );
}
```

Note: `JobResultDetail` already has its own back button — the implementer should either remove the back button from `JobResultDetail` and rely on `DetailPage`, or not wrap in `DetailPage` and keep the existing back button. Choose whichever is cleaner.

**Step 4: Verify, test, commit**

Run: `cd studio-app && npx tsc --noEmit`

Test: Jobs page shows filter dropdowns, jobs list, expand/collapse works, detail view works.

```bash
git add studio-app/src/pages/jobs/JobsPage.tsx
git commit -m "refactor jobs: replace triple tab-filters with dropdown filter bar"
```

---

## Task 11: Refactor HubPage — Clean Up Cards + Detail

The Hub page already works well structurally. Apply clinical aesthetic refinements.

**Files:**
- Modify: `studio-app/src/pages/hub/HubPage.tsx`

**Step 1: Use TabBar component**

Replace the inline tab-list with `TabBar`:

```tsx
import { TabBar } from "../../components/shared/TabBar";

const HUB_TABS = ["models", "datasets", "push"] as const;

// In render:
<TabBar tabs={HUB_TABS} active={tab} onChange={setTab} />
```

**Step 2: Verify and commit**

```bash
git add studio-app/src/pages/hub/HubPage.tsx
git commit -m "hub: use TabBar component"
```

---

## Task 12: Refactor TrainingPage — TabBar

Replace the header toggle buttons with TabBar.

**Files:**
- Modify: `studio-app/src/pages/training/TrainingPage.tsx`

**Step 1: Replace header buttons with TabBar**

Currently the Training page has "New Run", "Sweep", "History" as buttons in the PageHeader. Replace with a TabBar below the header.

Map the view state to tab names:
- `pick` and `wizard` both map to "new-run" tab
- `sweep` maps to "sweep" tab
- `history` maps to "history" tab

```tsx
import { TabBar } from "../../components/shared/TabBar";

type Tab = "new-run" | "sweep" | "history";

// Derive tab from view:
const tabFromView: Record<View, Tab> = { pick: "new-run", wizard: "new-run", sweep: "sweep", history: "history" };

function handleTabChange(t: Tab) {
  if (t === "new-run") { setView("pick"); setSelectedMethod(null); }
  else if (t === "sweep") setView("sweep");
  else setView("history");
}

// In render:
<PageHeader title="Training" />
<TabBar tabs={["new-run", "sweep", "history"] as const} active={tabFromView[view]} onChange={handleTabChange} format={(t) => t === "new-run" ? "New Run" : t.charAt(0).toUpperCase() + t.slice(1)} />
```

**Step 2: Verify and commit**

```bash
git add studio-app/src/pages/training/TrainingPage.tsx
git commit -m "training: replace header buttons with TabBar"
```

---

## Task 13: Update Remaining Pages to Use TabBar

Several pages render inline `tab-list` / `tab-item` markup. Replace all with the `TabBar` component.

**Files to check and update:**
- `studio-app/src/pages/datasets/DatasetListPanel.tsx` (if still rendering tabs)
- `studio-app/src/pages/models/ModelListPanel.tsx` (if still rendering tabs)
- Any other page with inline tab markup

**Step 1: Search for remaining `.tab-list` usage in TSX files**

Run: `grep -r "tab-list" studio-app/src/pages/ --include="*.tsx" -l`

For each file found, replace inline tab rendering with the `TabBar` component.

**Step 2: Verify and commit**

```bash
git add studio-app/src/pages/
git commit -m "replace all inline tab markup with TabBar component"
```

---

## Task 14: Clean Up — Remove Dead Code and Verify

Remove unused components and verify the full build.

**Files:**
- Check if `DatasetListPanel.tsx` is still imported anywhere — if not, consider removing (check if remote transfer features still need it)
- Check if `ModelListPanel.tsx` is still imported anywhere
- Remove any unused imports across modified files

**Step 1: Search for unused imports**

Run: `cd studio-app && npx tsc --noEmit 2>&1`

Fix any type errors or unused import warnings.

**Step 2: Full build verification**

Run: `cd studio-app && npm run build 2>&1`

Expected: Build succeeds with no errors.

**Step 3: Visual verification**

Open the app and check every page:
- [ ] Sidebar: expands/collapses, tooltips work, nav routing works
- [ ] Training: TabBar works, method picker renders, wizard works
- [ ] Datasets: list view, click to detail, back button, all tabs
- [ ] Models: list view, click to detail, back button, tabs
- [ ] Chat: renders correctly with new spacing
- [ ] Hub: TabBar, card grid, detail view
- [ ] Jobs: filter dropdowns, job list, expand, detail view
- [ ] Clusters: renders with new spacing
- [ ] A/B Compare: renders correctly
- [ ] Benchmarks: renders correctly
- [ ] Docs: renders correctly
- [ ] Settings: renders correctly
- [ ] Dark mode: toggle theme, verify all pages

**Step 4: Final commit**

```bash
git add -A
git commit -m "UI overhaul cleanup: remove dead code, verify build"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | CSS tokens: radii, sidebar width, headings | variables.css, reset.css |
| 2 | Layout: page padding, sidebar collapse, remove two-column | layout.css |
| 3 | Components: pill tabs, list-row, detail-page, filter-bar | components.css |
| 4 | Collapsible sidebar React component | AppSidebar.tsx, SidebarNavItem.tsx, App.tsx |
| 5 | ListRow shared component | ListRow.tsx (new) |
| 6 | TabBar shared component | TabBar.tsx (new) |
| 7 | DetailPage shared component | DetailPage.tsx (new) |
| 8 | Datasets → list-to-detail | DatasetsPage.tsx |
| 9 | Models → list-to-detail | ModelsPage.tsx |
| 10 | Jobs → filter bar + detail | JobsPage.tsx |
| 11 | Hub → TabBar | HubPage.tsx |
| 12 | Training → TabBar | TrainingPage.tsx |
| 13 | Remaining pages → TabBar | various |
| 14 | Clean up + verify | all |
