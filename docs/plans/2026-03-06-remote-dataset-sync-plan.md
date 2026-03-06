# Remote Dataset Sync + UI Cleanup — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add persistent remote dataset management (push/pull/list/delete via SSH), simplify job submission to require pre-pushed datasets, and clean up the Datasets UI (remove versions, top sources, diff tab).

**Architecture:** New `remote_dataset_ops.py` module provides four SSH-based operations. Job submission drops the data upload phase entirely — datasets must be pushed first. The Datasets page gets a Local/Remote tab split in the sidebar. Training forms use dataset names only (no paths).

**Tech Stack:** Python (paramiko SSH, tarfile), TypeScript/React (Tauri invoke), Rust (new Tauri commands that shell out to Python CLI).

---

### Task 1: Create `remote_dataset_ops.py` — push, list, pull, delete

**Files:**
- Create: `src/serve/remote_dataset_ops.py`

**Step 1: Create the module with all four operations**

```python
"""Remote dataset operations: push, list, pull, delete via SSH.

Manages a persistent dataset library on remote clusters at
{remote_workspace}/datasets/{name}/.
"""

from __future__ import annotations

import json
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

from core.constants import CATALOG_FILE_NAME, DATASETS_DIR_NAME, VERSIONS_DIR_NAME
from core.errors import ForgeRemoteError
from core.slurm_types import ClusterConfig
from serve.ssh_connection import SshSession
from store.cluster_registry import load_cluster


@dataclass(frozen=True)
class RemoteDatasetInfo:
    """Lightweight summary of a dataset on a remote cluster."""

    name: str
    record_count: int
    version_id: str
    synced_at: str


def push_dataset(
    session: SshSession,
    cluster: ClusterConfig,
    dataset_name: str,
    data_root: Path,
) -> RemoteDatasetInfo:
    """Push the latest local dataset snapshot to a remote cluster.

    Reads the local catalog, extracts the latest version's records.jsonl,
    uploads it with a metadata.json to {remote_workspace}/datasets/{name}/.
    """
    from store.catalog_io import read_catalog_file

    ds_dir = data_root / DATASETS_DIR_NAME / dataset_name
    catalog_path = ds_dir / CATALOG_FILE_NAME
    if not catalog_path.exists():
        raise ForgeRemoteError(f"Dataset '{dataset_name}' not found locally.")

    catalog = read_catalog_file(catalog_path)
    latest_id = catalog.get("latest_version", "")
    if not latest_id:
        raise ForgeRemoteError(
            f"Dataset '{dataset_name}' has no latest version.",
        )

    version_dir = ds_dir / VERSIONS_DIR_NAME / latest_id
    records_path = version_dir / "records.jsonl"
    if not records_path.exists():
        raise ForgeRemoteError(
            f"Records file missing for version {latest_id}.",
        )

    # Count records
    record_count = sum(1 for _ in records_path.open(encoding="utf-8"))

    # Find the manifest for record_count (if available)
    manifest_path = version_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record_count = manifest.get("record_count", record_count)

    from datetime import datetime, timezone

    synced_at = datetime.now(timezone.utc).isoformat()

    # Build metadata
    metadata = {
        "name": dataset_name,
        "record_count": record_count,
        "version_id": latest_id,
        "synced_at": synced_at,
    }

    # Tar records.jsonl + metadata.json
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        meta_file = tmp_path / "metadata.json"
        meta_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        tar_path = tmp_path / "dataset.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(str(records_path), arcname="records.jsonl")
            tar.add(str(meta_file), arcname="metadata.json")

        remote_ds = f"{cluster.remote_workspace}/datasets/{dataset_name}"
        session.mkdir_p(remote_ds)
        session.upload(tar_path, f"{remote_ds}/dataset.tar.gz")
        session.execute(
            f"cd {remote_ds} && tar xzf dataset.tar.gz && rm dataset.tar.gz",
        )

    return RemoteDatasetInfo(
        name=dataset_name,
        record_count=record_count,
        version_id=latest_id,
        synced_at=synced_at,
    )


def list_remote_datasets(
    session: SshSession,
    cluster: ClusterConfig,
) -> list[RemoteDatasetInfo]:
    """List all datasets on a remote cluster by reading metadata.json files."""
    datasets_dir = f"{cluster.remote_workspace}/datasets"
    stdout, _, code = session.execute(
        f"ls -1 {datasets_dir} 2>/dev/null || true", timeout=15,
    )
    if code != 0 or not stdout.strip():
        return []

    results: list[RemoteDatasetInfo] = []
    names = [n.strip() for n in stdout.strip().split("\n") if n.strip()]

    for name in names:
        meta_path = f"{datasets_dir}/{name}/metadata.json"
        out, _, rc = session.execute(f"cat {meta_path} 2>/dev/null", timeout=10)
        if rc != 0 or not out.strip():
            continue
        try:
            meta = json.loads(out)
            results.append(RemoteDatasetInfo(
                name=meta.get("name", name),
                record_count=meta.get("record_count", 0),
                version_id=meta.get("version_id", ""),
                synced_at=meta.get("synced_at", ""),
            ))
        except (json.JSONDecodeError, KeyError):
            continue

    return results


def delete_remote_dataset(
    session: SshSession,
    cluster: ClusterConfig,
    dataset_name: str,
) -> None:
    """Delete a dataset from a remote cluster."""
    remote_ds = f"{cluster.remote_workspace}/datasets/{dataset_name}"
    _, stderr, code = session.execute(f"rm -rf {remote_ds}", timeout=30)
    if code != 0:
        raise ForgeRemoteError(
            f"Failed to delete remote dataset '{dataset_name}': {stderr}",
        )


def pull_remote_dataset(
    session: SshSession,
    cluster: ClusterConfig,
    dataset_name: str,
    data_root: Path,
) -> Path:
    """Pull a remote dataset's records.jsonl to a local directory.

    Downloads the records.jsonl file and returns the path to it.
    The caller can then ingest it into the local dataset store.
    """
    remote_ds = f"{cluster.remote_workspace}/datasets/{dataset_name}"
    remote_records = f"{remote_ds}/records.jsonl"

    # Verify it exists
    _, _, code = session.execute(
        f"test -f {remote_records}", timeout=10,
    )
    if code != 0:
        raise ForgeRemoteError(
            f"Dataset '{dataset_name}' not found on cluster.",
        )

    local_dir = data_root / "cache" / "remote-pulls" / dataset_name
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / "records.jsonl"
    session.download(remote_records, local_path)

    return local_path
```

**Step 2: Verify import works**

Run: `cd "/Users/tomalmog/programming/Febuary 2026/forge" && PYTHONPATH=src python -c "from serve.remote_dataset_ops import push_dataset, list_remote_datasets, delete_remote_dataset, pull_remote_dataset, RemoteDatasetInfo; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/serve/remote_dataset_ops.py
git commit -m "add remote dataset ops: push, list, pull, delete via SSH"
```

---

### Task 2: Add CLI commands for remote dataset operations

**Files:**
- Modify: `src/cli/remote_command.py` (add `dataset-push`, `dataset-list`, `dataset-pull`, `dataset-delete` subcommands)
- Modify: `src/cli/remote_command_handlers.py` (add handler functions)

**Step 1: Add subcommand parsers to `remote_command.py`**

After the existing subcommand registrations (around the `cancel` and `pull-model` parsers), add four new subcommands:

```python
# Dataset operations
ds_push = sub.add_parser("dataset-push", help="Push a local dataset to a remote cluster")
ds_push.add_argument("--cluster", required=True)
ds_push.add_argument("--dataset", required=True, help="Local dataset name to push")

ds_list = sub.add_parser("dataset-list", help="List datasets on a remote cluster")
ds_list.add_argument("--cluster", required=True)

ds_pull = sub.add_parser("dataset-pull", help="Pull a remote dataset to local")
ds_pull.add_argument("--cluster", required=True)
ds_pull.add_argument("--dataset", required=True, help="Remote dataset name to pull")

ds_delete = sub.add_parser("dataset-delete", help="Delete a dataset from a remote cluster")
ds_delete.add_argument("--cluster", required=True)
ds_delete.add_argument("--dataset", required=True, help="Remote dataset name to delete")
```

Add dispatch entries in `run_remote_command()`:

```python
"dataset-push": _handle_dataset_push,
"dataset-list": _handle_dataset_list,
"dataset-pull": _handle_dataset_pull,
"dataset-delete": _handle_dataset_delete,
```

**Step 2: Add handler functions to `remote_command_handlers.py`**

```python
def _handle_dataset_push(client: ForgeClient, args: argparse.Namespace) -> int:
    from serve.remote_dataset_ops import push_dataset
    from serve.ssh_connection import SshSession
    from store.cluster_registry import load_cluster

    cluster = load_cluster(client._config.data_root, args.cluster)
    with SshSession(cluster) as session:
        info = push_dataset(session, cluster, args.dataset, client._config.data_root)
    print(f"Pushed '{info.name}' ({info.record_count} records) to {args.cluster}")
    return 0


def _handle_dataset_list(client: ForgeClient, args: argparse.Namespace) -> int:
    import json as json_mod

    from serve.remote_dataset_ops import list_remote_datasets
    from serve.ssh_connection import SshSession
    from store.cluster_registry import load_cluster

    cluster = load_cluster(client._config.data_root, args.cluster)
    with SshSession(cluster) as session:
        datasets = list_remote_datasets(session, cluster)
    if not datasets:
        print("No datasets on cluster.")
        # Print empty JSON array for Tauri to parse
        print("FORGE_JSON:" + json_mod.dumps([]))
        return 0
    for ds in datasets:
        print(f"  {ds.name}  {ds.record_count} records  synced {ds.synced_at}")
    # Machine-readable output for Tauri
    print("FORGE_JSON:" + json_mod.dumps([
        {"name": d.name, "record_count": d.record_count,
         "version_id": d.version_id, "synced_at": d.synced_at}
        for d in datasets
    ]))
    return 0


def _handle_dataset_pull(client: ForgeClient, args: argparse.Namespace) -> int:
    from serve.remote_dataset_ops import pull_remote_dataset
    from serve.ssh_connection import SshSession
    from store.cluster_registry import load_cluster

    cluster = load_cluster(client._config.data_root, args.cluster)
    with SshSession(cluster) as session:
        local_path = pull_remote_dataset(
            session, cluster, args.dataset, client._config.data_root,
        )
    print(f"Pulled '{args.dataset}' to {local_path}")
    return 0


def _handle_dataset_delete(client: ForgeClient, args: argparse.Namespace) -> int:
    from serve.remote_dataset_ops import delete_remote_dataset
    from serve.ssh_connection import SshSession
    from store.cluster_registry import load_cluster

    cluster = load_cluster(client._config.data_root, args.cluster)
    with SshSession(cluster) as session:
        delete_remote_dataset(session, cluster, args.dataset)
    print(f"Deleted '{args.dataset}' from {args.cluster}")
    return 0
```

**Step 3: Verify CLI parses correctly**

Run: `cd "/Users/tomalmog/programming/Febuary 2026/forge" && PYTHONPATH=src python -m cli.main remote dataset-push --help`
Expected: Shows help with `--cluster` and `--dataset` args

**Step 4: Commit**

```bash
git add src/cli/remote_command.py src/cli/remote_command_handlers.py
git commit -m "add CLI commands for remote dataset push, list, pull, delete"
```

---

### Task 3: Simplify job submission — remove data upload phase

**Files:**
- Modify: `src/serve/remote_job_submitter.py` (remove data upload calls, add dataset existence check)
- Modify: `src/serve/remote_data_upload.py` (remove `_handle_data_strategy`, `_upload_raw_dataset`, `_upload_dataset_catalog`, `_RECORD_BASED_METHODS`)
- Modify: `src/cli/remote_command_handlers.py` (simplify `_handle_submit`, remove `_resolve_dataset`)
- Modify: `src/cli/remote_command.py` (remove `--data-strategy` and `--dataset` flags from submit)

**Step 1: Update `remote_job_submitter.py`**

Remove the `data_strategy`, `dataset_path` params from `submit_remote_job()`. Add dataset name validation. The function should:

1. Drop `DataStrategy` and `DATA_PATH_FIELDS` imports
2. Drop `_handle_data_strategy` import
3. Remove `data_strategy` and `dataset_path` parameters
4. Remove the "Uploading data..." phase
5. After connecting, check that dataset exists on remote if `dataset_name` is in `method_args`
6. Write the remote dataset path into method_args so the agent knows where data lives

The key change in the SSH block (lines 107-126) — replace:
```python
_update_phase(data_root, job_id, "Uploading data...")
_handle_data_strategy(
    session, data_strategy, dataset_path, method_args,
    workdir, training_method, data_root,
)
```

With:
```python
# Verify dataset exists on cluster
ds_name = str(method_args.get("dataset_name", ""))
if ds_name:
    ds_path = f"{cluster.remote_workspace}/datasets/{ds_name}"
    _, _, rc = session.execute(f"test -d {ds_path}", timeout=10)
    if rc != 0:
        raise ForgeRemoteError(
            f"Dataset '{ds_name}' not found on cluster '{cluster_name}'. "
            "Push it first with: forge remote dataset-push "
            f"--cluster {cluster_name} --dataset {ds_name}"
        )
    # Point training config at the remote dataset
    method_args["dataset_path"] = f"{ds_path}/records.jsonl"
```

Apply the same simplification to `submit_remote_sweep()` — remove `data_strategy`, `dataset_path` params, remove the data upload block (lines 200-211), add the same dataset check.

**Step 2: Clean up `remote_data_upload.py`**

Remove these functions (they're no longer called):
- `_upload_raw_dataset` (lines 98-125)
- `_upload_dataset_catalog` (lines 128-210)
- `_handle_data_strategy` (lines 213-277)
- `_RECORD_BASED_METHODS` (line 95)

Remove unused imports: `DATA_PATH_FIELDS` from training_methods, `DataStrategy` from slurm_types, `CATALOG_FILE_NAME`, `DATASETS_DIR_NAME`, `VERSIONS_DIR_NAME` from constants.

What remains in `remote_data_upload.py`: `_upload_bundle`, `_upload_config`, `_upload_script`, `_submit_sbatch`, `_generate_script` — the core submission helpers.

**Step 3: Simplify `_handle_submit` in `remote_command_handlers.py`**

Remove `_resolve_dataset()` function entirely. Remove `data_strategy` and `dataset_path` from the `submit_remote_job()` call. Remove `dataset_path, data_strategy = _resolve_dataset(...)` call. The handler becomes:

```python
def _handle_submit(client: ForgeClient, args: argparse.Namespace) -> int:
    from core.slurm_types import SlurmResourceConfig
    from serve.remote_job_submitter import submit_remote_job

    method_args = json.loads(args.method_args)
    resources = _build_resources(args)
    record = submit_remote_job(
        data_root=client._config.data_root,
        cluster_name=args.cluster,
        training_method=args.method,
        method_args=method_args,
        resources=resources,
        pull_model=args.pull_model,
        model_name=args.model_name,
    )
    print(f"Submitted job {record.job_id} (Slurm: {record.slurm_job_id})")
    return 0
```

Similarly simplify `_handle_submit_sweep`.

**Step 4: Remove `--data-strategy` and `--dataset` from CLI parser**

In `remote_command.py`, in `_add_submit_args()`, remove the `--dataset` and `--data-strategy` argument registrations.

**Step 5: Verify import chain works**

Run: `cd "/Users/tomalmog/programming/Febuary 2026/forge" && PYTHONPATH=src python -c "from serve.remote_job_submitter import submit_remote_job; print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add src/serve/remote_job_submitter.py src/serve/remote_data_upload.py src/cli/remote_command_handlers.py src/cli/remote_command.py
git commit -m "simplify remote job submission: require pre-pushed datasets, remove data upload phase"
```

---

### Task 4: Add Tauri commands for remote dataset operations

**Files:**
- Modify: `studio-app/src-tauri/src/lib.rs` (register new commands)
- Modify: `studio-app/src-tauri/src/commands/remote_commands.rs` (or create if needed — add push/list/pull/delete Tauri commands)

These Tauri commands invoke the Python CLI (same pattern as existing remote commands). Each command calls `forge remote dataset-push/list/pull/delete` via subprocess.

**Step 1: Check existing remote commands structure**

Look at how existing remote commands (list_remote_jobs, sync_remote_job_status, etc.) are implemented in Rust — they invoke the Python CLI and parse output.

**Step 2: Add four Tauri commands**

Each follows the same pattern: construct CLI args, call `run_forge_command()`, parse output.

For `list_remote_datasets`: parse the `FORGE_JSON:` line from stdout to get the dataset list.
For `push_dataset_to_cluster`: invoke `forge remote dataset-push --cluster X --dataset Y`, return success/error.
For `pull_dataset_from_cluster`: invoke `forge remote dataset-pull --cluster X --dataset Y`.
For `delete_remote_dataset`: invoke `forge remote dataset-delete --cluster X --dataset Y`.

**Step 3: Register in `lib.rs`**

Add the four new commands to the `tauri::generate_handler![]` macro.

**Step 4: Verify Rust compiles**

Run: `cd "/Users/tomalmog/programming/Febuary 2026/forge/studio-app/src-tauri" && cargo check`
Expected: Compiles without errors

**Step 5: Commit**

```bash
git add studio-app/src-tauri/
git commit -m "add Tauri commands for remote dataset push, list, pull, delete"
```

---

### Task 5: Add TypeScript API layer for remote datasets

**Files:**
- Modify: `studio-app/src/api/remoteApi.ts` (add 4 new functions)
- Modify: `studio-app/src/types/remote.ts` (add `RemoteDatasetInfo` type)

**Step 1: Add type to `remote.ts`**

```typescript
export interface RemoteDatasetInfo {
  name: string;
  record_count: number;
  version_id: string;
  synced_at: string;
}
```

**Step 2: Add API functions to `remoteApi.ts`**

```typescript
import type { ClusterConfig, RemoteDatasetInfo, RemoteJobRecord } from "../types/remote";

export async function listRemoteDatasets(dataRoot: string, cluster: string): Promise<RemoteDatasetInfo[]> {
  return invoke<RemoteDatasetInfo[]>("list_remote_datasets", { dataRoot, cluster });
}

export async function pushDatasetToCluster(dataRoot: string, cluster: string, dataset: string): Promise<void> {
  return invoke<void>("push_dataset_to_cluster", { dataRoot, cluster, dataset });
}

export async function pullDatasetFromCluster(dataRoot: string, cluster: string, dataset: string): Promise<void> {
  return invoke<void>("pull_dataset_from_cluster", { dataRoot, cluster, dataset });
}

export async function deleteRemoteDataset(dataRoot: string, cluster: string, dataset: string): Promise<void> {
  return invoke<void>("delete_remote_dataset_cmd", { dataRoot, cluster, dataset });
}
```

**Step 3: Verify TypeScript compiles**

Run: `cd "/Users/tomalmog/programming/Febuary 2026/forge/studio-app" && npx tsc --noEmit`

**Step 4: Commit**

```bash
git add studio-app/src/api/remoteApi.ts studio-app/src/types/remote.ts
git commit -m "add TypeScript API layer for remote dataset operations"
```

---

### Task 6: Clean up Datasets UI — remove versions, top sources, diff tab

**Files:**
- Modify: `studio-app/src/pages/datasets/DatasetListPanel.tsx` (remove entire versions section)
- Modify: `studio-app/src/pages/datasets/DatasetDashboard.tsx` (remove Top Sources chart, remove Version metric card)
- Modify: `studio-app/src/pages/datasets/DatasetsPage.tsx` (remove "diff" from tabs)
- Modify: `studio-app/src/context/ForgeContext.tsx` (stop exposing selectedVersion/versions to UI, always pass null)

**Step 1: Strip versions from DatasetListPanel.tsx**

Remove lines 66-92 (the entire versions section: the `<h4>Versions</h4>`, the "Latest" button, and the version list map). Remove `versions`, `selectedVersion`, `setSelectedVersion` from the useForge destructure.

**Step 2: Strip Top Sources and Version from DatasetDashboard.tsx**

Remove the `sourceRows` mapping (lines 18-21). Remove the Top Sources `<div className="panel">` block (lines 39-44). Change the Version MetricCard to show the dataset name instead:
```tsx
<MetricCard label="Dataset" value={dashboard.dataset_name} />
```

**Step 3: Remove "diff" tab from DatasetsPage.tsx**

Change the tab type: `type Tab = "overview" | "samples" | "ingest" | "filter";`
Remove `"diff"` from the tabs array on line 35.
Remove `{tab === "diff" && <VersionDiffPanel />}` on line 47.
Remove the `VersionDiffPanel` import.

**Step 4: Verify TypeScript compiles**

Run: `cd "/Users/tomalmog/programming/Febuary 2026/forge/studio-app" && npx tsc --noEmit`

**Step 5: Commit**

```bash
git add studio-app/src/pages/datasets/
git commit -m "clean up datasets UI: remove versions, top sources, diff tab"
```

---

### Task 7: Add Local/Remote tabs to DatasetListPanel

**Files:**
- Modify: `studio-app/src/pages/datasets/DatasetListPanel.tsx` (add tab UI with Local/Remote, cluster selector, push/pull/delete actions)

**Step 1: Rewrite DatasetListPanel with tabs**

The component gets:
- A `listTab` state: `"local" | "remote"`
- When `"local"`: shows current dataset list with a push icon per row
- When `"remote"`: shows cluster selector dropdown + remote dataset list (fetched via `listRemoteDatasets`) with pull/delete buttons
- Loading state for remote fetches
- Push flow: click upload icon → if multiple clusters, show a picker; otherwise push directly to the only cluster

Key imports to add: `listClusters`, `listRemoteDatasets`, `pushDatasetToCluster`, `pullDatasetFromCluster`, `deleteRemoteDataset` from remoteApi. `Upload`, `Download`, `Trash2`, `Loader2` from lucide-react.

The local tab dataset rows should show: name (clickable, selects dataset), record count from dashboard, push icon button.

The remote tab should show: cluster dropdown at top, then dataset rows with name + record count + synced_at + pull button + delete button (two-step confirm).

**Step 2: Verify TypeScript compiles**

Run: `cd "/Users/tomalmog/programming/Febuary 2026/forge/studio-app" && npx tsc --noEmit`

**Step 3: Commit**

```bash
git add studio-app/src/pages/datasets/DatasetListPanel.tsx
git commit -m "add Local/Remote tabs to dataset list panel with push/pull/delete"
```

---

### Task 8: Update training form — remote dataset selector

**Files:**
- Modify: `studio-app/src/pages/training/ClusterSubmitSection.tsx` (remove Data Strategy selector, add remote dataset dropdown)
- Modify: `studio-app/src/api/commandArgs.ts` (remove `--data-strategy` from `buildRemoteSubmitArgs`, simplify)

**Step 1: Update ClusterSubmitSection**

Remove the "Data Strategy" `<FormField>` (lines 175-185). Remove `dataStrategy` from `ClusterSubmitConfig` interface and `DEFAULT_CLUSTER_CONFIG`.

Add a remote dataset dropdown that's populated via `listRemoteDatasets()` when a cluster is selected. Add a `remoteDataset: string` field to `ClusterSubmitConfig`. The dropdown shows datasets on the selected cluster.

**Step 2: Update `buildRemoteSubmitArgs` in `commandArgs.ts`**

Remove `"--data-strategy", config.dataStrategy` from the args array (line 297). The `dataset_name` should come from the remote dataset selection and be included in the method args JSON.

**Step 3: Verify TypeScript compiles**

Run: `cd "/Users/tomalmog/programming/Febuary 2026/forge/studio-app" && npx tsc --noEmit`

**Step 4: Commit**

```bash
git add studio-app/src/pages/training/ClusterSubmitSection.tsx studio-app/src/api/commandArgs.ts
git commit -m "update training form: remote dataset selector replaces data strategy"
```

---

### Task 9: Remove `DataStrategy` from Python types and clean up

**Files:**
- Modify: `src/core/slurm_types.py` (remove `DataStrategy` type alias)
- Modify: `src/serve/remote_data_upload.py` (remove `DataStrategy` import if still present)

**Step 1: Remove `DataStrategy` from slurm_types.py**

Remove `DataStrategy = Literal["scp", "shared", "s3"]` (line 25).

**Step 2: Grep for any remaining references**

Run: `grep -rn "DataStrategy" src/` and `grep -rn "data_strategy" src/` — fix any remaining references.

Also run: `grep -rn "dataStrategy" studio-app/src/` — remove from TypeScript types too.

**Step 3: Verify everything still imports**

Run: `cd "/Users/tomalmog/programming/Febuary 2026/forge" && PYTHONPATH=src python -c "from serve.remote_job_submitter import submit_remote_job; from serve.remote_data_upload import _upload_bundle; print('OK')"`

Run: `cd "/Users/tomalmog/programming/Febuary 2026/forge/studio-app" && npx tsc --noEmit`

**Step 4: Commit**

```bash
git add -A
git commit -m "remove DataStrategy type and all data_strategy references"
```

---

### Task 10: End-to-end verification

**Step 1: Python imports**

```bash
cd "/Users/tomalmog/programming/Febuary 2026/forge"
PYTHONPATH=src python -c "
from serve.remote_dataset_ops import push_dataset, list_remote_datasets
from serve.remote_job_submitter import submit_remote_job, submit_remote_sweep
from serve.remote_data_upload import _upload_bundle, _upload_config, _upload_script
from cli.remote_command_handlers import _handle_submit
print('All Python imports OK')
"
```

**Step 2: Rust compiles**

```bash
cd "/Users/tomalmog/programming/Febuary 2026/forge/studio-app/src-tauri" && cargo check
```

**Step 3: TypeScript compiles**

```bash
cd "/Users/tomalmog/programming/Febuary 2026/forge/studio-app" && npx tsc --noEmit
```

**Step 4: Run existing tests**

```bash
cd "/Users/tomalmog/programming/Febuary 2026/forge" && PYTHONPATH=src python -m pytest tests/ -x --timeout=30
```

**Step 5: Manual smoke test**

```bash
cd "/Users/tomalmog/programming/Febuary 2026/forge/studio-app" && npm run tauri dev
```

Verify:
- Datasets page loads with clean sidebar (no versions)
- No Top Sources chart in dashboard
- No Diff tab
- Local/Remote tabs visible
- Training form no longer shows Data Strategy

**Step 6: Commit any fixes**

```bash
git add -A
git commit -m "fix any issues found in end-to-end verification"
```
