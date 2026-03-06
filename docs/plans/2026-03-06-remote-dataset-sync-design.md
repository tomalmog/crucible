# Remote Dataset Sync + UI Cleanup

**Date**: 2026-03-06
**Status**: Approved

## Problem

Local and remote training have a disconnect: datasets only exist locally, and the remote job submission flow has to upload them every time (3 different strategies, lots of complexity). The Datasets UI also has clutter — version lists with opaque hashes, a Top Sources chart that isn't useful.

## Design

### Remote Dataset Storage

Persistent dataset library on each cluster at `{remote_workspace}/datasets/{name}/`:

```
{remote_workspace}/datasets/{name}/
├── metadata.json    # {name, record_count, version_id, synced_at}
└── records.jsonl    # flat records, latest version only
```

Datasets must be explicitly pushed to a cluster before training. No auto-upload.

### New Module: `src/serve/remote_dataset_ops.py`

Four operations, all via SSH:

- `push_dataset(cluster, dataset_name, data_root)` — tar latest snapshot, SCP to cluster
- `list_remote_datasets(cluster)` — SSH ls + read metadata.json per dataset
- `pull_remote_dataset(cluster, dataset_name, data_root)` — SCP back, ingest locally
- `delete_remote_dataset(cluster, dataset_name)` — SSH rm -rf

### Job Submission Simplification

- Remove entire data upload phase from remote job submission
- Remote jobs take a dataset **name**, not a path
- On submit: SSH check that dataset exists on cluster, fail if not ("Push it first")
- Training config points remote agent at `{remote_workspace}/datasets/{name}/records.jsonl`

### Datasets Page UI

**Sidebar**: Two tabs — `[Local]` and `[Remote ▼ cluster]`

**Local tab**:
- Dataset names + record count (always latest version)
- Push icon per row (cluster picker if multiple clusters)
- No version list, no version picker

**Remote tab**:
- Cluster selector dropdown
- Dataset names + record count + synced_at (fetched via SSH on tab switch)
- Pull and Delete buttons per row
- Loading spinner during SSH fetch

### Dashboard Changes

- Remove Top Sources chart
- Remove Diff tab
- Remove version ID from stats
- Remaining tabs: Overview, Samples, Ingest, Filter

### Training Form

- Local training: `DatasetSelect` dropdown shows local dataset names
- Remote training: `DatasetSelect` dropdown shows remote dataset names (SSH-fetched)
- No raw path input anywhere

### What Gets Deleted

- `_handle_data_strategy()`, `_upload_raw_dataset()`, `_upload_dataset_catalog()` from remote data upload
- Version picker + version list from DatasetListPanel
- Diff tab/panel
- Top Sources chart from DatasetDashboard
- `selectedVersion` UI usage in ForgeContext (field stays, always null)

## Future

- Download from HuggingFace directly to cluster (another way to populate `{remote_workspace}/datasets/`)
- Models page gets the same local/remote split treatment
