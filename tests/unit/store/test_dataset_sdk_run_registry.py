"""Unit tests for SDK run-spec and training-run registry helpers."""

from __future__ import annotations

from dataclasses import replace

from core.config import CrucibleConfig
from serve.training_run_registry import TrainingRunRegistry
from store.dataset_sdk import CrucibleClient


def test_client_run_spec_delegates_to_shared_executor(monkeypatch) -> None:
    """CrucibleClient.run_spec should route to shared execution engine."""
    config = replace(CrucibleConfig.from_env())
    client = CrucibleClient(config)

    def _fake_execute(client_arg: CrucibleClient, spec_file: str) -> tuple[str, ...]:
        _ = client_arg
        return (f"executed={spec_file}",)

    monkeypatch.setattr("store.dataset_sdk.execute_run_spec_file", _fake_execute)
    output = client.run_spec("pipeline.yaml")

    assert output == ("executed=pipeline.yaml",)


def test_client_hardware_profile_returns_profile_payload(monkeypatch) -> None:
    """CrucibleClient.hardware_profile should return detect_hardware_profile output."""
    config = replace(CrucibleConfig.from_env())
    client = CrucibleClient(config)

    class _FakeProfile:
        def to_dict(self) -> dict[str, object]:
            return {"accelerator": "cpu", "gpu_count": 0}

    monkeypatch.setattr("store.dataset_sdk.detect_hardware_profile", lambda: _FakeProfile())
    payload = client.hardware_profile()

    assert payload == {"accelerator": "cpu", "gpu_count": 0}


def test_client_run_registry_helpers_load_runs_and_lineage(tmp_path) -> None:
    """SDK helper methods should read run IDs, run records, and lineage graph."""
    config = replace(CrucibleConfig.from_env(), data_root=tmp_path)
    client = CrucibleClient(config)
    registry = TrainingRunRegistry(tmp_path)
    run_record = registry.start_run(
        dataset_name="demo",
        output_dir=str(tmp_path / "out"),
        parent_model_path=None,
        config_hash="abc123",
    )

    assert (
        client.list_training_runs() == (run_record.run_id,)
        and client.get_training_run(run_record.run_id).run_id == run_record.run_id
        and isinstance(client.get_lineage_graph().get("runs"), dict)
    )
