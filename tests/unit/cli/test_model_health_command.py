"""Unit tests for model-health CLI command wiring."""

from __future__ import annotations

import argparse
from typing import cast

from pytest import MonkeyPatch

from cli.model_health_command import run_model_health_command
from serve.model_health_suite import ModelHealthCheckOptions
from store.dataset_sdk import CrucibleClient


def test_targeted_model_health_without_dataset_does_not_load_dataset(
    monkeypatch: MonkeyPatch,
) -> None:
    records_seen: list[list[object]] = []

    def fake_run_suite(
        suite_id: str,
        options: ModelHealthCheckOptions,
        records: list[object],
    ) -> dict[str, object]:
        records_seen.append(records)
        return {"suite": suite_id, "model": options.model_path}

    monkeypatch.setattr("cli.model_health_command.run_model_health_suite", fake_run_suite)

    client = _FakeClient(should_raise_on_dataset=True)
    result = run_model_health_command(
        cast(CrucibleClient, client),
        _args(checks="weight-norms"),
    )

    assert result == 0
    assert records_seen == [[]]
    assert client.dataset_names == []


def test_targeted_model_health_loads_dataset_when_required(
    monkeypatch: MonkeyPatch,
) -> None:
    records_seen: list[list[object]] = []

    def fake_run_suite(
        suite_id: str,
        options: ModelHealthCheckOptions,
        records: list[object],
    ) -> dict[str, object]:
        records_seen.append(records)
        return {"suite": suite_id, "dataset": options.dataset_name}

    monkeypatch.setattr("cli.model_health_command.run_model_health_suite", fake_run_suite)

    client = _FakeClient(should_raise_on_dataset=False)
    result = run_model_health_command(
        cast(CrucibleClient, client),
        _args(dataset="basic", checks="activation-norms"),
    )

    assert result == 0
    assert client.dataset_names == ["basic"]
    assert records_seen == [[{"text": "sample"}]]


def _args(dataset: str = "", checks: str = "") -> argparse.Namespace:
    return argparse.Namespace(
        model_path="/models/candidate.pt",
        dataset=dataset,
        suite="targeted",
        probe_text="",
        clean_text="",
        corrupted_text="",
        label_field="",
        max_samples=2,
        base_model="",
        checks=checks,
        layer_indices="",
        output_dir="./outputs/model-health",
    )


class _FakeDataset:
    def load_records(self) -> tuple[object, list[dict[str, str]]]:
        return object(), [{"text": "sample"}]


class _FakeClient:
    def __init__(self, should_raise_on_dataset: bool) -> None:
        self.dataset_names: list[str] = []
        self._should_raise_on_dataset = should_raise_on_dataset

    def dataset(self, dataset_name: str) -> _FakeDataset:
        self.dataset_names.append(dataset_name)
        if self._should_raise_on_dataset:
            raise AssertionError("dataset should not be loaded")
        return _FakeDataset()
