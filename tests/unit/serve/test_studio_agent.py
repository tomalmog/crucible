"""Unit tests for Studio agent control-tag parsing."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from serve import agent_backends, studio_agent


def test_medical_demo_request_accepts_explicit_dataset_and_benchmark() -> None:
    """The original precise demo prompt should keep routing to the workflow."""
    prompt = (
        "My medical-assistant model is failing medical safety triage. "
        "Fine-tune it on medical_safety_triage_training, run "
        "medical_safety_triage before and after, and show me the improvement."
    )

    assert studio_agent.is_medical_safety_demo_request(prompt)


def test_medical_demo_request_infers_relevant_benchmark() -> None:
    """Users should not need to name the demo benchmark or training dataset."""
    prompt = "my medical-assistant model is failing on relevant benchmarks. can you improve it."

    assert studio_agent.is_medical_safety_demo_request(prompt)


def test_medical_demo_request_ignores_non_action_model_mentions() -> None:
    """A model mention alone should not hijack ordinary chat or lookup requests."""
    prompt = "what is my medical-assistant model?"

    assert not studio_agent.is_medical_safety_demo_request(prompt)


def test_medical_demo_workflow_uses_live_tool_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The routed workflow should build artifacts from fresh tool outputs."""
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_execute_tool(name: str, tool_input: dict[str, object]) -> str:
        calls.append((name, tool_input))
        if name == "train":
            return json.dumps({
                "job_id": "job-live-train",
                "model_path": "/tmp/live-trained.pt",
                "status": "completed",
            })
        if name == "run_benchmark" and tool_input["model_path"] == "/tmp/live-trained.pt":
            return json.dumps({
                "job_id": "job-live-after",
                "average_score": 91.0,
                "benchmarks": [
                    {"name": "medical_safety_triage", "score": 91.0, "correct": 91, "total": 100},
                ],
            })
        return json.dumps({
            "job_id": "job-live-before",
            "average_score": 27.0,
            "benchmarks": [
                {"name": "medical_safety_triage", "score": 27.0, "correct": 27, "total": 100},
            ],
        })

    monkeypatch.setattr(studio_agent, "execute_tool", fake_execute_tool)
    monkeypatch.setattr(
        studio_agent,
        "_resolve_registered_model_path",
        lambda _root, _name: "/tmp/live-base-model",
    )

    result = studio_agent._maybe_run_medical_safety_demo(
        conversation_path=tmp_path / "conversation.json",
        user_message="my medical-assistant model is failing on relevant benchmarks. can you improve it.",
        data_root=str(tmp_path),
        event_sink=None,
    )

    assert result is not None
    assert [name for name, _input in calls] == ["run_benchmark", "train", "run_benchmark"]
    assert result["artifact_messages"][0]["artifact"]["jobId"] == "job-live-before"
    assert result["artifact_messages"][1]["artifact"]["jobId"] == "job-live-after"


def test_run_agent_turn_returns_workspace_directives(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Agent turns should surface workspace directives and strip them from text."""
    monkeypatch.setattr(studio_agent, "_get_tools", lambda: ([], {}))
    monkeypatch.setattr(studio_agent, "_build_system_prompt", lambda _ctx, _root: "system")

    fake_mcp_server = types.ModuleType("serve.mcp_server")
    fake_mcp_server._ensure_backends = lambda: None
    monkeypatch.setitem(sys.modules, "serve.mcp_server", fake_mcp_server)

    def fake_call_anthropic(
        _api_key: str,
        _model: str,
        _system: str,
        _messages: list[dict[str, object]],
        _tools: list[dict[str, object]],
    ) -> agent_backends.LlmResponse:
        return agent_backends.LlmResponse(
            [
                {
                    "type": "text",
                    "text": (
                        "I set up the build workspace.\n"
                        "<workspace_mode>plan</workspace_mode>\n"
                        "<workspace_cards>\n"
                        "- artifact\n"
                        "trace, context\n"
                        "</workspace_cards>\n"
                        "<navigate_to>/build</navigate_to>"
                    ),
                },
            ],
            "end_turn",
        )

    monkeypatch.setattr(agent_backends, "call_anthropic", fake_call_anthropic)

    result = studio_agent.run_agent_turn(
        conversation_path=tmp_path / "conversation.json",
        user_message="Set up the workspace",
        app_context={},
        api_key="test-key",
        data_root=str(tmp_path),
    )

    assert result == {
        "role": "assistant",
        "content": "I set up the build workspace.",
        "tools_used": [],
        "navigate_to": "/build",
        "workspace_mode": "plan",
        "workspace_cards": ["artifact", "trace", "context"],
    }


def test_load_conversation_for_display_strips_persisted_control_tags(
    tmp_path: Path,
) -> None:
    """Reloaded assistant text should not leak control tags into the UI."""
    conversation_path = tmp_path / "conversation.json"
    conversation_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Help me organize the build page"},
                    {
                        "role": "assistant",
                        "content": (
                            "Here is the plan.\n"
                            "<workspace_mode>plan</workspace_mode>\n"
                            "<workspace_cards>\nartifact\ntrace\n</workspace_cards>\n"
                            "<navigate_to>/build</navigate_to>"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Queued next step.\n"
                                    "<pending_chain>\nWait\n</pending_chain>"
                                ),
                            },
                            {"type": "tool_use", "name": "submit_remote_training"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": "<script_update>print('hidden')</script_update>",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    display = studio_agent.load_conversation_for_display(conversation_path)

    assert display == [
        {"role": "user", "content": "Help me organize the build page"},
        {"role": "assistant", "content": "Here is the plan."},
        {
            "role": "assistant",
            "content": "Queued next step.",
            "tools_used": ["submit_remote_training"],
        },
    ]


def test_remote_submit_tools_expose_full_resource_schema() -> None:
    """Agent tool schemas should include the same core Slurm knobs as Studio."""
    tools, _ = studio_agent._build_tool_definitions()
    by_name = {tool["name"]: tool for tool in tools}
    resource_fields = {"partition", "nodes", "gpus_per_node", "cpus_per_task", "gpu_type"}

    for tool_name in (
        "submit_remote_training",
        "submit_remote_eval",
        "submit_remote_interp",
        "submit_remote_sweep",
    ):
        schema = by_name[tool_name]["input_schema"]
        assert resource_fields.issubset(set(schema["properties"]))
