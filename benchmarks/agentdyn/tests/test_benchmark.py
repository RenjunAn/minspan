import json
from types import SimpleNamespace
from typing import ClassVar

from agentdojo import benchmark
from agentdojo.benchmark import convert_old_messages_format, load_task_results
from agentdojo.logging import OutputLogger


class FakePipeline:
    name = "fake-pipeline"


class FakeSuite:
    name = "fake-suite"
    injection_tasks: ClassVar = {"injection_task_0": SimpleNamespace(ID="injection_task_0")}

    def get_injection_task_by_id(self, injection_task_id):
        return SimpleNamespace(ID=injection_task_id)

    def run_task_with_pipeline(self, agent_pipeline, user_task, injection_task=None, injections=None):
        return True, True


class FakeAttack:
    name = "important_instructions"
    is_dos_attack = False

    def attack(self, user_task, injection_task):
        return {}


def fail_if_loaded(*args, **kwargs):
    raise AssertionError("force_rerun=True should not load existing task results")


def test_task_results_rebuild_uses_function_call_namespace(monkeypatch):
    calls = []

    def fake_model_rebuild(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(benchmark.TaskResults, "model_rebuild", fake_model_rebuild)
    monkeypatch.setattr(benchmark, "_TASK_RESULTS_MODEL_REBUILT", False)

    benchmark._ensure_task_results_model_rebuilt()

    assert calls == [{"_types_namespace": {"FunctionCall": benchmark.FunctionCall}}]


def test_load_task_results_resolves_function_call_forward_refs(tmp_path):
    path = tmp_path / "fake-pipeline" / "fake-suite" / "user_task_0" / "none"
    path.mkdir(parents=True)
    (path / "none.json").write_text(
        json.dumps(
            {
                "suite_name": "fake-suite",
                "pipeline_name": "fake-pipeline",
                "user_task_id": "user_task_0",
                "injection_task_id": None,
                "attack_type": None,
                "injections": {},
                "messages": [
                    {"role": "user", "content": [{"type": "text", "content": "Do it."}]},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"function": "search_product", "args": {"query": "watch"}, "id": "call_1"}],
                    },
                ],
                "error": None,
                "utility": True,
                "security": True,
                "duration": 1.0,
                "agentdojo_package_version": "unknown",
            }
        )
    )

    result = load_task_results("fake-pipeline", "fake-suite", "user_task_0", "none", "none", tmp_path)

    assert result.utility is True
    assert result.messages[1]["tool_calls"][0].function == "search_product"


def test_force_rerun_skips_loading_no_injection_results(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "load_task_results", fail_if_loaded)

    with OutputLogger(str(tmp_path)):
        utility, security = benchmark.run_task_without_injection_tasks(
            FakeSuite(),
            FakePipeline(),
            SimpleNamespace(ID="user_task_0"),
            tmp_path,
            force_rerun=True,
        )

    assert utility is True
    assert security is True


def test_force_rerun_skips_loading_injection_results(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "load_task_results", fail_if_loaded)

    with OutputLogger(str(tmp_path)):
        utility_results, security_results = benchmark.run_task_with_injection_tasks(
            FakeSuite(),
            FakePipeline(),
            SimpleNamespace(ID="user_task_0"),
            FakeAttack(),
            tmp_path,
            force_rerun=True,
            injection_tasks=["injection_task_0"],
        )

    assert utility_results == {("user_task_0", "injection_task_0"): True}
    assert security_results == {("user_task_0", "injection_task_0"): True}


def test_convert_old_messages_format():
    # Test with various message types
    old_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there", "tool_calls": None},
        {
            "role": "tool",
            "content": "123",
            "tool_call": {"function": "add", "args": {"a": 1, "b": 2}},
            "tool_call_id": None,
            "error": None,
        },
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "assistant", "content": None, "tool_calls": [{"function": "multiply", "args": {"a": 3, "b": 4}}]},
    ]

    converted = convert_old_messages_format(old_messages)

    # Check that each message was properly converted
    assert len(converted) == 5

    # User message
    assert converted[0]["role"] == "user"
    assert len(converted[0]["content"]) == 1
    assert converted[0]["content"][0]["type"] == "text"
    assert converted[0]["content"][0]["content"] == "Hello"

    # Assistant message with content
    assert converted[1]["role"] == "assistant"
    assert converted[1]["content"] is not None
    assert len(converted[1]["content"]) == 1
    assert converted[1]["content"][0]["type"] == "text"
    assert converted[1]["content"][0]["content"] == "Hi there"
    assert converted[1]["tool_calls"] is None

    # Tool message
    assert converted[2]["role"] == "tool"
    assert len(converted[2]["content"]) == 1
    assert converted[2]["content"][0]["type"] == "text"
    assert converted[2]["content"][0]["content"] == "123"
    assert converted[2]["tool_call"]["function"] == "add"

    # System message
    assert converted[3]["role"] == "system"
    assert len(converted[3]["content"]) == 1
    assert converted[3]["content"][0]["type"] == "text"
    assert converted[3]["content"][0]["content"] == "You are a helpful assistant"

    # Assistant message with tool calls but no content
    assert converted[4]["role"] == "assistant"
    assert converted[4]["content"] is None
    assert converted[4]["tool_calls"] is not None

    # Check that original messages are not modified
    assert isinstance(old_messages[0]["content"], str)

    # Check return type is list[ChatMessage]
    assert isinstance(converted, list)
    for message in converted:
        assert isinstance(message, dict)
