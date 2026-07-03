import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FunctionCall:
    def __init__(self, function: str, args: dict):
        self.function = function
        self.args = args
        self.id = None


class EmptyEnv:
    pass


class FunctionsRuntime:
    pass


def text_content_block_from_string(content: str) -> dict:
    return {"type": "text", "content": content}


def get_text_content_as_str(content_blocks: list[dict]) -> str:
    return "\n".join(block["content"] for block in content_blocks if block.get("content") is not None)


class BasePipelineElement:
    pass


class Logger:
    @staticmethod
    def get():
        return types.SimpleNamespace(context={}, save=lambda: None)


def install_agentdojo_stubs():
    agentdojo = types.ModuleType("agentdojo")
    agent_pipeline = types.ModuleType("agentdojo.agent_pipeline")
    base_pipeline_element = types.ModuleType("agentdojo.agent_pipeline.base_pipeline_element")
    base_pipeline_element.BasePipelineElement = BasePipelineElement
    logging_module = types.ModuleType("agentdojo.logging")
    logging_module.Logger = Logger
    types_module = types.ModuleType("agentdojo.types")
    types_module.ChatMessage = dict
    types_module.get_text_content_as_str = get_text_content_as_str
    types_module.text_content_block_from_string = text_content_block_from_string
    functions_runtime = types.ModuleType("agentdojo.functions_runtime")
    functions_runtime.EmptyEnv = EmptyEnv
    functions_runtime.Env = EmptyEnv
    functions_runtime.FunctionsRuntime = FunctionsRuntime
    piarena = types.ModuleType("piarena")
    piarena_config = types.ModuleType("piarena.config")
    piarena_config.load_json_env_config = lambda _: None
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    sys.modules.update(
        {
            "agentdojo": agentdojo,
            "agentdojo.agent_pipeline": agent_pipeline,
            "agentdojo.agent_pipeline.base_pipeline_element": base_pipeline_element,
            "agentdojo.logging": logging_module,
            "agentdojo.types": types_module,
            "agentdojo.functions_runtime": functions_runtime,
            "piarena": piarena,
            "piarena.config": piarena_config,
            "torch": torch,
        }
    )


def load_adapter_module():
    install_agentdojo_stubs()
    path = ROOT / "agents" / "agentdojo" / "src" / "agentdojo" / "agent_pipeline" / "piarena_defense_adapter.py"
    spec = importlib.util.spec_from_file_location("piarena_defense_adapter_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeDefense:
    def __init__(self):
        self.calls = []

    def execute(self, target_inst: str, context: str) -> dict:
        self.calls.append((target_inst, context))
        return {
            "detect_flag": "bad" in context,
            "cleaned_context": context.replace("bad", "clean"),
            "predicted_drop_spans": [],
            "input_tokens": 1,
            "latency_ms": 0,
            "error": None,
        }


def tool_message(content: str, *, error: str | None = None):
    return {
        "role": "tool",
        "content": [text_content_block_from_string(content)],
        "tool_call_id": None,
        "tool_call": FunctionCall(function="read_tool", args={}),
        "error": error,
    }


def test_piarena_adapter_sanitizes_all_trailing_tool_messages(monkeypatch):
    module = load_adapter_module()
    defense = FakeDefense()
    adapter = module.PIArenaDefenseAdapter(defense_name="fake", defense_config={})
    monkeypatch.setattr(adapter, "_get_defense", lambda: defense)
    events = []
    monkeypatch.setattr(module, "_record_defense_event", events.append)
    messages = [
        {"role": "user", "content": [text_content_block_from_string("Do the task.")]},
        {"role": "assistant", "content": None, "tool_calls": [FunctionCall(function="read_tool", args={})]},
        tool_message("first bad output"),
        tool_message("second bad output"),
    ]

    _, _, _, updated_messages, _ = adapter.query(
        "Do the task.",
        FunctionsRuntime(),
        EmptyEnv(),
        messages,
        {},
    )

    assert updated_messages[2]["content"][0]["content"] == "first clean output"
    assert updated_messages[3]["content"][0]["content"] == "second clean output"
    assert [call[1] for call in defense.calls] == ["first bad output", "second bad output"]
    assert [event["original_tool_output"] for event in events] == ["first bad output", "second bad output"]


def test_piarena_adapter_sanitizes_tool_error_field(monkeypatch):
    module = load_adapter_module()
    defense = FakeDefense()
    adapter = module.PIArenaDefenseAdapter(defense_name="fake", defense_config={})
    monkeypatch.setattr(adapter, "_get_defense", lambda: defense)
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [FunctionCall(function="read_tool", args={})]},
        tool_message("", error="bad tool failure"),
    ]

    _, _, _, updated_messages, _ = adapter.query(
        "Do the task.",
        FunctionsRuntime(),
        EmptyEnv(),
        messages,
        {},
    )

    assert updated_messages[1]["error"] == "clean tool failure"
    assert updated_messages[1]["content"][0]["content"] == ""
    assert defense.calls == [("Do the task.", "bad tool failure")]
