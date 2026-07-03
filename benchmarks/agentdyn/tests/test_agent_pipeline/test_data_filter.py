import json
from typing import cast

from pydantic import BaseModel

from agentdojo.agent_pipeline import agent_pipeline as agent_pipeline_module
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.data_filter import DataFilterDefense
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor
from agentdojo.functions_runtime import EmptyEnv, FunctionCall, FunctionsRuntime
from agentdojo.logging import Logger
from agentdojo.types import ChatToolResultMessage, text_content_block_from_string


class FakeGeneratedText:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeGeneration:
    def __init__(self, text: str) -> None:
        self.outputs = [FakeGeneratedText(text)]


class FakeFilterModel:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.sampling_params = []

    def generate(self, prompts, sampling_params):
        self.prompts.extend(prompts)
        self.sampling_params.append(sampling_params)
        return [FakeGeneration(_extract_data(prompt).replace(" MALICIOUS", "")) for prompt in prompts]


class FailingFilterModel:
    def generate(self, prompts, sampling_params):
        raise RuntimeError("vllm failed")


class FakeEventLogger(Logger):
    def __init__(self) -> None:
        self.context = {}
        self.saved = 0

    def save(self):
        self.saved += 1

    def log(self, *args, **kwargs):
        pass

    def log_error(self, *args, **kwargs):
        pass


class FakeLLM(BasePipelineElement):
    name = "deepseek-v4-flash"

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        return query, runtime, env, messages, extra_args


class FakeToolResult(BaseModel):
    value: str


def test_data_filter_passes_optional_vllm_engine_kwargs():
    seen_kwargs = {}

    class CapturingLLM:
        def __init__(self, **kwargs) -> None:
            seen_kwargs.update(kwargs)

    DataFilterDefense(
        llm_cls=CapturingLLM,
        model_path="local-datafilter",
        tensor_parallel_size=2,
        dtype="float16",
        max_model_len=32768,
        gpu_memory_utilization=0.9,
    )

    assert seen_kwargs == {
        "model": "local-datafilter",
        "tensor_parallel_size": 2,
        "dtype": "float16",
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.9,
    }


def test_make_data_filter_defense_reads_optional_vllm_engine_env(monkeypatch):
    seen_kwargs = {}

    class CapturingDataFilterDefense:
        def __init__(self, **kwargs) -> None:
            seen_kwargs.update(kwargs)

    monkeypatch.setattr(agent_pipeline_module, "DataFilterDefense", CapturingDataFilterDefense)
    monkeypatch.setenv("DATAFILTER_MAX_MODEL_LEN", "32768")
    monkeypatch.setenv("DATAFILTER_GPU_MEMORY_UTILIZATION", "0.9")

    agent_pipeline_module._make_data_filter_defense()

    assert seen_kwargs["max_model_len"] == 32768
    assert seen_kwargs["gpu_memory_utilization"] == 0.9


def test_data_filter_recursively_filters_all_tail_tool_messages_and_logs_events():
    filter_model = FakeFilterModel()
    defense = DataFilterDefense(filter_model=filter_model)
    old_tool = _tool_message("old_call", "read_old", json.dumps({"body": "old MALICIOUS"}))
    assistant_message = {"role": "assistant", "content": None, "tool_calls": []}
    current_tool_1 = _tool_message("call_1", "read_email", json.dumps({"body": "hello MALICIOUS"}))
    current_tool_2 = _tool_message("call_2", "read_file", json.dumps({"note": "keep MALICIOUS"}))
    messages = [old_tool, assistant_message, current_tool_1, current_tool_2]
    event_logger = FakeEventLogger()

    with event_logger:
        _, _, _, output_messages, _ = defense.query(
            "Summarize the current tool results.",
            FunctionsRuntime(),
            EmptyEnv(),
            messages,
            {},
        )

    assert output_messages[0]["content"] == [text_content_block_from_string(json.dumps({"body": "old MALICIOUS"}))]
    output_tool_1 = cast(ChatToolResultMessage, output_messages[2])
    output_tool_2 = cast(ChatToolResultMessage, output_messages[3])
    assert json.loads(output_tool_1["content"][0]["content"]) == {"body": "hello"}
    assert json.loads(output_tool_2["content"][0]["content"]) == {"note": "keep"}
    assert len(filter_model.prompts) == 2
    assert all(
        "Summarize the current tool results. <|end_of_instruction|>" in prompt for prompt in filter_model.prompts
    )

    events = event_logger.context["data_filter_events"]
    assert len(events) == 2
    assert events[0]["tool_name"] == "read_email"
    assert events[0]["tool_call_id"] == "call_1"
    assert events[0]["original_tool_output"] == json.dumps({"body": "hello MALICIOUS"})
    assert json.loads(events[0]["filtered_tool_output"]) == {"body": "hello"}
    assert events[0]["changed"] is True
    assert events[0]["api_ok"] is True
    assert events[0]["json_parse_ok"] is True
    assert events[0]["filter_mode"] == "json_recursive"
    assert event_logger.saved == 2


def test_data_filter_falls_back_to_raw_text_when_tool_output_is_not_json():
    filter_model = FakeFilterModel()
    defense = DataFilterDefense(filter_model=filter_model)
    messages = [_tool_message("call_1", "search_web", "plain result MALICIOUS")]
    event_logger = FakeEventLogger()

    with event_logger:
        _, _, _, output_messages, _ = defense.query("Read the result.", FunctionsRuntime(), EmptyEnv(), messages, {})

    assert output_messages[0]["content"] == [text_content_block_from_string("plain result")]
    event = event_logger.context["data_filter_events"][0]
    assert event["api_ok"] is True
    assert event["json_parse_ok"] is False
    assert event["filter_mode"] == "raw_text"


def test_data_filter_fail_open_on_model_error_and_logs_failure():
    defense = DataFilterDefense(filter_model=FailingFilterModel())
    messages = [_tool_message("call_1", "search_web", "plain result MALICIOUS")]
    event_logger = FakeEventLogger()

    with event_logger:
        _, _, _, output_messages, _ = defense.query("Read the result.", FunctionsRuntime(), EmptyEnv(), messages, {})

    assert output_messages[0]["content"] == [text_content_block_from_string("plain result MALICIOUS")]
    event = event_logger.context["data_filter_events"][0]
    assert event["api_ok"] is False
    assert event["changed"] is False
    assert event["error"] == "vllm failed"


def test_data_filter_pipeline_registration_uses_json_tool_output(monkeypatch):
    monkeypatch.setattr(
        "agentdojo.agent_pipeline.agent_pipeline._make_data_filter_defense",
        lambda: DataFilterDefense(filter_model=FakeFilterModel()),
    )

    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm=FakeLLM(),
            model_id=None,
            defense="data_filter",
            system_message="system",
            system_message_name=None,
            tool_delimiter="tool",
        )
    )

    tools_loop = list(pipeline.elements)[3]
    assert pipeline.name == "deepseek-v4-flash-data_filter"
    assert isinstance(tools_loop, ToolsExecutionLoop)
    assert isinstance(tools_loop.elements[1], DataFilterDefense)
    tools_executor = cast(ToolsExecutor, tools_loop.elements[0])
    assert tools_executor.output_formatter(FakeToolResult(value="hello")) == '{"value": "hello"}'


def _tool_message(call_id: str, function: str, content: str) -> ChatToolResultMessage:
    tool_call = FunctionCall(function=function, args={}, id=call_id)
    return ChatToolResultMessage(
        role="tool",
        content=[text_content_block_from_string(content)],
        tool_call_id=call_id,
        tool_call=tool_call,
        error=None,
    )


def _extract_data(prompt: str) -> str:
    return prompt.split("<|end_of_instruction|> ", 1)[1].split("\n<|eot_id|>", 1)[0]
