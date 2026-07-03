import importlib
from types import SimpleNamespace

from openai.types.chat import ChatCompletionMessage

from agentdojo.agent_pipeline import agent_pipeline
from agentdojo.agent_pipeline.llms.openai_llm import (
    OpenAILLM,
    OpenAILLMToolFilter,
    _message_to_openai,
    _openai_to_assistant_message,
)
from agentdojo.models import MODEL_NAMES, MODEL_PROVIDERS, ModelsEnum
from agentdojo.types import ChatSystemMessage, text_content_block_from_string

agent_pipeline._add_repo_to_syspath("DRIFT")
drift_client = importlib.import_module("client")
DRIFTLLM = importlib.import_module("DRIFTLLM").DRIFTLLM


class FakeLogger:
    def info(self, message):
        pass


class FailingDriftClient:
    def __init__(self, *args, **kwargs):
        raise AssertionError("DeepSeek DRIFT models must not use this client")


def test_deepseek_v4_flash_is_registered():
    assert ModelsEnum.DEEPSEEK_V4_FLASH == "deepseek-v4-flash"
    assert MODEL_PROVIDERS[ModelsEnum.DEEPSEEK_V4_FLASH] == "deepseek"
    assert MODEL_NAMES["deepseek-v4-flash"] == "DeepSeek"


def test_deepseek_v4_pro_is_registered():
    assert ModelsEnum.DEEPSEEK_V4_PRO == "deepseek-v4-pro"
    assert MODEL_PROVIDERS[ModelsEnum.DEEPSEEK_V4_PRO] == "deepseek"
    assert MODEL_NAMES["deepseek-v4-pro"] == "DeepSeek"


def test_deepseek_provider_uses_deepseek_api(monkeypatch):
    created_client_kwargs = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            created_client_kwargs.update(kwargs)

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(agent_pipeline.openai, "OpenAI", FakeOpenAI)

    llm = agent_pipeline.get_llm("deepseek", "deepseek-v4-flash", None, "tool")

    assert isinstance(llm, OpenAILLM)
    assert llm.model == "deepseek-v4-flash"
    assert created_client_kwargs == {
        "api_key": "sk-test",
        "base_url": "https://api.deepseek.com",
    }


def test_deepseek_provider_accepts_v4_pro(monkeypatch):
    class FakeOpenAI:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(agent_pipeline.openai, "OpenAI", FakeOpenAI)

    llm = agent_pipeline.get_llm("deepseek", "deepseek-v4-pro", None, "tool")

    assert isinstance(llm, OpenAILLM)
    assert llm.model == "deepseek-v4-pro"


def test_deepseek_provider_requires_deepseek_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    try:
        agent_pipeline.get_llm("deepseek", "deepseek-v4-flash", None, "tool")
    except ValueError as e:
        assert "DEEPSEEK_API_KEY" in str(e)
    else:
        raise AssertionError("Expected missing DeepSeek API key to fail")


def test_deepseek_system_message_uses_system_role():
    message = ChatSystemMessage(
        role="system",
        content=[text_content_block_from_string("Use tools carefully.")],
    )

    converted = _message_to_openai(message, "deepseek-v4-flash")

    assert converted["role"] == "system"


def test_deepseek_v4_pro_system_message_uses_system_role():
    message = ChatSystemMessage(
        role="system",
        content=[text_content_block_from_string("Use tools carefully.")],
    )

    converted = _message_to_openai(message, "deepseek-v4-pro")

    assert converted["role"] == "system"


def test_deepseek_preserves_reasoning_content_from_response():
    message = ChatCompletionMessage.model_construct(
        role="assistant",
        content="I'll call a tool.",
        reasoning_content="The task needs current page data.",
    )

    converted = _openai_to_assistant_message(message)

    assert converted.get("reasoning_content") == "The task needs current page data."


def test_deepseek_replays_reasoning_content_in_assistant_messages():
    message = _openai_to_assistant_message(
        ChatCompletionMessage.model_construct(
            role="assistant",
            content="I'll call a tool.",
            reasoning_content="The task needs current page data.",
        )
    )

    converted = _message_to_openai(message, "deepseek-v4-flash")

    assert dict(converted).get("reasoning_content") == "The task needs current page data."


def test_deepseek_v4_pro_replays_reasoning_content_in_assistant_messages():
    message = _openai_to_assistant_message(
        ChatCompletionMessage.model_construct(
            role="assistant",
            content="I'll call a tool.",
            reasoning_content="The task needs current page data.",
        )
    )

    converted = _message_to_openai(message, "deepseek-v4-pro")

    assert dict(converted).get("reasoning_content") == "The task needs current page data."


def test_deepseek_tool_filter_disables_thinking_for_defense_call(monkeypatch):
    captured_request = {}

    class FakeMessage:
        content = "safe_tool"
        tool_calls = None
        reasoning_content = None

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        def __init__(self):
            self.choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            captured_request.update(kwargs)
            return FakeResponse()

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    tool_filter = OpenAILLMToolFilter("Pick tools.", client, "deepseek-v4-pro")
    runtime = SimpleNamespace(functions={}, update_functions=lambda tools: None)

    tool_filter.query("Do it.", runtime, messages=[])

    assert captured_request["extra_body"] == {"thinking": {"type": "disabled"}}


def test_drift_deepseek_client_uses_deepseek_api(monkeypatch):
    created_client_kwargs = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            created_client_kwargs.update(kwargs)

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(drift_client.openai, "OpenAI", FakeOpenAI)

    client = drift_client.DeepSeekModel(model="deepseek-v4-flash", logger=FakeLogger())

    assert client.model == "deepseek-v4-flash"
    assert client.label == "DeepSeek"
    assert created_client_kwargs == {
        "api_key": "sk-test",
        "base_url": "https://api.deepseek.com",
    }


def test_drift_client_factory_routes_deepseek_to_deepseek(monkeypatch):
    created_model = {}

    class FakeDeepSeekModel:
        def __init__(self, model, logger):
            created_model["model"] = model
            self.logger = logger

    monkeypatch.setattr(drift_client, "DeepSeekModel", FakeDeepSeekModel, raising=False)
    monkeypatch.setattr(drift_client, "OpenAIModel", FailingDriftClient)
    monkeypatch.setattr(drift_client, "GoogleModel", FailingDriftClient)
    monkeypatch.setattr(drift_client, "OpenRouterModel", FailingDriftClient)

    client = agent_pipeline._make_drift_client("deepseek-v4-flash", FakeLogger())

    assert isinstance(client, FakeDeepSeekModel)
    assert created_model == {"model": "deepseek-v4-flash"}


def test_drift_client_factory_routes_deepseek_v4_pro_to_deepseek(monkeypatch):
    created_model = {}

    class FakeDeepSeekModel:
        def __init__(self, model, logger):
            created_model["model"] = model
            self.logger = logger

    monkeypatch.setattr(drift_client, "DeepSeekModel", FakeDeepSeekModel, raising=False)
    monkeypatch.setattr(drift_client, "OpenAIModel", FailingDriftClient)
    monkeypatch.setattr(drift_client, "GoogleModel", FailingDriftClient)
    monkeypatch.setattr(drift_client, "OpenRouterModel", FailingDriftClient)

    client = agent_pipeline._make_drift_client("deepseek-v4-pro", FakeLogger())

    assert isinstance(client, FakeDeepSeekModel)
    assert created_model == {"model": "deepseek-v4-pro"}


def test_drift_deepseek_client_captures_reasoning_content(monkeypatch):
    captured_request = {}

    class FakeUsage:
        completion_tokens = 1
        prompt_tokens = 2
        total_tokens = 3

    class FakeMessage:
        content = "<final_answer>done</final_answer>"
        reasoning_content = "private model reasoning"

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        def __init__(self):
            self.choices = [FakeChoice()]
            self.usage = FakeUsage()

    class FakeCompletions:
        def create(self, **kwargs):
            captured_request.update(kwargs)
            return FakeResponse()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(drift_client.openai, "OpenAI", FakeOpenAI)

    client = drift_client.DeepSeekModel(model="deepseek-v4-flash", logger=FakeLogger())

    completion = client.agent_run([{"role": "system", "content": "Answer."}])

    assert completion == ["<final_answer>done</final_answer>"]
    assert client.last_reasoning_content == "private model reasoning"
    assert "max_tokens" in captured_request
    assert "max_completion_tokens" not in captured_request
    assert "extra_body" not in captured_request


def test_drift_deepseek_defense_llm_run_disables_thinking(monkeypatch):
    captured_request = {}

    class FakeUsage:
        completion_tokens = 1
        prompt_tokens = 2
        total_tokens = 3

    class FakeMessage:
        content = "A"
        reasoning_content = None

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        def __init__(self):
            self.choices = [FakeChoice()]
            self.usage = FakeUsage()

    class FakeCompletions:
        def create(self, **kwargs):
            captured_request.update(kwargs)
            return FakeResponse()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(drift_client.openai, "OpenAI", FakeOpenAI)

    client = drift_client.DeepSeekModel(model="deepseek-v4-flash", logger=FakeLogger())

    assert client.llm_run("Classify.", "Tool description.") == "A"
    assert captured_request["extra_body"] == {"thinking": {"type": "disabled"}}


def test_drift_deepseek_agent_run_can_disable_thinking_for_defense_planning(monkeypatch):
    captured_request = {}

    class FakeUsage:
        completion_tokens = 1
        prompt_tokens = 2
        total_tokens = 3

    class FakeMessage:
        content = "<function_trajectory>[]</function_trajectory>"
        reasoning_content = None

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        def __init__(self):
            self.choices = [FakeChoice()]
            self.usage = FakeUsage()

    class FakeCompletions:
        def create(self, **kwargs):
            captured_request.update(kwargs)
            return FakeResponse()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(drift_client.openai, "OpenAI", FakeOpenAI)

    client = drift_client.DeepSeekModel(model="deepseek-v4-flash", logger=FakeLogger())
    client.agent_run([{"role": "system", "content": "Build constraints."}], thinking_type="disabled")

    assert captured_request["extra_body"] == {"thinking": {"type": "disabled"}}


def test_drift_constraint_build_disables_thinking_but_tool_reasoning_uses_default():
    calls = []

    class FakePlanningClient:
        supports_thinking_control = True
        last_reasoning_content = None

        def agent_run(self, messages, tools=None, **kwargs):
            calls.append(kwargs.get("thinking_type"))
            if len(calls) == 1:
                return ["<function_trajectory>[]</function_trajectory>"]
            return ["<final_answer>done</final_answer>"]

    drift_llm = DRIFTLLM(
        SimpleNamespace(dynamic_validation=False, build_constraints=True, injection_isolation=False),
        client=FakePlanningClient(),
        logger=FakeLogger(),
    )
    runtime = SimpleNamespace(functions={})

    drift_llm.query(
        "Do it.",
        runtime,
        None,
        [{"role": "user", "content": "Do it."}],
        {},
    )

    assert calls == ["disabled", None]


def test_drift_query_adds_client_reasoning_content_to_assistant_message():
    class FakeReasoningClient:
        last_reasoning_content = "private model reasoning"

        def agent_run(self, *args, **kwargs):
            return ["<final_answer>done</final_answer>"]

    drift_llm = DRIFTLLM(
        SimpleNamespace(dynamic_validation=False, build_constraints=False, injection_isolation=False),
        client=FakeReasoningClient(),
        logger=FakeLogger(),
    )
    runtime = SimpleNamespace(functions={})

    _, _, _, messages, _ = drift_llm.query(
        "Do it.",
        runtime,
        None,
        [{"role": "user", "content": "Do it."}],
        {},
    )

    assert messages[-1]["reasoning_content"] == "private model reasoning"


def test_drift_replays_reasoning_content_in_assistant_messages():
    drift_llm = DRIFTLLM(
        SimpleNamespace(dynamic_validation=False, build_constraints=False, injection_isolation=False),
        client=object(),
        logger=FakeLogger(),
    )

    converted = drift_llm._message_to_sharegpt(
        {
            "role": "assistant",
            "content": "<final_answer>done</final_answer>",
            "tool_calls": [],
            "reasoning_content": "private model reasoning",
        }
    )

    assert converted["reasoning_content"] == "private model reasoning"
