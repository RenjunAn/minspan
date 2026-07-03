import json
from types import SimpleNamespace
from typing import ClassVar

import pytest
from pydantic import BaseModel

from agentdojo.agent_pipeline import agent_pipeline as agent_pipeline_module
from agentdojo.agent_pipeline import token_tagger as token_tagger_module
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.openai_llm import _message_to_openai
from agentdojo.agent_pipeline.token_tagger import (
    DATAFILTER_SYSTEM_PROMPT,
    TaggerBatchError,
    TaggerPrediction,
    TokenTaggerBackend,
    ToolOutputTaggerDefense,
    _checkpoint_fingerprint,
    build_datafilter_tagger_prompt,
    build_encoder_tagger_prompt,
    predictions_to_data_spans,
    reconstruct_without_spans,
)
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor
from agentdojo.functions_runtime import EmptyEnv, FunctionCall, FunctionsRuntime
from agentdojo.logging import Logger
from agentdojo.types import (
    ChatToolResultMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)


class FakeBackend:
    backend_type = "fake"
    architecture: ClassVar[dict[str, str]] = {"model_type": "fake"}
    checkpoint_fingerprint = "0123456789abcdef"

    def __init__(self, predictions: list[TaggerPrediction]) -> None:
        self.predictions = predictions
        self.calls: list[tuple[list[str], list[str]]] = []

    def sanitize_batch(
        self,
        instructions: list[str],
        tool_outputs: list[str],
    ) -> list[TaggerPrediction]:
        self.calls.append((list(instructions), list(tool_outputs)))
        return self.predictions


class FailingBackend:
    backend_type = "fake"

    def sanitize_batch(self, instructions, tool_outputs):
        raise RuntimeError("inference failed")


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

    def query(
        self,
        query,
        runtime,
        env=EmptyEnv(),
        messages=[],
        extra_args={},
    ):
        return query, runtime, env, messages, extra_args


class FakeToolResult(BaseModel):
    value: str


def test_reconstruct_without_spans_returns_original():
    assert reconstruct_without_spans("abc", []) == "abc"


def test_reconstruct_without_spans_merges_overlaps():
    assert (
        reconstruct_without_spans(
            "safe MALICIOUS tail",
            [{"start": 5, "end": 10}, {"start": 9, "end": 14}],
        )
        == "safe  tail"
    )


def test_defense_batches_complete_json_tool_outputs():
    backend = FakeBackend(
        [
            TaggerPrediction(
                filtered_tool_output="safe ",
                predicted_drop_spans=[{"start": 5, "end": 14}],
                input_tokens=8,
            ),
            TaggerPrediction(
                filtered_tool_output='{"body": "ok "}',
                predicted_drop_spans=[{"start": 13, "end": 22}],
                input_tokens=12,
            ),
        ]
    )
    defense = ToolOutputTaggerDefense(
        defense_name="modernbert_tagger",
        checkpoint_path="/models/mb",
        backend=backend,
    )
    messages = [
        _tool_message("1", "read_file", "safe MALICIOUS"),
        _tool_message("2", "read_email", '{"body": "ok MALICIOUS"}'),
    ]

    output_messages = defense.query(
        "Summarize.",
        FunctionsRuntime(),
        EmptyEnv(),
        messages,
        {},
    )[3]

    assert backend.calls == [
        (
            ["Summarize.", "Summarize."],
            ["safe MALICIOUS", '{"body": "ok MALICIOUS"}'],
        )
    ]
    assert get_text_content_as_str(output_messages[0]["content"]) == "safe "
    assert get_text_content_as_str(output_messages[1]["content"]) == '{"body": "ok "}'


def test_defense_sanitizes_tool_error_used_by_openai_adapter():
    backend = FakeBackend(
        [
            TaggerPrediction(
                filtered_tool_output="sanitized error",
                predicted_drop_spans=[{"start": 0, "end": 9}],
                input_tokens=4,
            )
        ]
    )
    defense = ToolOutputTaggerDefense(
        defense_name="modernbert_tagger",
        checkpoint_path="/models/mb",
        backend=backend,
    )
    message = _tool_message(
        "1",
        "read_file",
        "",
        error="MALICIOUS error instructions",
    )
    event_logger = FakeEventLogger()

    with event_logger:
        output_message = defense.query(
            "Read.",
            FunctionsRuntime(),
            EmptyEnv(),
            [message],
            {},
        )[3][0]

    assert backend.calls == [(["Read."], ["MALICIOUS error instructions"])]
    assert output_message["error"] == "sanitized error"
    assert _message_to_openai(output_message, "deepseek-v4-flash")["content"] == "sanitized error"
    assert event_logger.context["tagger_defense_events"][0]["input_field"] == "error"


def test_defense_uses_safe_placeholder_when_tool_error_is_fully_removed():
    backend = FakeBackend(
        [
            TaggerPrediction(
                filtered_tool_output="",
                predicted_drop_spans=[{"start": 0, "end": 28}],
                input_tokens=4,
            )
        ]
    )
    defense = ToolOutputTaggerDefense(
        defense_name="modernbert_tagger",
        checkpoint_path="/models/mb",
        backend=backend,
    )
    message = _tool_message(
        "1",
        "read_file",
        "UNFILTERED CONTENT",
        error="MALICIOUS error instructions",
    )

    output_message = defense.query(
        "Read.",
        FunctionsRuntime(),
        EmptyEnv(),
        [message],
        {},
    )[3][0]

    assert output_message["error"] == "[tool error removed by token tagger]"
    assert _message_to_openai(output_message, "deepseek-v4-flash")["content"] == (
        "[tool error removed by token tagger]"
    )


def test_datafilter_prompt_marks_only_complete_tool_output_as_data():
    prompt = build_datafilter_tagger_prompt("Summarize.", '{"body":"x"}')

    assert prompt.text[prompt.data_start : prompt.data_end] == '{"body":"x"}'
    assert DATAFILTER_SYSTEM_PROMPT in prompt.text


def test_encoder_prompt_has_no_llama_chat_tokens():
    prompt = build_encoder_tagger_prompt("Summarize.", '{"body":"x"}')

    assert prompt.text == 'Summarize. <|end_of_instruction|> {"body":"x"}'
    assert prompt.text[prompt.data_start : prompt.data_end] == '{"body":"x"}'


def test_predictions_to_spans_clips_tokens_to_data_region():
    spans = predictions_to_data_spans(
        predictions=[1, 1, 0],
        offsets=[(0, 6), (5, 12), (12, 16)],
        data_start=8,
        data_end=16,
    )

    assert spans == [{"start": 0, "end": 4}]


def test_encoder_checkpoint_loader_uses_complete_saved_model(monkeypatch, tmp_path):
    (tmp_path / "tagger_config.json").write_text(
        json.dumps(_valid_encoder_config()),
        encoding="utf-8",
    )
    tokenizer = FakeTokenizer()
    model = FakeModel()
    tokenizer_loader = FakePretrainedLoader(tokenizer)
    model_loader = FakePretrainedLoader(model)
    monkeypatch.setattr(token_tagger_module, "AutoTokenizer", tokenizer_loader)
    monkeypatch.setattr(
        token_tagger_module,
        "AutoModelForTokenClassification",
        model_loader,
    )

    backend = TokenTaggerBackend.from_encoder_checkpoint(
        checkpoint_path=str(tmp_path),
        device="cpu",
        batch_size=4,
    )

    assert tokenizer_loader.calls == [((str(tmp_path),), {"use_fast": True})]
    assert model_loader.calls == [((str(tmp_path),), {})]
    assert backend.prompt_builder is build_encoder_tagger_prompt
    assert backend.backend_type == "modernbert"
    assert len(backend.checkpoint_fingerprint) == 64
    assert model.eval_calls == 1
    assert model.to_calls == ["cpu"]


@pytest.mark.parametrize(
    "config_update",
    [
        {"label2id": {"KEEP": 1, "DROP": 0}},
        {"prompt_format_version": 1},
        {"base_model_type": "bert"},
    ],
)
def test_encoder_checkpoint_loader_rejects_incompatible_contract(
    monkeypatch,
    tmp_path,
    config_update,
):
    config = _valid_encoder_config()
    config.update(config_update)
    (tmp_path / "tagger_config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        token_tagger_module,
        "AutoTokenizer",
        FakePretrainedLoader(FakeTokenizer()),
    )
    monkeypatch.setattr(
        token_tagger_module,
        "AutoModelForTokenClassification",
        FakePretrainedLoader(FakeModel()),
    )

    with pytest.raises(ValueError, match="checkpoint"):
        TokenTaggerBackend.from_encoder_checkpoint(
            checkpoint_path=str(tmp_path),
            device="cpu",
            batch_size=1,
        )


def test_datafilter_checkpoint_loader_requires_bidirectional_head(
    monkeypatch,
    tmp_path,
):
    (tmp_path / "tagger_config.json").write_text(
        json.dumps({"head_type": "linear"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        token_tagger_module,
        "AutoTokenizer",
        FakePretrainedLoader(FakeTokenizer()),
    )

    with pytest.raises(ValueError, match="bidir_transformer"):
        TokenTaggerBackend.from_datafilter_checkpoint(
            checkpoint_path=str(tmp_path),
            backbone_model="/models/DataFilter",
            device="cpu",
            batch_size=1,
        )


def test_datafilter_checkpoint_loader_uses_separate_backbone(
    monkeypatch,
    tmp_path,
):
    config = {
        "head_type": "bidir_transformer",
        "hidden_size": 4096,
        "label2id": {"KEEP": 0, "DROP": 1},
        "prompt_format_version": 1,
        "base_model_name": "/models/DataFilter",
        "base_model_revision": None,
        "head_config": {
            "projection_dim": 512,
            "num_attention_heads": 8,
            "ffn_dim": 2048,
            "num_transformer_layers": 1,
            "dropout": 0.1,
        },
    }
    (tmp_path / "tagger_config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    (tmp_path / "tagger_head.safetensors").write_bytes(b"weights")
    tokenizer = FakeTokenizer()
    backbone = FakeBackbone()
    model = FakeModel()
    tokenizer_loader = FakePretrainedLoader(tokenizer)
    backbone_loader = FakePretrainedLoader(backbone)
    monkeypatch.setattr(token_tagger_module, "AutoTokenizer", tokenizer_loader)
    monkeypatch.setattr(token_tagger_module, "AutoModel", backbone_loader)
    monkeypatch.setattr(
        token_tagger_module,
        "_build_frozen_datafilter_tagger",
        lambda loaded_backbone, loaded_config: (
            model if loaded_backbone is backbone and loaded_config == config else None
        ),
    )
    monkeypatch.setattr(
        token_tagger_module,
        "load_safetensors_file",
        lambda path: {"head.weight": path.name},
    )

    backend = TokenTaggerBackend.from_datafilter_checkpoint(
        checkpoint_path=str(tmp_path),
        backbone_model="/models/DataFilter",
        device="cpu",
        batch_size=2,
    )

    assert backbone_loader.calls[0][0] == ("/models/DataFilter",)
    assert tokenizer_loader.calls == [((str(tmp_path),), {"use_fast": True})]
    assert model.head.loaded_state == {"head.weight": "tagger_head.safetensors"}
    assert backend.prompt_builder is build_datafilter_tagger_prompt
    assert backend.backend_type == "datafilter_bidir"
    assert len(backend.checkpoint_fingerprint) == 64


def test_datafilter_checkpoint_rejects_backbone_reference_mismatch(
    monkeypatch,
    tmp_path,
):
    config = _valid_datafilter_config(
        base_model_name="/models/trained-backbone",
    )
    (tmp_path / "tagger_config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    (tmp_path / "tagger_head.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(
        token_tagger_module,
        "AutoTokenizer",
        FakePretrainedLoader(FakeTokenizer()),
    )
    monkeypatch.setattr(
        token_tagger_module,
        "AutoModel",
        FakePretrainedLoader(FakeBackbone()),
    )
    monkeypatch.setattr(
        token_tagger_module,
        "load_safetensors_file",
        lambda path: {},
    )

    with pytest.raises(ValueError, match="DATAFILTER_ALLOW_BACKBONE_MISMATCH"):
        TokenTaggerBackend.from_datafilter_checkpoint(
            checkpoint_path=str(tmp_path),
            backbone_model="/models/runtime-backbone",
            device="cpu",
            batch_size=1,
        )


def test_checkpoint_fingerprint_changes_with_model_bytes(tmp_path):
    (tmp_path / "tagger_config.json").write_text(
        json.dumps(_valid_encoder_config()),
        encoding="utf-8",
    )
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"first")
    first = _checkpoint_fingerprint(tmp_path)

    weights.write_bytes(b"second")
    second = _checkpoint_fingerprint(tmp_path)

    assert first != second
    assert len(first) == 64


def test_invalid_filtered_json_is_retained_and_logged():
    backend = FakeBackend(
        [
            TaggerPrediction(
                filtered_tool_output='{"body": ',
                predicted_drop_spans=[{"start": 9, "end": 23}],
                input_tokens=7,
                latency_ms=4,
            )
        ]
    )
    defense = ToolOutputTaggerDefense(
        defense_name="modernbert_tagger",
        checkpoint_path="/models/mb",
        backend=backend,
    )
    event_logger = FakeEventLogger()

    with event_logger:
        output_messages = defense.query(
            "Read.",
            FunctionsRuntime(),
            EmptyEnv(),
            [_tool_message("1", "read_file", '{"body": "MALICIOUS"}')],
            {},
        )[3]

    assert get_text_content_as_str(output_messages[0]["content"]) == '{"body": '
    event = event_logger.context["tagger_defense_events"][0]
    assert event["original_json_valid"] is True
    assert event["filtered_json_valid"] is False
    assert event["sanitized"] is True
    assert event["success"] is True
    assert event["predicted_drop_spans"] == [{"start": 9, "end": 23}]
    assert event["architecture"] == {"model_type": "fake"}
    assert event_logger.saved == 1


def test_backend_error_keeps_original_output_and_marks_unsanitized():
    defense = ToolOutputTaggerDefense(
        defense_name="datafilter_bidir_tagger",
        checkpoint_path="/models/bidir",
        backend=FailingBackend(),
    )
    original = '{"body": "MALICIOUS"}'
    event_logger = FakeEventLogger()

    with event_logger:
        output_messages = defense.query(
            "Read.",
            FunctionsRuntime(),
            EmptyEnv(),
            [_tool_message("1", "read_file", original)],
            {},
        )[3]

    assert get_text_content_as_str(output_messages[0]["content"]) == original
    event = event_logger.context["tagger_defense_events"][0]
    assert event["success"] is False
    assert event["sanitized"] is False
    assert event["error"] == "inference failed"
    assert event["changed"] is False


def test_backend_failure_only_falls_back_affected_sub_batch():
    backend = object.__new__(TokenTaggerBackend)
    backend.batch_size = 1

    def fake_model_batch(instructions, tool_outputs):
        if tool_outputs == ["bad"]:
            raise TaggerBatchError(
                "CUDA OOM for token lengths [99]",
                input_tokens=[99],
            )
        return [
            TaggerPrediction(
                filtered_tool_output="clean",
                predicted_drop_spans=[{"start": 4, "end": 7}],
                input_tokens=3,
            )
        ]

    backend._sanitize_model_batch = fake_model_batch

    predictions = backend.sanitize_batch(
        ["instruction", "instruction"],
        ["safeBAD", "bad"],
    )

    assert predictions[0].filtered_tool_output == "clean"
    assert predictions[0].error is None
    assert predictions[1].filtered_tool_output == "bad"
    assert predictions[1].input_tokens == 99
    assert predictions[1].error == "CUDA OOM for token lengths [99]"


def test_backend_splits_failed_batch_to_preserve_successful_items():
    backend = object.__new__(TokenTaggerBackend)
    backend.batch_size = 2
    calls = []

    def fake_model_batch(instructions, tool_outputs):
        calls.append(list(tool_outputs))
        if len(tool_outputs) > 1:
            raise TaggerBatchError(
                "CUDA OOM for mixed batch",
                input_tokens=[3, 99],
            )
        if tool_outputs == ["bad"]:
            raise TaggerBatchError(
                "CUDA OOM for bad item",
                input_tokens=[99],
            )
        return [
            TaggerPrediction(
                filtered_tool_output="clean",
                predicted_drop_spans=[{"start": 4, "end": 7}],
                input_tokens=3,
            )
        ]

    backend._sanitize_model_batch = fake_model_batch

    predictions = backend.sanitize_batch(
        ["instruction", "instruction"],
        ["safeBAD", "bad"],
    )

    assert calls == [["safeBAD", "bad"], ["safeBAD"], ["bad"]]
    assert predictions[0].filtered_tool_output == "clean"
    assert predictions[0].error is None
    assert predictions[1].filtered_tool_output == "bad"
    assert predictions[1].error == "CUDA OOM for bad item"


@pytest.mark.parametrize(
    ("defense_name", "factory_name"),
    [
        (
            "datafilter_bidir_tagger",
            "_make_datafilter_bidir_tagger_defense",
        ),
        ("modernbert_tagger", "_make_modernbert_tagger_defense"),
    ],
)
def test_tagger_pipeline_registration_uses_json_formatter(
    monkeypatch,
    defense_name,
    factory_name,
):
    defense = ToolOutputTaggerDefense(
        defense_name=defense_name,
        checkpoint_path="/models/checkpoint",
        backend=FakeBackend([]),
    )
    monkeypatch.setattr(
        agent_pipeline_module,
        factory_name,
        lambda: defense,
        raising=False,
    )

    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm=FakeLLM(),
            model_id=None,
            defense=defense_name,
            system_message="system",
            system_message_name=None,
            tool_delimiter="tool",
        )
    )

    tools_loop = list(pipeline.elements)[3]
    assert isinstance(tools_loop, ToolsExecutionLoop)
    assert tools_loop.elements[1] is defense
    assert pipeline.name == (f"deepseek-v4-flash-{defense_name}-0123456789ab")
    tools_executor = tools_loop.elements[0]
    assert isinstance(tools_executor, ToolsExecutor)
    assert json.loads(tools_executor.output_formatter(FakeToolResult(value="x"))) == {"value": "x"}


def test_modernbert_factory_reads_checkpoint_and_runtime_env(monkeypatch):
    backend = FakeBackend([])
    seen = {}

    def fake_loader(**kwargs):
        seen.update(kwargs)
        return backend

    monkeypatch.setenv("MODERNBERT_TAGGER_CHECKPOINT", "/models/mb")
    monkeypatch.setenv("TAGGER_DEVICE", "cuda:0")
    monkeypatch.setenv("MODERNBERT_TAGGER_BATCH_SIZE", "6")
    monkeypatch.setattr(
        token_tagger_module.TokenTaggerBackend,
        "from_encoder_checkpoint",
        fake_loader,
    )

    defense = agent_pipeline_module._make_modernbert_tagger_defense()

    assert defense.checkpoint_path == "/models/mb"
    assert seen == {
        "checkpoint_path": "/models/mb",
        "device": "cuda:0",
        "batch_size": 6,
    }


def test_datafilter_factory_reads_checkpoint_backbone_and_runtime_env(
    monkeypatch,
):
    backend = FakeBackend([])
    seen = {}

    def fake_loader(**kwargs):
        seen.update(kwargs)
        return backend

    monkeypatch.setenv("DATAFILTER_TAGGER_CHECKPOINT", "/models/bidir")
    monkeypatch.setenv("DATAFILTER_BACKBONE_MODEL", "/models/DataFilter")
    monkeypatch.setattr(
        token_tagger_module.TokenTaggerBackend,
        "from_datafilter_checkpoint",
        fake_loader,
    )

    defense = agent_pipeline_module._make_datafilter_bidir_tagger_defense()

    assert defense.checkpoint_path == "/models/bidir"
    assert seen == {
        "checkpoint_path": "/models/bidir",
        "backbone_model": "/models/DataFilter",
        "backbone_revision": None,
        "allow_backbone_mismatch": False,
        "device": "cuda",
        "batch_size": 1,
    }


def test_shared_batch_size_is_a_backward_compatible_fallback(monkeypatch):
    backend = FakeBackend([])
    seen = {}

    def fake_loader(**kwargs):
        seen.update(kwargs)
        return backend

    monkeypatch.setenv("MODERNBERT_TAGGER_CHECKPOINT", "/models/mb")
    monkeypatch.setenv("TAGGER_BATCH_SIZE", "3")
    monkeypatch.delenv("MODERNBERT_TAGGER_BATCH_SIZE", raising=False)
    monkeypatch.setattr(
        token_tagger_module.TokenTaggerBackend,
        "from_encoder_checkpoint",
        fake_loader,
    )

    agent_pipeline_module._make_modernbert_tagger_defense()

    assert seen["batch_size"] == 3


class FakePretrainedLoader:
    def __init__(self, value) -> None:
        self.value = value
        self.calls = []

    def from_pretrained(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.value


class FakeTokenizer:
    is_fast = True
    pad_token_id = 0
    model_max_length = 1024


class FakeHead:
    def __init__(self) -> None:
        self.loaded_state = None

    def load_state_dict(self, state, strict=True):
        assert strict is True
        self.loaded_state = state


class FakeModel:
    def __init__(self) -> None:
        self.eval_calls = 0
        self.to_calls = []
        self.head = FakeHead()
        self.config = SimpleNamespace(
            model_type="modernbert",
            hidden_size=8,
            label2id={"KEEP": 0, "DROP": 1},
        )

    def eval(self):
        self.eval_calls += 1
        return self

    def to(self, device):
        self.to_calls.append(device)
        return self


class FakeBackbone:
    config = SimpleNamespace(hidden_size=4096, use_cache=True)


def _valid_encoder_config(**updates):
    config = {
        "head_type": "encoder",
        "base_model_type": "modernbert",
        "label2id": {"KEEP": 0, "DROP": 1},
        "prompt_format_version": 2,
    }
    config.update(updates)
    return config


def _valid_datafilter_config(**updates):
    config = {
        "head_type": "bidir_transformer",
        "hidden_size": 4096,
        "label2id": {"KEEP": 0, "DROP": 1},
        "prompt_format_version": 1,
        "base_model_name": "/models/DataFilter",
        "base_model_revision": None,
        "head_config": {
            "projection_dim": 512,
            "num_attention_heads": 8,
            "ffn_dim": 2048,
            "num_transformer_layers": 1,
            "dropout": 0.1,
        },
    }
    config.update(updates)
    return config


def _tool_message(
    call_id: str,
    function: str,
    content: str,
    *,
    error: str | None = None,
) -> ChatToolResultMessage:
    tool_call = FunctionCall(function=function, args={}, id=call_id)
    return ChatToolResultMessage(
        role="tool",
        content=[text_content_block_from_string(content)],
        tool_call_id=call_id,
        tool_call=tool_call,
        error=error,
    )
