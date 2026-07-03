"""
PIArena Defense Adapter for AgentDojo Pipeline

This module provides an adapter to use PIArena defenses within AgentDojo's
agent pipeline. It filters tool outputs using PIArena's defense execute() method.

Defense Types Supported:
    - Detection (PromptGuard, DataSentinel, PIGuard, AttentionTracker):
      Empty context if injection detected
    - Sanitization (DataFilter, PISanitizer):
      Return cleaned context
    - Hybrid (PromptArmor, PromptLocate):
      Return cleaned context (with detection info)

Usage:
    Set environment variables before running:
        export PIARENA_DEFENSE=datafilter
        export PIARENA_PATH=/path/to/PIArena  # optional, auto-detected
"""

import json
import sys
import os
import time
from collections.abc import Sequence

import torch

# Add PIArena to path
PIARENA_PATH = os.environ.get("PIARENA_PATH")
if PIARENA_PATH is None:
    # Try to auto-detect: assume agentdojo is inside PIArena/agents/
    _current = os.path.dirname(os.path.abspath(__file__))
    # Go up from agents/agentdojo/src/agentdojo/agent_pipeline to PIArena root
    _potential_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_current)))))
    if os.path.exists(os.path.join(_potential_path, "piarena")):
        PIARENA_PATH = _potential_path

if PIARENA_PATH and PIARENA_PATH not in sys.path:
    sys.path.insert(0, PIARENA_PATH)

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.logging import Logger
from agentdojo.types import ChatMessage, get_text_content_as_str, text_content_block_from_string
from agentdojo.functions_runtime import Env, EmptyEnv, FunctionsRuntime
from piarena.config import load_json_env_config


EMPTY_SANITIZED_ERROR = "[tool error removed by PIArena defense]"


def _tool_name(message: dict) -> str:
    tool_call = message.get("tool_call")
    if tool_call is None:
        return ""
    if isinstance(tool_call, dict):
        return str(tool_call.get("function", ""))
    return str(getattr(tool_call, "function", ""))


def _record_defense_event(event: dict) -> None:
    logger = Logger.get()
    context = getattr(logger, "context", None)
    save = getattr(logger, "save", None)
    if not isinstance(context, dict) or not callable(save):
        return
    context.setdefault("piarena_defense_events", []).append(event)
    save()


def _apply_defense_to_tool_output(defense, target_inst: str, tool_output: str) -> tuple[str, dict]:
    result = defense.execute(target_inst=target_inst, context=tool_output)
    if "cleaned_context" in result:
        cleaned = result["cleaned_context"]
        if isinstance(cleaned, (dict, list)):
            return json.dumps(cleaned, indent=2, ensure_ascii=False), result
        return str(cleaned), result
    if result.get("detect_flag"):
        return "", result
    return tool_output, result


class PIArenaDefenseAdapter(BasePipelineElement):
    """
    AgentDojo pipeline element that filters tool outputs using PIArena defenses.

    This adapter intercepts tool messages in the conversation and applies
    PIArena's defense mechanism to filter potentially malicious content.

    The defense to use is read from the PIARENA_DEFENSE environment variable.
    Default is 'datafilter' if not set.
    """

    def __init__(self, defense_name: str = None, defense_config: dict = None):
        # Get defense name from env or parameter
        self.defense_name = defense_name or os.environ.get("PIARENA_DEFENSE", "datafilter")
        if defense_config is None:
            defense_config = load_json_env_config("PIARENA_DEFENSE_CONFIG")
        self.defense_config = defense_config

        # Lazy load defense to avoid import issues
        self._defense = None
        print(f"[PIArenaDefenseAdapter] Will use defense: {self.defense_name}")

    def _get_defense(self):
        """Lazy load the defense."""
        if self._defense is None:
            # Initialize CUDA properly before loading defense models
            # This ensures consistent device handling when running in AgentDojo subprocess
            if torch.cuda.is_available():
                torch.cuda.init()
                torch.cuda.set_device(0)
                print(f"[PIArenaDefenseAdapter] CUDA initialized, using device: cuda:0")
            
            from piarena.defenses import get_defense
            self._defense = get_defense(self.defense_name, self.defense_config)
            print(f"[PIArenaDefenseAdapter] Loaded defense: {self.defense_name}")
        return self._defense

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        """
        Process messages and filter tool outputs.

        Only processes if the last message is a tool message.
        Applies the defense to the complete serialized output of each trailing tool message.
        """

        # Only filter if the last message is a tool message
        if len(messages) == 0 or messages[-1]["role"] != "tool":
            return query, runtime, env, messages, extra_args

        defense = self._get_defense()
        first_tool_index = len(messages)
        while first_tool_index > 0 and messages[first_tool_index - 1]["role"] == "tool":
            first_tool_index -= 1

        processed_messages = list(messages[:first_tool_index])
        for msg in messages[first_tool_index:]:
            try:
                input_field = "error" if msg.get("error") is not None else "content"
                raw_data = (
                    str(msg.get("error"))
                    if input_field == "error"
                    else get_text_content_as_str(msg.get("content") or [])
                )
                started_at = time.perf_counter()
                cleaned_str, defense_result = _apply_defense_to_tool_output(
                    defense,
                    target_inst=query,
                    tool_output=raw_data,
                )
                latency_ms = int((time.perf_counter() - started_at) * 1000)

                updated_msg = dict(msg)
                if input_field == "error":
                    updated_msg["error"] = cleaned_str or EMPTY_SANITIZED_ERROR
                else:
                    updated_msg["content"] = [text_content_block_from_string(cleaned_str)]
                processed_messages.append(updated_msg)

                _record_defense_event(
                    {
                        "defense": self.defense_name,
                        "tool_name": _tool_name(msg),
                        "tool_call_id": str(msg.get("tool_call_id") or ""),
                        "input_field": input_field,
                        "original_tool_output": raw_data,
                        "filtered_tool_output": updated_msg["error"] if input_field == "error" else cleaned_str,
                        "changed": cleaned_str != raw_data,
                        "detect_flag": bool(defense_result.get("detect_flag")),
                        "predicted_drop_spans": defense_result.get("predicted_drop_spans", []),
                        "input_tokens": defense_result.get("input_tokens", 0),
                        "latency_ms": defense_result.get("latency_ms", latency_ms),
                        "success": defense_result.get("error") is None,
                        "error": defense_result.get("error"),
                    }
                )
            except Exception as e:
                print(f"[PIArenaDefenseAdapter] Skipped cleaning due to error: {e}")
                processed_messages.append(msg)

        return query, runtime, env, processed_messages, extra_args
