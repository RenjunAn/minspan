import json

from agentdojo.agent_pipeline.tool_execution import tool_result_to_str


def test_tool_result_to_str_preserves_default_dict_string_format():
    assert tool_result_to_str({"name": "Alice", "score": 1}) == "{'name': 'Alice', 'score': 1}"


def test_tool_result_to_str_formats_dict_as_json_when_requested():
    assert tool_result_to_str({"name": "Alice", "score": 1}, dump_fn=json.dumps) == '{"name": "Alice", "score": 1}'
