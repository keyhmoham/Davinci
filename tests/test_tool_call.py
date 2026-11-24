import json
import pytest

from src.davinci.tool_call import (
    ToolCallError,
    ToolExecutionError,
    ToolRegistry,
    parse_tool_call,
)


def make_registry():
    registry = ToolRegistry()
    registry.register(
        "add",
        "Add two integers",
        {
            "type": "object",
            "properties": {
                "left": {"type": "integer"},
                "right": {"type": "integer"},
            },
            "required": ["left", "right"],
            "additionalProperties": False,
        },
        lambda left, right: {"sum": left + right},
    )
    return registry


def test_parse_and_execute_registered_tool():
    registry = make_registry()
    output = registry.execute_json('{"name":"add","arguments":{"left":2,"right":3}}')
    assert json.loads(output) == {"sum": 5}


@pytest.mark.parametrize("text", [
    "",
    "not json",
    "[]",
    '{"name":"add","arguments":{"left":1},"extra":true}',
    '{"name":"add","name":"add","arguments":{}}',
    '{"name":"add","arguments":[]}',
    '{"name":"bad name","arguments":{}}',
])
def test_parser_rejects_malformed_or_ambiguous_calls(text):
    with pytest.raises(ToolCallError):
        parse_tool_call(text)


def test_registry_rejects_unknown_tools_and_extra_arguments():
    registry = make_registry()
    with pytest.raises(ToolCallError, match="Unknown tool"):
        registry.execute('{"name":"missing","arguments":{}}')
    with pytest.raises(ToolCallError, match="additional property"):
        registry.execute('{"name":"add","arguments":{"left":1,"right":2,"shell":"x"}}')


def test_schema_rejects_wrong_types_and_missing_required_values():
    registry = make_registry()
    with pytest.raises(ToolCallError, match="expected integer"):
        registry.execute('{"name":"add","arguments":{"left":true,"right":2}}')
    with pytest.raises(ToolCallError, match="required property"):
        registry.execute('{"name":"add","arguments":{"left":1}}')


def test_registry_exposes_definitions_without_callables():
    definitions = make_registry().definitions()
    assert definitions[0]["name"] == "add"
    assert "function" not in definitions[0]


def test_tool_exceptions_are_wrapped():
    registry = ToolRegistry()
    registry.register("fail", "Always fails", {"type": "object"}, lambda: 1 / 0)
    with pytest.raises(ToolExecutionError, match="failed"):
        registry.execute('{"name":"fail","arguments":{}}')


def test_non_json_tool_results_are_rejected():
    registry = ToolRegistry()
    registry.register("bad_result", "Returns an object", {"type": "object"}, lambda: object())
    with pytest.raises(ToolExecutionError, match="JSON-serializable"):
        registry.execute_json('{"name":"bad_result","arguments":{}}')


def test_parser_rejects_non_standard_json_constants():
    with pytest.raises(ToolCallError, match="Non-standard JSON"):
        parse_tool_call('{"name":"add","arguments":{"left":NaN,"right":1}}')


def test_registry_rejects_unsupported_schema_keywords():
    registry = ToolRegistry()
    with pytest.raises(ValueError, match="unsupported schema keywords"):
        registry.register(
            "unsafe_schema", "Bad schema",
            {"type": "object", "properties": {}, "oneOf": []},
            lambda: None,
        )


def test_async_tool_is_rejected_by_sync_registry():
    async def async_result():
        return {"ok": True}
    registry = ToolRegistry()
    registry.register("async_tool", "Async tool", {"type": "object"}, async_result)
    with pytest.raises(ToolExecutionError, match="Async tools are not supported"):
        registry.execute('{"name":"async_tool","arguments":{}}')
