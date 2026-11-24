"""Strict, allowlisted JSON tool-call protocol for Davinci inference.

This module never executes arbitrary generated code. Only callables explicitly
registered by the host application can be invoked.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import re
import inspect
from typing import Any, Callable, Mapping


_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")


class ToolCallError(ValueError):
    """Invalid model-emitted tool call or arguments."""


class ToolExecutionError(RuntimeError):
    """A registered tool raised an exception during execution."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ToolCallError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def parse_tool_call(text: str) -> ToolCall:
    """Parse exactly one JSON object: {"name": str, "arguments": object}."""
    if not isinstance(text, str) or not text.strip():
        raise ToolCallError("Tool-call output must be a non-empty JSON string")
    def reject_constant(value):
        raise ToolCallError(f"Non-standard JSON constant is not allowed: {value}")

    try:
        payload = json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=reject_constant
        )
    except ToolCallError:
        raise
    except json.JSONDecodeError as exc:
        raise ToolCallError(f"Invalid tool-call JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ToolCallError("Tool-call JSON must be an object")
    if set(payload) != {"name", "arguments"}:
        raise ToolCallError("Tool-call object must contain exactly 'name' and 'arguments'")
    name, arguments = payload["name"], payload["arguments"]
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ToolCallError("Tool name has an invalid format")
    if not isinstance(arguments, dict):
        raise ToolCallError("Tool arguments must be a JSON object")
    return ToolCall(name=name, arguments=arguments)


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and math.isfinite(value)
        )
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise ToolCallError(f"Unsupported JSON Schema type: {expected!r}")


_ALLOWED_SCHEMA_KEYS = {
    "type", "properties", "required", "additionalProperties", "items", "enum",
    "minItems", "maxItems", "uniqueItems", "minLength", "maxLength", "pattern",
    "minimum", "maximum", "description", "title", "default", "examples",
}
_ALLOWED_SCHEMA_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}


def _validate_schema_definition(schema: Mapping[str, Any], path: str = "$schema") -> None:
    if not isinstance(schema, Mapping):
        raise ValueError(f"{path} must be a JSON Schema object")
    unknown = set(schema) - _ALLOWED_SCHEMA_KEYS
    if unknown:
        raise ValueError(f"{path} uses unsupported schema keywords: {sorted(unknown)}")
    expected = schema.get("type")
    if expected is not None and (not isinstance(expected, str) or expected not in _ALLOWED_SCHEMA_TYPES):
        raise ValueError(f"{path}.type is unsupported")
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise ValueError(f"{path}.properties must be an object")
    for name, child in properties.items():
        if not isinstance(name, str):
            raise ValueError(f"{path}.properties keys must be strings")
        _validate_schema_definition(child, f"{path}.properties.{name}")
    required = schema.get("required", [])
    if (not isinstance(required, list) or any(not isinstance(x, str) for x in required)
            or len(required) != len(set(required))):
        raise ValueError(f"{path}.required must be a unique list of strings")
    additional = schema.get("additionalProperties", False)
    if not isinstance(additional, bool):
        _validate_schema_definition(additional, f"{path}.additionalProperties")
    if "items" in schema:
        _validate_schema_definition(schema["items"], f"{path}.items")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ValueError(f"{path}.enum must be a non-empty list")
    for key in ("minItems", "maxItems", "minLength", "maxLength"):
        if key in schema and (
            not isinstance(schema[key], int) or isinstance(schema[key], bool) or schema[key] < 0
        ):
            raise ValueError(f"{path}.{key} must be a non-negative integer")
    if schema.get("minItems", 0) > schema.get("maxItems", math.inf):
        raise ValueError(f"{path}.minItems cannot exceed maxItems")
    if schema.get("minLength", 0) > schema.get("maxLength", math.inf):
        raise ValueError(f"{path}.minLength cannot exceed maxLength")
    if "uniqueItems" in schema and not isinstance(schema["uniqueItems"], bool):
        raise ValueError(f"{path}.uniqueItems must be boolean")
    if "pattern" in schema:
        if not isinstance(schema["pattern"], str):
            raise ValueError(f"{path}.pattern must be a string")
        try:
            re.compile(schema["pattern"])
        except re.error as exc:
            raise ValueError(f"{path}.pattern is invalid: {exc}") from exc
    for key in ("minimum", "maximum"):
        if key in schema and (
            not isinstance(schema[key], (int, float)) or isinstance(schema[key], bool)
            or (isinstance(schema[key], float) and not math.isfinite(schema[key]))
        ):
            raise ValueError(f"{path}.{key} must be a finite number")
    if schema.get("minimum", -math.inf) > schema.get("maximum", math.inf):
        raise ValueError(f"{path}.minimum cannot exceed maximum")


def _validate_value(value: Any, schema: Mapping[str, Any], path: str) -> None:
    if not isinstance(schema, Mapping):
        raise ToolCallError(f"{path}: schema must be an object")
    expected = schema.get("type")
    if expected is not None and not _matches_type(value, expected):
        raise ToolCallError(f"{path}: expected {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ToolCallError(f"{path}: value is not in enum")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            raise ToolCallError(f"{path}: invalid object schema")
        for key in required:
            if key not in value:
                raise ToolCallError(f"{path}.{key}: required property is missing")
        additional = schema.get("additionalProperties", False)
        for key, item in value.items():
            if key in properties:
                _validate_value(item, properties[key], f"{path}.{key}")
            elif additional is False:
                raise ToolCallError(f"{path}.{key}: additional property is not allowed")
            elif isinstance(additional, Mapping):
                _validate_value(item, additional, f"{path}.{key}")

    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ToolCallError(f"{path}: fewer than minItems")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ToolCallError(f"{path}: more than maxItems")
        if schema.get("uniqueItems"):
            encoded = [json.dumps(v, sort_keys=True, ensure_ascii=False) for v in value]
            if len(encoded) != len(set(encoded)):
                raise ToolCallError(f"{path}: array items must be unique")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                _validate_value(item, item_schema, f"{path}[{index}]")

    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ToolCallError(f"{path}: string shorter than minLength")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ToolCallError(f"{path}: string longer than maxLength")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise ToolCallError(f"{path}: string does not match pattern")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ToolCallError(f"{path}: number below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ToolCallError(f"{path}: number above maximum")


class ToolRegistry:
    """Host-controlled allowlist of synchronous tools and their argument schemas."""

    def __init__(self):
        self._tools: dict[str, tuple[ToolDefinition, Callable[..., Any]]] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Mapping[str, Any],
        function: Callable[..., Any],
    ) -> None:
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise ValueError("Tool name must match [A-Za-z][A-Za-z0-9_.-]{0,63}")
        if name in self._tools:
            raise ValueError(f"Tool {name!r} is already registered")
        if not isinstance(description, str):
            raise ValueError("Tool description must be a string")
        if not isinstance(parameters, Mapping) or parameters.get("type") != "object":
            raise ValueError("Tool parameters must be a JSON Schema object")
        _validate_schema_definition(parameters)
        if not callable(function):
            raise ValueError("Tool implementation must be callable")
        definition = ToolDefinition(name, description, deepcopy(dict(parameters)))
        self._tools[name] = (definition, function)

    def definitions(self) -> list[dict[str, Any]]:
        """Return model-facing definitions without exposing Python callables."""
        return [
            {
                "name": definition.name,
                "description": definition.description,
                "parameters": deepcopy(definition.parameters),
            }
            for definition, _ in self._tools.values()
        ]

    def execute(self, call: ToolCall | str) -> Any:
        """Validate and invoke one registered tool; never evaluates generated code."""
        parsed = parse_tool_call(call) if isinstance(call, str) else call
        if not isinstance(parsed, ToolCall):
            raise ToolCallError("execute expects ToolCall or serialized JSON")
        if not isinstance(parsed.name, str) or not _NAME_RE.fullmatch(parsed.name):
            raise ToolCallError("Tool name has an invalid format")
        if not isinstance(parsed.arguments, dict):
            raise ToolCallError("Tool arguments must be a JSON object")
        if parsed.name not in self._tools:
            raise ToolCallError(f"Unknown tool: {parsed.name}")
        definition, function = self._tools[parsed.name]
        _validate_value(parsed.arguments, definition.parameters, "$")
        try:
            result = function(**parsed.arguments)
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise ToolExecutionError("Async tools are not supported by this synchronous registry")
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"Tool {parsed.name!r} failed: {exc}") from exc
        return result

    def execute_json(self, text: str) -> str:
        """Execute a registered call and return a UTF-8-friendly JSON result."""
        result = self.execute(text)
        try:
            return json.dumps(result, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ToolExecutionError("Tool result is not strict JSON-serializable") from exc
