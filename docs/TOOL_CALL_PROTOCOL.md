# Davinci tool-call protocol

The canonical host-side tool protocol is implemented in `src/davinci/tool_call.py`.
It deliberately separates model output parsing from tool execution.

## Wire format

A model emits exactly one JSON object:

```json
{"name":"add","arguments":{"left":2,"right":3}}
```

The parser rejects surrounding prose, Markdown fences, duplicate JSON keys,
unknown fields, invalid tool names, and non-object arguments. Keep the tool-call
format stable in the SFT dataset and evaluation harness.

## Host-controlled execution

The application registers each callable explicitly in `ToolRegistry`, together
with a JSON Schema object describing its arguments. Only registered functions
can execute. Generated Python, shell, SQL, or other code is never evaluated by
this module. The registry returns model-facing definitions without exposing
Python callables.

The built-in validator supports a constrained JSON Schema subset: object,
array, string, integer, number, boolean, null, required properties, property
schemas, additionalProperties, enum, array item schemas, min/max item counts,
uniqueItems, min/max string lengths, regex pattern, and numeric minimum/maximum.
Unsupported schema keywords are not interpreted; keep schemas within this
documented subset or validate them in the host application with a full JSON
Schema implementation.

`execute_json` serializes tool results with `ensure_ascii=False` and rejects
non-JSON results and NaN/Infinity. Tools are synchronous; asynchronous tools
should be adapted by the host runtime rather than run implicitly here.

## Tests

```bash
PYTHONPATH=. python -m pytest -q tests/test_tool_call.py
```

This is a small protocol boundary, not a complete agent loop. The application
still owns tool authorization, user consent, timeouts, rate limits, sandboxing,
secret redaction, audit logs, and the policy for returning tool observations
to the model.
