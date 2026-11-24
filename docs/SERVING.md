# Davinci inference server

`src/davinci/server.py` exposes a small OpenAI-compatible HTTP interface around an already-loaded Davinci model and SentencePiece tokenizer.

## Install and run

Install the optional serving dependencies:

```bash
python -m pip install -r requirements-server.txt
```

Load the exact model checkpoint and tokenizer, then create the app:

```python
import os
import sentencepiece as spm
import uvicorn
from src.davinci.model import DavinciForCausalLM
from src.davinci.server import create_app

model = DavinciForCausalLM.from_pretrained("/path/to/model", map_location="cpu")
tokenizer = spm.SentencePieceProcessor(model_file="/path/to/davinci_sentencepiece.model")
app = create_app(
    model, tokenizer,
    api_key=os.environ.get("DAVINCI_API_KEY"),
    max_input_tokens=4096,
    max_new_tokens=512,
    max_total_tokens=4096,
    max_concurrent_requests=1,
)
uvicorn.run(app, host="127.0.0.1", port=8000)
```

Do not expose the unauthenticated server to a public network. Configure a strong API key, TLS termination, network controls, and monitoring at the deployment layer. The `/health` endpoint is intentionally public for basic health checks; `/v1/models` and completion requests use bearer authentication when an API key is configured.

## Endpoints

- `GET /health` — basic process/model health.
- `GET /v1/models` — OpenAI-style model list.
- `POST /v1/chat/completions` — non-streaming or Server-Sent Events streaming chat completions.

The request accepts `model`, `messages`, `max_tokens`, `temperature`, `top_p`, `top_k`, and `stream`. Unsupported fields and message roles are rejected. The final message must be from the user or a tool. The server enforces prompt, output, total-token, and concurrency limits.

## Security and operational limits

- The server does not execute tool calls. Tool dispatch must remain in a separate trusted host process with allowlists, consent, timeouts, and audit logging.
- Configure `api_key`; the default `None` disables authentication and is suitable only for isolated local development.
- The app switches the shared model to evaluation mode at creation so concurrent inference requests cannot accidentally toggle dropout through train/eval mode restoration. This is a single-process reference server with a bounded semaphore, not continuous batching or a multi-tenant serving engine.
- Streaming is token-by-token and may have lower throughput than a dedicated inference engine.
- No model quantization, paged attention, speculative decoding, distributed serving, request cancellation protocol, or production metrics are implemented.
- The model and tokenizer must be loaded from the same verified run lineage.

## Tests

```bash
python -m pip install -r requirements-test.txt
PYTHONPATH=. python -m pytest -q tests/test_server.py
```