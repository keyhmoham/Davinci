import json
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from src.davinci.server import create_app


class FakeTokenizer:
    def encode(self, text, out_type=int):
        return [1, 1]
    def decode(self, ids):
        return "".join("x" for token in ids if token == 2)
    def eos_id(self):
        return 0


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(vocab_size=5)
    def _logits(self, batch):
        logits = torch.zeros(batch, 1, 5, device=self.weight.device)
        logits[:, :, 2] = 5.0
        return logits
    def prefill(self, input_ids):
        return self._logits(input_ids.shape[0]), []
    def decode(self, input_ids, position, caches):
        return self._logits(input_ids.shape[0])


def make_client(**kwargs):
    model = FakeModel()
    app = create_app(model, FakeTokenizer(), max_input_tokens=20, max_new_tokens=4,
                     max_total_tokens=24, **kwargs)
    return TestClient(app), model


def test_health_and_models_endpoints():
    client, _ = make_client()
    assert client.get("/health").json()["status"] == "ok"
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "davinci"


def test_chat_completion_returns_openai_compatible_response():
    client, model = make_client()
    response = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "سلام"}],
        "max_tokens": 2, "temperature": 0,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "xx"
    assert body["usage"]["completion_tokens"] == 2
    assert model.training is True


def test_streaming_chat_completion_emits_sse_and_done_marker():
    client, model = make_client()
    response = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 2, "temperature": 0, "stream": True,
    })
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "chat.completion.chunk" in response.text
    assert "[DONE]" in response.text
    assert model.training is True


def test_bearer_authentication_is_enforced():
    client, _ = make_client(api_key="secret")
    assert client.get("/v1/models").status_code == 401
    response = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200


def test_prompt_and_generation_limits_are_enforced():
    model = FakeModel()
    client = TestClient(create_app(model, FakeTokenizer(), max_input_tokens=1,
                                   max_new_tokens=4, max_total_tokens=4))
    too_long = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1,
    })
    assert too_long.status_code == 413
    excessive_output = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5,
    })
    assert excessive_output.status_code == 400


def test_invalid_message_roles_and_final_role_are_rejected():
    client, _ = make_client()
    invalid = client.post("/v1/chat/completions", json={
        "messages": [{"role": "assistant", "content": "not a user turn"}],
    })
    assert invalid.status_code == 400


def test_server_configuration_rejects_inconsistent_limits():
    with pytest.raises(ValueError, match="must not exceed"):
        create_app(FakeModel(), FakeTokenizer(), max_input_tokens=100, max_total_tokens=50)