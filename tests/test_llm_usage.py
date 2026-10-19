"""Billed usage is recorded per request (OpenRouter reports the cost), failed attempts included."""
import asyncio
import json
from types import SimpleNamespace

from pusula.llm import chat_json


class _Usage:
    def __init__(self, cost):
        self.cost = cost

    def model_dump(self):
        return {"cost": self.cost, "completion_tokens": 10, "completion_tokens_details": {"reasoning_tokens": 7}}


class _Client:
    def __init__(self, replies):
        self.replies = list(replies)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **_):
        content, finish = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=content))],
                               usage=_Usage(0.001))


def test_usage_includes_failed_attempts(monkeypatch):
    monkeypatch.setattr("pusula.llm.asyncio.sleep", _no_wait)  # the retry back-off
    client = _Client([("{", "length"), (json.dumps({"a": "A"}), "stop")])
    usage = []
    out = asyncio.run(chat_json(client, "m", [], {"type": "object"}, usage=usage))
    assert out == {"a": "A"}
    assert len(usage) == 2 and sum(u["cost"] for u in usage) == 0.002
    assert [u["finish_reason"] for u in usage] == ["length", "stop"]  # runaway outputs can be counted


def test_usage_is_optional():
    client = _Client([(json.dumps({"a": "B"}), "stop")])
    assert asyncio.run(chat_json(client, "m", [], {"type": "object"})) == {"a": "B"}


async def _no_wait(*_):
    return None
