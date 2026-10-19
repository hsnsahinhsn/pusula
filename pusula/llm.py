"""Async client for an OpenAI-compatible server (vLLM) with JSON-schema constrained output."""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Awaitable, Callable, Iterable, Optional

from rich.console import Console

console = Console()

DEFAULT_BASE_URL = os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
DEFAULT_MODEL = os.getenv("LLM_MODEL_NAME", "qwen")


def make_client(base_url: str = DEFAULT_BASE_URL, timeout: float = 600.0):
    from openai import AsyncOpenAI

    return AsyncOpenAI(base_url=base_url, api_key=os.getenv("LLM_API_KEY", "EMPTY"), timeout=timeout)


async def chat_json(
    client,
    model: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    *,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 2048,
    retries: int = 3,
    extra_body: Optional[dict[str, Any]] = None,
    usage: Optional[list[dict[str, Any]]] = None,
) -> Optional[dict[str, Any]]:
    """One structured completion; returns the parsed object, or None after `retries` failures.

    With ``usage``, each response's token usage (and the cost, when the server reports it, e.g. OpenRouter) is appended
    to that list, failed attempts included: they were billed too.

    Connection failures (the server restarting, e.g. after a watchdog restart on a shared GPU) do not
    use up retries: the call waits for the server to come back and tries again.
    """
    from openai import APIConnectionError

    attempt = 0
    while attempt < retries:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "output", "schema": schema, "strict": True},
                },
                # Qwen3.x: answer directly; structured output and thinking do not mix. Other models pass their own
                # options (e.g. gpt-oss: {"reasoning_effort": "medium"}).
                extra_body=extra_body if extra_body is not None else {"chat_template_kwargs": {"enable_thinking": False}},
            )
            if usage is not None and getattr(resp, "usage", None) is not None:
                usage.append(resp.usage.model_dump())
            choice = resp.choices[0]
            if choice.finish_reason == "length":
                raise ValueError("output truncated")
            return json.loads(choice.message.content)
        except APIConnectionError:
            await wait_for_server(str(client.base_url).rstrip("/"))
        except Exception as exc:  # invalid JSON, truncation, server-side error
            attempt += 1
            if attempt == retries:
                console.print(f"[yellow]LLM isteği başarısız ({type(exc).__name__}: {exc})[/yellow]")
                return None
            await asyncio.sleep(1.5 * attempt)
    return None


async def wait_for_server(base_url: str = DEFAULT_BASE_URL, timeout_s: float = 3600.0) -> None:
    import urllib.request

    url = base_url.rstrip("/") + "/models"
    t0 = time.time()
    while True:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"LLM server at {url} did not come up in {timeout_s:.0f}s")
        await asyncio.sleep(10)


async def run_pool(
    items: Iterable[Any],
    worker: Callable[[Any], Awaitable[Any]],
    concurrency: int,
    on_result: Callable[[Any, Any], None],
    label: str = "iş",
    total: Optional[int] = None,
) -> tuple[int, int]:
    """Run `worker` over `items` with bounded concurrency; returns (ok, failed)."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 4)
    ok = failed = 0
    t0 = time.time()

    async def consumer():
        nonlocal ok, failed
        while True:
            item = await queue.get()
            if item is _STOP:
                queue.task_done()
                return
            try:
                result = await worker(item)
            except Exception as exc:
                console.print(f"[red]{label} hatası: {exc}[/red]")
                result = None
            if result is None:
                failed += 1
            else:
                ok += 1
                on_result(item, result)
            done = ok + failed
            if done % 200 == 0:
                rate = done / max(time.time() - t0, 1e-6)
                tot = f"/{total}" if total else ""
                console.print(f"{label}: {done}{tot} (başarılı {ok}, hata {failed}) {rate:.1f}/s")
            queue.task_done()

    consumers = [asyncio.create_task(consumer()) for _ in range(concurrency)]
    for item in items:
        await queue.put(item)
    for _ in consumers:
        await queue.put(_STOP)
    await asyncio.gather(*consumers)
    return ok, failed


_STOP = object()
