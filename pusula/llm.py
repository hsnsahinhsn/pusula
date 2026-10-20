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


class ServerUnavailable(RuntimeError):
    """The server stayed unreachable for $LLM_SERVER_WAIT seconds: the whole run stops (run_pool re-raises it), so a dead
    server ends the client quickly instead of letting every remaining item wait for it in turn."""


def make_client(base_url: str = DEFAULT_BASE_URL, timeout: float | None = None):
    """``timeout`` (seconds) defaults to $LLM_TIMEOUT or 600; long thinking answers on a busy server need more.
    $LLM_SDK_RETRIES (default 2) is the openai SDK's own retry count; chat_json retries on top of it."""
    from openai import AsyncOpenAI

    timeout = timeout if timeout is not None else float(os.getenv("LLM_TIMEOUT", "600"))
    return AsyncOpenAI(base_url=base_url, api_key=os.getenv("LLM_API_KEY", "EMPTY"), timeout=timeout,
                       max_retries=int(os.getenv("LLM_SDK_RETRIES", "2")))


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
    retry_sampling: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """One structured completion; returns the parsed object, or None after `retries` failures.

    ``retry_sampling`` (e.g. ``{"temperature": 1.0, "top_p": 0.95, "top_k": 20}``) replaces the sampling settings after
    a failed attempt: a greedy thinking model that ran into a loop would repeat the same loop on every retry. Keys other
    than temperature and top_p go to the request body.

    With ``usage``, each response's token usage (and the cost, when the server reports it, e.g. OpenRouter) is appended
    to that list, failed attempts included: they were billed too.

    Connection failures (the server restarting, e.g. after a watchdog restart on a shared GPU) do not
    use up retries: the call waits for the server to come back (at most $LLM_SERVER_WAIT seconds, default 600, then
    ServerUnavailable) and tries again. A request timeout does use up a retry: a hung server still answers /models.
    """
    from openai import APIConnectionError, APITimeoutError

    # Qwen3.x: answer directly; structured output and thinking do not mix. Other models pass their own
    # options (e.g. gpt-oss: {"reasoning_effort": "medium"}).
    body = extra_body if extra_body is not None else {"chat_template_kwargs": {"enable_thinking": False}}
    attempt = 0
    while attempt < retries:
        temp, tp, extra = temperature, top_p, body
        if attempt and retry_sampling:
            rest = {k: v for k, v in retry_sampling.items() if k not in ("temperature", "top_p")}
            temp, tp, extra = retry_sampling.get("temperature", temp), retry_sampling.get("top_p", tp), {**body, **rest}
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temp,
                top_p=tp,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "output", "schema": schema, "strict": True},
                },
                extra_body=extra,
            )
            choice = resp.choices[0]
            if usage is not None and getattr(resp, "usage", None) is not None:
                usage.append({**resp.usage.model_dump(), "finish_reason": choice.finish_reason})
            if choice.finish_reason == "length":
                raise ValueError("output truncated")
            return json.loads(choice.message.content)
        except APITimeoutError as exc:  # a subclass of APIConnectionError: must come first
            attempt += 1
            if attempt == retries:
                console.print(f"[yellow]LLM isteği zaman aşımı ({retries} deneme): {exc}[/yellow]")
                return None
        except APIConnectionError:
            await wait_for_server(str(client.base_url).rstrip("/"), float(os.getenv("LLM_SERVER_WAIT", "600")))
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
            raise ServerUnavailable(f"LLM server at {url} did not come up in {timeout_s:.0f}s")
        await asyncio.sleep(10)


async def run_pool(
    items: Iterable[Any],
    worker: Callable[[Any], Awaitable[Any]],
    concurrency: int,
    on_result: Callable[[Any, Any], None],
    label: str = "iş",
    total: Optional[int] = None,
) -> tuple[int, int]:
    """Run `worker` over `items` with bounded concurrency; returns (ok, failed).

    ServerUnavailable from a worker stops the run: the remaining items are drained without work and the error is
    re-raised at the end (the caller exits non-zero; finished results are already written, a rerun resumes by id)."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 4)
    ok = failed = 0
    gone: list[ServerUnavailable] = []
    t0 = time.time()

    async def consumer():
        nonlocal ok, failed
        while True:
            item = await queue.get()
            if item is _STOP:
                queue.task_done()
                return
            if gone:
                queue.task_done()
                continue
            try:
                result = await worker(item)
            except ServerUnavailable as exc:
                gone.append(exc)
                console.print(f"[red]{label}: sunucu yok, iş duruyor ({exc})[/red]")
                queue.task_done()
                continue
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
    if gone:
        raise gone[0]
    return ok, failed


_STOP = object()
