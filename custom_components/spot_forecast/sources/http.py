"""JSON GET with the retry policy shared by all sources."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import json
import logging
from typing import Any

import aiohttp

_LOGGER = logging.getLogger(__name__)

TIMEOUT = aiohttp.ClientTimeout(total=60, connect=10)
ATTEMPTS = 3
FIRST_BACKOFF = 2.0  # seconds, doubled after each failed attempt
RATE_LIMIT_WAIT = 30.0  # minimum wait after HTTP 429
MAX_RETRY_AFTER = 120.0  # cap of a Retry-After the server asks for (seconds)
_REASON_BYTES = 4096  # of a client error's body, read for its explanation
_REASON_CHARS = 300
_CHUNK = 64 * 1024  # read size while a body is checked against a limit


# ``hass.async_add_executor_job``: runs ``func(*args)`` in a worker thread. The sources decode
# and parse their payloads through it, so large documents never block the event loop.
RunBlocking = Callable[..., Awaitable[Any]]


class SourceError(Exception):
    """A data source could not deliver usable data."""


class SourceRateLimited(SourceError):
    """The source kept answering HTTP 429."""


class SourceNotFound(SourceError):
    """The source answered HTTP 404: what was asked for does not exist (yet)."""


async def pause(seconds: float) -> None:
    """Wait between requests; the single seam tests replace to avoid sleeping."""
    await asyncio.sleep(seconds)


def decode_json(body: bytes, url: str) -> Any:
    """``body`` as JSON; ``SourceError`` if it is not (blocking for large bodies)."""
    try:
        return json.loads(body)
    except ValueError as e:
        raise SourceError(f"{url}: invalid JSON: {e}") from e


async def _read(resp: aiohttp.ClientResponse, url: str, max_bytes: int | None) -> bytes:
    """The body, or ``SourceError`` once it is known to exceed ``max_bytes``."""
    if max_bytes is None:
        return await resp.read()
    try:
        declared = int(resp.headers.get("Content-Length", ""))
    except ValueError:
        declared = None  # absent (chunked) or garbled: count while reading
    if declared is not None and declared > max_bytes:
        raise SourceError(f"{url}: {declared} bytes, more than the limit of {max_bytes}")
    body = bytearray()
    async for chunk in resp.content.iter_chunked(_CHUNK):
        body += chunk
        if len(body) > max_bytes:
            raise SourceError(f"{url}: more than the limit of {max_bytes} bytes")
    return bytes(body)


def _retry_after(value: str | None) -> float:
    """Seconds of a ``Retry-After`` header in its seconds form, capped; 0 if absent or a date."""
    try:
        return min(max(float(int(value or "")), 0.0), MAX_RETRY_AFTER)
    except ValueError:
        return 0.0


async def _reason(resp: aiohttp.ClientResponse, url: str) -> str:
    """``": <reason>"`` of a JSON error body such as Open-Meteo's, else ``""``.

    Open-Meteo explains a rejected request (HTTP 400) as ``{"error": true, "reason": ...}``.
    """
    try:
        doc = json.loads(await _read(resp, url, _REASON_BYTES))
    except (SourceError, aiohttp.ClientError, TimeoutError, ValueError):
        return ""
    reason = doc.get("reason") if isinstance(doc, dict) else None
    return f": {reason[:_REASON_CHARS]}" if isinstance(reason, str) and reason else ""


async def get_body(
    session: aiohttp.ClientSession,
    url: str,
    params: dict[str, Any],
    max_bytes: int | None = None,
) -> bytes:
    """Raw body of ``GET url?params``, not decoded.

    Up to ``ATTEMPTS`` tries for HTTP 429, 5xx, connection errors and timeouts, with a
    back-off doubling from ``FIRST_BACKOFF``; after a 429 it waits at least
    ``RATE_LIMIT_WAIT``, and after a 429 or 503 at least the ``Retry-After`` seconds (at
    most ``MAX_RETRY_AFTER``). Other HTTP errors fail at once, with the ``reason`` of a
    JSON error body in the message. Raises ``SourceRateLimited`` if the last try was a
    429, ``SourceNotFound`` on a 404, else ``SourceError`` (also, not retried, for a body
    longer than ``max_bytes``: declared, or found while reading).
    """
    backoff = FIRST_BACKOFF
    for attempt in range(1, ATTEMPTS + 1):
        cause: Exception | None = None
        try:
            async with session.get(url, params=params, timeout=TIMEOUT) as resp:
                status = resp.status
                if status < 300:
                    return await _read(resp, url, max_bytes)
                retry_after = _retry_after(resp.headers.get("Retry-After"))
                if 400 <= status < 500 and status not in (404, 429):
                    raise SourceError(f"{url}: HTTP {status}{await _reason(resp, url)}")
        except (aiohttp.ClientError, TimeoutError) as e:
            error: SourceError = SourceError(f"{url}: {type(e).__name__}: {e}")
            cause = e
            wait = backoff
        else:
            if status == 429:
                error = SourceRateLimited(f"{url}: HTTP 429 (rate limited)")
                wait = max(backoff, RATE_LIMIT_WAIT, retry_after)
            elif status >= 500:
                error = SourceError(f"{url}: HTTP {status}")
                wait = max(backoff, retry_after) if status == 503 else backoff
            elif status == 404:
                raise SourceNotFound(f"{url}: HTTP 404")
            else:
                raise SourceError(f"{url}: HTTP {status}")
        if attempt == ATTEMPTS:
            raise error from cause
        _LOGGER.debug("%s (attempt %d/%d), retrying in %.0f s", error, attempt, ATTEMPTS, wait)
        await pause(wait)
        backoff *= 2
    raise AssertionError("unreachable")
