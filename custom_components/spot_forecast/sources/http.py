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


async def get_body(session: aiohttp.ClientSession, url: str, params: dict[str, Any]) -> bytes:
    """Raw body of ``GET url?params``, not decoded.

    Up to ``ATTEMPTS`` tries for HTTP 429, 5xx, connection errors and timeouts, with a
    back-off doubling from ``FIRST_BACKOFF``; after a 429 it waits at least
    ``RATE_LIMIT_WAIT``. Other HTTP errors fail at once. Raises ``SourceRateLimited`` if
    the last try was a 429, ``SourceNotFound`` on a 404, else ``SourceError``.
    """
    backoff = FIRST_BACKOFF
    for attempt in range(1, ATTEMPTS + 1):
        cause: Exception | None = None
        try:
            async with session.get(url, params=params, timeout=TIMEOUT) as resp:
                status = resp.status
                if status < 300:
                    return await resp.read()
        except (aiohttp.ClientError, TimeoutError) as e:
            error: SourceError = SourceError(f"{url}: {type(e).__name__}: {e}")
            cause = e
            wait = backoff
        else:
            if status == 429:
                error = SourceRateLimited(f"{url}: HTTP 429 (rate limited)")
                wait = max(backoff, RATE_LIMIT_WAIT)
            elif status >= 500:
                error = SourceError(f"{url}: HTTP {status}")
                wait = backoff
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
