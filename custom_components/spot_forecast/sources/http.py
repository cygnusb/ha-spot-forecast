"""JSON GET with the retry policy shared by all sources."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

_LOGGER = logging.getLogger(__name__)

TIMEOUT = aiohttp.ClientTimeout(total=60, connect=10)
ATTEMPTS = 3
FIRST_BACKOFF = 2.0  # seconds, doubled after each failed attempt
RATE_LIMIT_WAIT = 30.0  # minimum wait after HTTP 429


class SourceError(Exception):
    """A data source could not deliver usable data."""


class SourceRateLimited(SourceError):
    """The source kept answering HTTP 429."""


async def pause(seconds: float) -> None:
    """Wait between requests; the single seam tests replace to avoid sleeping."""
    await asyncio.sleep(seconds)


async def get_json(session: aiohttp.ClientSession, url: str, params: dict[str, Any]) -> Any:
    """Decoded JSON body of ``GET url?params``.

    Up to ``ATTEMPTS`` tries for HTTP 429, 5xx, connection errors and timeouts, with a
    back-off doubling from ``FIRST_BACKOFF``; after a 429 it waits at least
    ``RATE_LIMIT_WAIT``. Other HTTP errors and undecodable bodies fail at once. Raises
    ``SourceRateLimited`` if the last try was a 429, else ``SourceError``.
    """
    backoff = FIRST_BACKOFF
    for attempt in range(1, ATTEMPTS + 1):
        cause: Exception | None = None
        try:
            async with session.get(url, params=params, timeout=TIMEOUT) as resp:
                status = resp.status
                if status < 300:
                    try:
                        return await resp.json(content_type=None)
                    except ValueError as e:
                        raise SourceError(f"{url}: invalid JSON: {e}") from e
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
            else:
                raise SourceError(f"{url}: HTTP {status}")
        if attempt == ATTEMPTS:
            raise error from cause
        _LOGGER.debug("%s (attempt %d/%d), retrying in %.0f s", error, attempt, ATTEMPTS, wait)
        await pause(wait)
        backoff *= 2
    raise AssertionError("unreachable")
