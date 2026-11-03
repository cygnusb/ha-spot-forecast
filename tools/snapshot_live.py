#!/usr/bin/env python3
"""Snapshot a real Home Assistant installation for the card screenshots.

Reads, over the WebSocket API, what the example cards read: the states of the price
sensor, the peak-risk binary sensor and ``sun.sun``, the hourly long-term statistics of
the actual price and the six forecast horizons, and the price sensor's recent history.
Writes them in the scenario format of ``card_scenarios.py`` to
tools/scenarios/live.json.gz, which ``gen_card_screenshots.py`` renders (scenario
``live``). Read-only: nothing on the instance changes.

The surcharge option (grid fees and taxes) is a personal setting: it is set to 0 in the
snapshot unless ``--keep-surcharge`` is given, so the card shows the exchange price.
``sun.sun`` (which would tell the instance's location) is computed for the middle of
Germany as in ``card_scenarios.py``, and the states lose their ``context``.

Usage:
    HASS_SERVER=https://ha.example HASS_TOKEN=... .venv-ha/bin/python tools/snapshot_live.py
        [--zone de_lu] [--keep-surcharge]
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime, timedelta
import gzip
import json
import os
from pathlib import Path
import ssl
import sys
from zoneinfo import ZoneInfo

import aiohttp

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.card_scenarios import _sun  # noqa: E402

OUT = REPO / "tools" / "scenarios" / "live.json.gz"
HORIZONS = range(1, 7)
STATISTICS_DAYS = 190  # the look-back card's longest range (6 months), with a margin
HISTORY_DAYS = 3  # as card_scenarios: the price sensor's recorded state, three days back


class Client:
    def __init__(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        self._ws = ws
        self._id = 0

    async def call(self, message: dict) -> object:
        self._id += 1
        await self._ws.send_json({"id": self._id, **message})
        while True:
            reply = await self._ws.receive_json()
            if reply.get("id") == self._id and reply.get("type") == "result":
                if not reply["success"]:
                    raise RuntimeError(f"{message['type']}: {reply['error']}")
                return reply["result"]


async def snapshot(server: str, token: str, zone: str, keep_surcharge: bool) -> dict:
    price = f"sensor.spot_forecast_{zone}_price"
    peak = f"binary_sensor.spot_forecast_{zone}_peak_risk"
    statistic_ids = [f"spot_forecast:{zone}_price"] + [
        f"spot_forecast:{zone}_forecast_h{h}" for h in HORIZONS
    ]
    url = server.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
    try:  # the python.org builds come without CA certificates
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    async with (
        aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=context)) as session,
        session.ws_connect(f"{url}/api/websocket", max_msg_size=0) as ws,
    ):
        await ws.receive_json()  # auth_required
        await ws.send_json({"type": "auth", "access_token": token})
        if (await ws.receive_json())["type"] != "auth_ok":
            raise RuntimeError("authentication failed")
        client = Client(ws)
        config = await client.call({"type": "get_config"})
        now = datetime.now(UTC).replace(microsecond=0)
        wanted = {price, peak}
        states = {s["entity_id"]: s for s in await client.call({"type": "get_states"})}
        missing = wanted - states.keys()
        if missing:
            raise RuntimeError(f"not on the instance: {', '.join(sorted(missing))}")
        statistics = await client.call(
            {
                "type": "recorder/statistics_during_period",
                "start_time": (now - timedelta(days=STATISTICS_DAYS)).isoformat(),
                "end_time": (now + timedelta(days=8)).isoformat(),
                "statistic_ids": statistic_ids,
                "period": "hour",
                "types": ["mean", "min", "max", "state"],
            }
        )
        history = await client.call(
            {
                "type": "history/history_during_period",
                "start_time": (now - timedelta(days=HISTORY_DAYS)).isoformat(),
                "end_time": now.isoformat(),
                "entity_ids": [price],
                "minimal_response": True,
                "no_attributes": True,
            }
        )
    keep = {k: {f: v for f, v in states[k].items() if f != "context"} for k in sorted(wanted)}
    keep["sun.sun"] = _sun(now)
    if not keep_surcharge:
        keep[price]["attributes"]["surcharge"] = 0.0
    tz = config["time_zone"]
    local = now.astimezone(ZoneInfo(tz))
    return {
        "now": local.isoformat(),
        "now_ms": int(now.timestamp() * 1000),
        "time_zone": tz,
        "local_today": local.date().isoformat(),
        "source": f"Home Assistant {config['version']}, snapshot {now.isoformat()}",
        "states": keep,
        "statistics": statistics,
        "history": {price: history.get(price, [])},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--zone", default="de_lu", help="the zone slug in the entity ids")
    parser.add_argument("--keep-surcharge", action="store_true")
    args = parser.parse_args()
    server, token = os.environ.get("HASS_SERVER"), os.environ.get("HASS_TOKEN")
    if not server or not token:
        parser.error("set HASS_SERVER and HASS_TOKEN")
    data = asyncio.run(snapshot(server, token, args.zone, args.keep_surcharge))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(gzip.compress(json.dumps(data, separators=(",", ":")).encode(), mtime=0))
    rows = {k.split(":")[1]: len(v) for k, v in data["statistics"].items()}
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size // 1024} KiB) at {data['now']}")
    print(f"statistics rows: {rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
