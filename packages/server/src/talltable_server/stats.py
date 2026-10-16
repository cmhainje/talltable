"""Basic request statistics for the /status page: rates, outcomes and durations per endpoint.

In memory, for this process only (the server runs a single uvicorn worker), reset on restart.
Deliberately keeps no client addresses and no query text: /status is public, and DuckDB's error
messages quote the user's SQL, so failures are recorded by category only.
"""

import html
import time

from collections import Counter, defaultdict, deque
from datetime import datetime, timezone

from starlette.requests import ClientDisconnect

WINDOWS = [("1 min", 60), ("15 min", 900), ("1 h", 3600), ("24 h", 86400)]
DURATION_WINDOW_SEC = 3600
MAX_RECORDS = 500_000  # at most this many requests are kept for the 24 h window
RECENT_FAILURES = 20

# result categories, besides "ok"
CLIENT_ERROR = "client error"            # 4xx: bad request, bad SQL, unknown partition, ...
BUSY = "busy"                            # 503: no pixel query slot in time
UNAVAILABLE = "partition unavailable"    # 503: partition mid-update
SERVER_ERROR = "server error"            # 5xx / unhandled exception
TIMEOUT = "timeout"                      # streamed query hit the query time limit
QUERY_ERROR = "query error"              # streamed query failed in DuckDB (after a 200 was sent)
DISCONNECTED = "client disconnected"     # client went away before the response was complete


class Stats:
    def __init__(self):
        self.started = time.time()
        self.records = deque(maxlen=MAX_RECORDS)  # (end time, endpoint, result, duration)
        self.totals = Counter()                   # (endpoint, result) -> count since start
        self.in_flight = Counter()
        self.failures = deque(maxlen=RECENT_FAILURES)
        self.endpoints = None                     # "METHOD /path" of the app's routes

    def endpoint(self, app, method, path):
        """the request's route, or "other" (so arbitrary URLs can't grow the tables)"""
        if self.endpoints is None:
            self.endpoints = {f"{m} {r.path}" for r in app.routes for m in getattr(r, "methods", None) or ()}
        key = f"{method} {path}"
        return key if key in self.endpoints else "other"

    def finish(self, endpoint, status, result, duration):
        if result is None:
            if status is None or status >= 500:
                result = SERVER_ERROR
            elif status >= 400:
                result = CLIENT_ERROR
            else:
                result = "ok"
        now = time.time()
        self.records.append((now, endpoint, result, duration))
        self.totals[(endpoint, result)] += 1
        if result != "ok":
            self.failures.append((now, endpoint, status, result))

    def snapshot(self, slots_total, slots_in_use, slots_waiting):
        now = time.time()
        while self.records and self.records[0][0] < now - WINDOWS[-1][1]:
            self.records.popleft()

        per = defaultdict(lambda: {"requests": {w: 0 for w, _ in WINDOWS}, "results_1h": Counter(), "durations_1h": []})
        for t, endpoint, result, duration in self.records:
            e = per[endpoint]
            for name, sec in WINDOWS:
                if t >= now - sec:
                    e["requests"][name] += 1
            if t >= now - DURATION_WINDOW_SEC:
                e["results_1h"][result] += 1
                e["durations_1h"].append(duration)

        endpoints = {}
        for endpoint in sorted(set(per) | {e for e, _ in self.totals} | set(self.in_flight)):
            e = per[endpoint]
            d = sorted(e["durations_1h"])
            endpoints[endpoint] = {
                "in_flight": self.in_flight[endpoint],
                "requests": e["requests"],
                "results_1h": dict(e["results_1h"]),
                "duration_1h_sec": {
                    "p50": _pct(d, 0.5), "p90": _pct(d, 0.9), "max": d[-1] if d else None,
                },
                "since_start": {r: n for (ep, r), n in self.totals.items() if ep == endpoint},
            }
        return {
            "now": _iso(now),
            "started": _iso(self.started),
            "uptime_sec": round(now - self.started),
            "pixel_query_slots": {"total": slots_total, "in_use": slots_in_use, "waiting": slots_waiting},
            "endpoints": endpoints,
            "recent_failures": [
                {"time": _iso(t), "endpoint": ep, "http_status": st, "result": r}
                for t, ep, st, r in reversed(self.failures)
            ],
        }


def _pct(sorted_values, q):
    if not sorted_values:
        return None
    return round(sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))], 3)


def _iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


class StatsMiddleware:
    """Records every HTTP request when its response is completely sent (for a streamed query,
    that's the end of the stream, not the first byte). The request's outcome can be refined by
    the app through `scope["state"]["stats"]["result"]` (see the categories above): a streamed
    query that fails, times out or is abandoned has already sent a 200."""

    def __init__(self, app, stats: Stats, exclude=("/status",)):
        self.app = app
        self.stats = stats
        self.exclude = set(exclude)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] in self.exclude:
            return await self.app(scope, receive, send)

        endpoint = self.stats.endpoint(scope["app"], scope["method"], scope["path"])
        rec = {"status": None, "result": None}
        scope.setdefault("state", {})["stats"] = rec

        async def send_status(message):
            if message["type"] == "http.response.start":
                rec["status"] = message["status"]
            await send(message)

        t0 = time.monotonic()
        self.stats.in_flight[endpoint] += 1
        try:
            await self.app(scope, receive, send_status)
        except (ClientDisconnect, OSError):
            rec["result"] = rec["result"] or DISCONNECTED
            raise
        except BaseException:
            rec["result"] = rec["result"] or SERVER_ERROR
            raise
        finally:
            self.stats.in_flight[endpoint] -= 1
            self.stats.finish(endpoint, rec["status"], rec["result"], time.monotonic() - t0)


def set_result(scope, result):
    """mark the current request's outcome (no-op outside StatsMiddleware)"""
    rec = scope.get("state", {}).get("stats")
    if rec is not None and rec["result"] is None:
        rec["result"] = result


def render_html(snap):
    esc = html.escape
    windows = [w for w, _ in WINDOWS]
    results = sorted({r for e in snap["endpoints"].values() for r in e["results_1h"]} - {"ok"})
    slots = snap["pixel_query_slots"]

    def fmt(x):
        return "–" if x is None else f"{x:.2f}"

    rows = []
    for endpoint, e in snap["endpoints"].items():
        d = e["duration_1h_sec"]
        rows.append(
            "<tr>"
            + f"<td>{esc(endpoint)}</td><td>{e['in_flight']}</td>"
            + "".join(f"<td>{e['requests'][w]}</td>" for w in windows)
            + f"<td>{e['results_1h'].get('ok', 0)}</td>"
            + "".join(f"<td>{e['results_1h'].get(r, 0)}</td>" for r in results)
            + f"<td>{fmt(d['p50'])}</td><td>{fmt(d['p90'])}</td><td>{fmt(d['max'])}</td>"
            + "</tr>"
        )
    failures = "".join(
        f"<tr><td>{esc(f['time'])}</td><td>{esc(f['endpoint'])}</td><td>{f['http_status'] or '–'}</td><td>{esc(f['result'])}</td></tr>"
        for f in snap["recent_failures"]
    ) or '<tr><td colspan="4">none</td></tr>'
    head = (
        "<tr><th rowspan=2>endpoint</th><th rowspan=2>in flight</th>"
        f"<th colspan={len(windows)}>requests</th><th colspan={1 + len(results)}>results, last hour</th>"
        "<th colspan=3>duration, last hour (s)</th></tr><tr>"
        + "".join(f"<th>{w}</th>" for w in windows)
        + "<th>ok</th>" + "".join(f"<th>{esc(r)}</th>" for r in results)
        + "<th>p50</th><th>p90</th><th>max</th></tr>"
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="15">
<title>talltable status</title>
<style>
body {{ font: 14px system-ui, sans-serif; margin: 1.5em; color: #222; background: #fff; }}
table {{ border-collapse: collapse; margin: 0.5em 0 1.5em; }}
th, td {{ border: 1px solid #ccc; padding: 3px 8px; text-align: right; }}
th {{ background: #f3f3f3; }} td:first-child {{ text-align: left; }}
@media (prefers-color-scheme: dark) {{ body {{ color: #ddd; background: #161616; }} th {{ background: #262626; }} th, td {{ border-color: #444; }} }}
</style></head><body>
<h1>talltable status</h1>
<p>up since {esc(snap['started'])} ({snap['uptime_sec'] // 3600} h {snap['uptime_sec'] % 3600 // 60} min);
as of {esc(snap['now'])}; refreshes every 15 s; <a href="?format=json">json</a></p>
<p><b>pixel query slots:</b> {slots['in_use']} of {slots['total']} in use, {slots['waiting']} waiting</p>
<table>{head}{''.join(rows)}</table>
<h2>recent failures</h2>
<table><tr><th>time (UTC)</th><th>endpoint</th><th>HTTP</th><th>result</th></tr>{failures}</table>
</body></html>"""
