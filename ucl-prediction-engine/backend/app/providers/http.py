"""Cliente HTTP con reintentos, limitación de ritmo y registro de cada consulta."""
from __future__ import annotations

import time
from typing import Any

import httpx

from ..db import FetchLog, SessionLocal


class FetchError(RuntimeError):
    pass


class HttpClient:
    def __init__(self, source: str, min_interval: float = 0.25, timeout: float = 25.0,
                 headers: dict[str, str] | None = None, log: bool = True):
        self.source = source
        self.min_interval = min_interval
        self._last = 0.0
        self.log = log
        self.client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": "ucl-prediction-engine/1.0", **(headers or {})},
            follow_redirects=True,
        )

    def _record(self, url: str, status: int | None, ok: bool, error: str | None) -> None:
        if not self.log:
            return
        try:
            with SessionLocal() as s:
                s.add(FetchLog(source=self.source, url=url, status=status, ok=ok, error=error))
                s.commit()
        except Exception:  # el registro nunca debe romper la ingesta
            pass

    def get_json(self, url: str, params: dict[str, Any] | None = None, retries: int = 3) -> Any:
        last_err: str | None = None
        for attempt in range(retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.client.get(url, params=params)
                full = str(r.request.url)
                if r.status_code == 200:
                    self._record(full, 200, True, None)
                    return r.json()
                last_err = f"HTTP {r.status_code}"
                if r.status_code in (400, 401, 403, 404):
                    self._record(full, r.status_code, False, last_err)
                    raise FetchError(f"{self.source}: {last_err} en {full}")
            except httpx.HTTPError as e:
                last_err = str(e)
            time.sleep(2 ** attempt)
        self._record(url, None, False, last_err)
        raise FetchError(f"{self.source}: fallo tras reintentos ({last_err}) en {url}")
