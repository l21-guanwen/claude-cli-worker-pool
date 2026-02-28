"""Claude CLI distributed worker pool — dispatches to HTTP worker containers.

Routing strategy: Weighted Least Connections + EWMA response time.
  score = (active_calls + 1) × avg_response_ms
New sessions and stateless completes go to the lowest-score worker.
Resume sessions always route to their pinned worker (session affinity).

Zero framework dependencies — only aiohttp.
"""
from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)

# EWMA decay factor: 0.2 means ~20% weight on the latest sample.
# Higher = more reactive to recent changes; lower = smoother/more stable.
_EWMA_ALPHA = 0.2

# Initial avg_response_ms before any real data — set high so all workers
# start equal and the pool falls back to least-active-calls ordering.
_INITIAL_RESPONSE_MS = 5000.0


@dataclass
class PoolResponse:
    content: str
    error: str | None = None


@dataclass
class WorkerStats:
    url: str
    active_calls: int
    total_requests: int
    avg_response_ms: float
    score: float
    is_healthy: bool


class WorkerClient:
    """HTTP client for a single Claude CLI worker container.

    Tracks EWMA latency and active call count for smart routing.
    score = (active_calls + 1) × avg_response_ms — lower is better.
    EWMA is only updated on successful calls to avoid skewing latency low on errors.
    """

    def __init__(self, base_url: str, max_concurrent: int = 5) -> None:
        self._url = base_url.rstrip("/")
        self._sem = asyncio.Semaphore(max_concurrent)
        self._active = 0
        self._total_requests = 0
        self._avg_response_ms = _INITIAL_RESPONSE_MS
        self._is_healthy = True  # updated by pool health checks
        self._session: aiohttp.ClientSession | None = None  # lazy singleton

    @property
    def active(self) -> int:
        return self._active

    @property
    def score(self) -> float:
        """Routing score — lower is better. inf if unhealthy."""
        if not self._is_healthy:
            return float("inf")
        return (self._active + 1) * self._avg_response_ms

    @property
    def stats(self) -> WorkerStats:
        return WorkerStats(
            url=self._url,
            active_calls=self._active,
            total_requests=self._total_requests,
            avg_response_ms=round(self._avg_response_ms, 1),
            score=round(self.score, 1),
            is_healthy=self._is_healthy,
        )

    def _update_latency(self, elapsed_ms: float) -> None:
        """Update EWMA with a new latency sample (success only)."""
        self._avg_response_ms = (
            _EWMA_ALPHA * elapsed_ms + (1 - _EWMA_ALPHA) * self._avg_response_ms
        )

    def _get_session(self, timeout: float = 300.0) -> aiohttp.ClientSession:
        """Lazily create and reuse a single ClientSession per worker."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout + 10),
            )
        return self._session

    async def close(self) -> None:
        """Close the persistent HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def health(self) -> bool:
        try:
            s = self._get_session(timeout=5)
            async with s.get(f"{self._url}/health") as r:
                self._is_healthy = r.status == 200
                return self._is_healthy
        except Exception:
            self._is_healthy = False
            return False

    async def complete(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        max_tokens: int,
        timeout: float,
    ) -> PoolResponse:
        payload = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "model": model,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        return await self._post("/complete", payload, timeout)

    async def start_session(
        self,
        session_id: str,
        prompt: str,
        system_prompt: str,
        model: str,
        max_tokens: int,
        timeout: float,
    ) -> PoolResponse:
        payload = {
            "session_id": session_id,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "model": model,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        return await self._post("/start_session", payload, timeout)

    async def resume_session(
        self,
        session_id: str,
        prompt: str,
        timeout: float,
    ) -> PoolResponse:
        payload = {"session_id": session_id, "prompt": prompt, "timeout": timeout}
        return await self._post("/resume_session", payload, timeout)

    async def _post(self, endpoint: str, payload: dict, timeout: float) -> PoolResponse:
        async with self._sem:
            self._active += 1
            t0 = time.monotonic()
            success = False
            try:
                s = self._get_session(timeout)
                async with s.post(f"{self._url}{endpoint}", json=payload) as r:
                    if r.status != 200:
                        text = await r.text()
                        # Don't update EWMA on errors — they skew latency down
                        return PoolResponse(content="", error=f"{r.status}: {text}")
                    data = await r.json()
                    success = True
                    return PoolResponse(content=data.get("content", ""))
            except Exception as e:
                return PoolResponse(content="", error=str(e))
            finally:
                elapsed_ms = (time.monotonic() - t0) * 1000
                if success:
                    self._update_latency(elapsed_ms)
                    self._total_requests += 1
                self._active -= 1


class ClaudeCLIPool:
    """Distributed Claude CLI pool — smart routing + session affinity.

    Routing for new sessions (complete / start_session):
      Picks the worker with the lowest score = (active_calls + 1) × avg_response_ms.
      This naturally routes away from slow or overloaded workers.

    Routing for resume_session:
      Always routes to the pinned worker (session_id → worker_index map).
      Claude CLI stores session state locally, so affinity is mandatory.

    Args:
        worker_urls: List of worker HTTP base URLs.
        model: Model identifier for claude -p calls.
        max_tokens: Max tokens per call.
        timeout: Request timeout in seconds.
        max_concurrent_per_worker: Semaphore size per worker (default: 5).
    """

    def __init__(
        self,
        worker_urls: list[str],
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 8192,
        timeout: float = 300.0,
        max_concurrent_per_worker: int = 5,
    ) -> None:
        if not worker_urls:
            raise ValueError("worker_urls must not be empty")
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._workers = [
            WorkerClient(url, max_concurrent_per_worker) for url in worker_urls
        ]
        self._session_map: dict[str, int] = {}  # session_id -> worker_index
        self._lock = asyncio.Lock()
        self._complete_rr: int = 0  # round-robin counter for stateless complete() calls
        logger.info(
            f"ClaudeCLIPool: {len(self._workers)} workers, "
            f"{max_concurrent_per_worker} concurrent/worker"
        )

    def _pick_worker(self) -> int:
        """Pick the worker with the lowest routing score (fastest + least loaded).

        Score = (active_calls + 1) × avg_response_ms — lower is better.
        Unhealthy workers get score=inf and are never picked.
        Falls back to worker-0 if all workers are unhealthy.
        """
        best_idx = 0
        best_score = float("inf")
        for i, w in enumerate(self._workers):
            s = w.score
            if s < best_score:
                best_score = s
                best_idx = i
        if best_score == float("inf"):
            logger.warning(
                "[pool] All workers unhealthy — routing to worker-0 as fallback"
            )
        else:
            logger.debug(f"[pool] Picked worker-{best_idx} (score={best_score:.0f})")
        return best_idx

    async def complete(self, prompt: str, system_prompt: str = "", model: str = "") -> PoolResponse:
        # Round-robin across all workers for stateless calls — ensures even account usage.
        # EWMA scoring is not used here because it tends to cluster on low-index workers,
        # starving higher-index workers (and their accounts) of traffic.
        async with self._lock:
            idx = self._complete_rr % len(self._workers)
            self._complete_rr += 1
        worker = self._workers[idx]
        logger.debug(f"[pool] complete → worker-{idx} (rr)")
        return await worker.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model or self._model,
            max_tokens=self._max_tokens,
            timeout=self._timeout,
        )

    async def start_session(
        self,
        session_id: str,
        prompt: str,
        system_prompt: str = "",
        model: str = "",
    ) -> PoolResponse:
        async with self._lock:
            worker_idx = self._pick_worker()
            self._session_map[session_id] = worker_idx
        worker = self._workers[worker_idx]
        logger.debug(
            f"[pool] start_session {session_id[:8]} → worker-{worker_idx} "
            f"(score={worker.score:.0f})"
        )
        return await worker.start_session(
            session_id=session_id,
            prompt=prompt,
            system_prompt=system_prompt,
            model=model or self._model,
            max_tokens=self._max_tokens,
            timeout=self._timeout,
        )

    async def resume_session(self, session_id: str, prompt: str) -> PoolResponse:
        async with self._lock:
            worker_idx = self._session_map.get(session_id)
        if worker_idx is None:
            raise RuntimeError(
                f"[pool] Unknown session: {session_id} — call start_session first"
            )
        worker = self._workers[worker_idx]
        logger.debug(
            f"[pool] resume_session {session_id[:8]} → worker-{worker_idx} (pinned)"
        )
        return await worker.resume_session(
            session_id=session_id,
            prompt=prompt,
            timeout=self._timeout,
        )

    async def health_check(self) -> dict:
        """Parallel health check all workers. Updates each worker's is_healthy flag."""
        results = await asyncio.gather(
            *[w.health() for w in self._workers],
            return_exceptions=True,
        )
        healthy = sum(1 for r in results if r is True)
        worker_stats = [w.stats.__dict__ for w in self._workers]
        logger.info(f"[pool] health: {healthy}/{len(self._workers)} workers OK")
        return {
            "healthy": healthy,
            "total": len(self._workers),
            "workers": worker_stats,
        }

    async def close(self) -> None:
        """Close all worker HTTP sessions."""
        await asyncio.gather(*[w.close() for w in self._workers])

    @property
    def worker_count(self) -> int:
        return len(self._workers)

    @property
    def active_calls(self) -> int:
        return sum(w.active for w in self._workers)
