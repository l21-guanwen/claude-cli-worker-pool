"""Integration tests for Claude CLI Worker Pool.

Requires the pool service to be running (accounts × workers):
  cd docker && WORKERS_PER_ACCOUNT=3 docker-compose up -d

All tests call the pool service (single external endpoint).
Workers are not directly accessible -- no tests call worker ports directly.

Usage:
  POOL_URL=http://localhost:8090 python -m pytest tests/test_pool.py -v
"""
from __future__ import annotations
import asyncio
import os
import uuid

import aiohttp
import pytest

POOL_URL = os.environ.get("POOL_URL", "http://localhost:8090")


def _unique_sid(prefix: str) -> str:
    """Generate a unique session ID per test run.

    Claude CLI rejects --session-id if that UUID already exists on disk.
    Using a run-unique suffix avoids collisions across test runs.
    """
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.mark.asyncio
async def test_health_check():
    """Pool service reports all workers healthy (count depends on WORKERS_PER_ACCOUNT × accounts)."""
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{POOL_URL}/health") as r:
            assert r.status == 200
            data = await r.json(content_type=None)

    assert data["status"] == "ok"
    assert data["healthy"] > 0, f"Expected at least 1 healthy worker, got: {data}"
    assert data["healthy"] == data["total"], (
        f"Not all workers are healthy: {data['healthy']}/{data['total']}"
    )
    assert "workers" in data


@pytest.mark.asyncio
async def test_complete_stateless():
    """Stateless complete returns non-empty content."""
    async with aiohttp.ClientSession() as s:
        async with s.post(
            f"{POOL_URL}/complete",
            json={"prompt": "Say 'hello' in one word."},
        ) as r:
            assert r.status == 200
            data = await r.json(content_type=None)

    assert "content" in data
    assert len(data["content"]) > 0


@pytest.mark.asyncio
async def test_session_affinity():
    """Session context is preserved across start_session -> resume_session.

    The pool service must route resume to the same worker as start.
    Claude CLI session state lives on that worker's local filesystem.
    Plain strings are accepted as session IDs (pool converts to UUID5).
    """
    session_id = _unique_sid("test-affinity")
    async with aiohttp.ClientSession() as s:
        # Start: plant a fact
        async with s.post(
            f"{POOL_URL}/start_session",
            json={
                "session_id": session_id,
                "prompt": "Remember: the magic word is 'banana'.",
            },
        ) as r:
            assert r.status == 200, f"start_session failed: {await r.text()}"

        # Resume: ask about that fact
        async with s.post(
            f"{POOL_URL}/resume_session",
            json={
                "session_id": session_id,
                "prompt": "What was the magic word?",
            },
        ) as r:
            assert r.status == 200, f"resume_session failed: {await r.text()}"
            data = await r.json(content_type=None)

    assert "banana" in data["content"].lower(), (
        f"Expected 'banana' in response, got: {data['content']}"
    )


@pytest.mark.asyncio
async def test_concurrent_sessions():
    """6 concurrent sessions start without error -- exercises pool routing."""
    sessions = [_unique_sid(f"concurrent-{i:03d}") for i in range(6)]

    async def start_one(session: aiohttp.ClientSession, sid: str) -> tuple[int, str]:
        async with session.post(
            f"{POOL_URL}/start_session",
            json={"session_id": sid, "prompt": "Say hi."},
        ) as r:
            return r.status, await r.text()

    async with aiohttp.ClientSession() as s:
        results = await asyncio.gather(*[start_one(s, sid) for sid in sessions])

    statuses = [status for status, _ in results]
    errors = [(sid, body) for (status, body), sid in zip(results, sessions) if status != 200]
    assert all(status == 200 for status in statuses), (
        f"Not all sessions returned 200: {statuses}\nErrors: {errors}"
    )


@pytest.mark.asyncio
async def test_workers_not_externally_accessible():
    """Worker containers should not be reachable via the docker internal network.

    Workers run on the internal docker network only — they have no published ports.
    Any service on localhost:8000 should NOT be a Claude worker process.
    A Claude worker health response has 'active', 'max', and 'config_dir' fields.
    """
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=2)
        ) as s:
            async with s.get("http://localhost:8000/health") as r:
                if r.status == 200:
                    data = await r.json(content_type=None)
                    # If it's a worker, it will have worker-specific fields
                    if "active" in data and "config_dir" in data:
                        pytest.fail(
                            f"Claude worker is accessible on host port :8000 — "
                            "workers should have no published ports. Response: {data}"
                        )
                    # Some other service on :8000 — that's fine
    except (aiohttp.ClientConnectorError, asyncio.TimeoutError):
        pass  # No service on :8000 — workers are internal only (ideal case)
