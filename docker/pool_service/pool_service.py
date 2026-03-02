"""Claude CLI Pool Service — load balancer for worker containers.

Single external endpoint at :8090. Clients never call workers directly.
Session affinity (session_id -> worker_index) is maintained here.
Smart routing: EWMA latency x active calls scoring picks the best worker.

Worker URL expansion (Account × Workers model):
  ACCOUNT_BASE_URLS = "http://account-1,http://account-2"
  WORKERS_PER_ACCOUNT = 3
  BASE_PORT = 8000
  → worker_urls = [
      "http://account-1:8000", "http://account-1:8001", "http://account-1:8002",
      "http://account-2:8000", "http://account-2:8001", "http://account-2:8002",
    ]  (6 workers total — EWMA routes across all 6)

Session IDs: Claude CLI requires valid UUIDs. Callers may pass any string;
the pool service converts it to a deterministic UUID5 transparently.
"""
from __future__ import annotations
import logging
import os
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from claude_cli_pool import ClaudeCLIPool

app = FastAPI(title="ClaudeCLIPoolService")
logger = logging.getLogger("pool_service")

_pool: ClaudeCLIPool | None = None

_SESSION_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # uuid.NAMESPACE_URL


def _to_uuid(session_id: str) -> str:
    """Convert any string to a deterministic UUID5. No-op if already a valid UUID."""
    try:
        uuid.UUID(session_id)
        return session_id
    except ValueError:
        return str(uuid.uuid5(_SESSION_NS, session_id))


def _expand_worker_urls(
    account_base_urls: list[str],
    workers_per_account: int,
    base_port: int,
) -> list[str]:
    """Expand account base URLs into a flat list of worker URLs.

    Each account runs WORKERS_PER_ACCOUNT processes on consecutive ports
    starting at BASE_PORT. Example:
      account_base_urls = ["http://account-1", "http://account-2"]
      workers_per_account = 3, base_port = 8000
      →  ["http://account-1:8000", "http://account-1:8001", "http://account-1:8002",
           "http://account-2:8000", "http://account-2:8001", "http://account-2:8002"]
    """
    return [
        f"{acct.rstrip('/')}:{base_port + i}"
        for acct in account_base_urls
        for i in range(workers_per_account)
    ]


@app.on_event("startup")
async def startup() -> None:
    global _pool

    # Support both old WORKER_URLS (flat list) and new ACCOUNT_BASE_URLS × WORKERS_PER_ACCOUNT.
    if os.environ.get("WORKER_URLS"):
        # Legacy flat list — used in tests or single-worker setups
        worker_urls = [u.strip() for u in os.environ["WORKER_URLS"].split(",") if u.strip()]
        logger.info(f"[pool] Using WORKER_URLS directly: {worker_urls}")
    else:
        account_urls_raw = os.environ.get("ACCOUNT_BASE_URLS", "")
        if not account_urls_raw:
            raise RuntimeError("Either WORKER_URLS or ACCOUNT_BASE_URLS environment variable is required")
        account_urls = [u.strip() for u in account_urls_raw.split(",") if u.strip()]
        workers_per = int(os.environ.get("WORKERS_PER_ACCOUNT", "1"))
        base_port = int(os.environ.get("BASE_PORT", "8000"))
        worker_urls = _expand_worker_urls(account_urls, workers_per, base_port)
        logger.info(
            f"[pool] Expanded {len(account_urls)} account(s) × {workers_per} workers → "
            f"{len(worker_urls)} total workers: {worker_urls}"
        )

    _pool = ClaudeCLIPool(
        worker_urls=worker_urls,
        model=os.environ.get("MODEL", "claude-sonnet-4-6"),
        max_tokens=int(os.environ.get("MAX_TOKENS", "8192")),
        timeout=float(os.environ.get("TIMEOUT", "300")),
        max_concurrent_per_worker=int(os.environ.get("MAX_CONCURRENT_PER_WORKER", "5")),
    )
    logger.info(f"Pool service started: {len(worker_urls)} workers")


@app.on_event("shutdown")
async def shutdown() -> None:
    if _pool:
        await _pool.close()


# -- Request models --

class CompleteRequest(BaseModel):
    prompt: str
    system_prompt: str = ""
    model: str = ""  # per-request override; empty = use pool default


class SessionRequest(BaseModel):
    session_id: str
    prompt: str
    system_prompt: str = ""
    model: str = ""  # per-request override; empty = use pool default


class ResumeRequest(BaseModel):
    session_id: str
    prompt: str


class CloseRequest(BaseModel):
    session_id: str


# -- Endpoints --

@app.get("/health")
async def health() -> dict:
    result = await _pool.health_check()
    return {"status": "ok", "active_calls": _pool.active_calls, **result}


@app.post("/complete")
async def complete(req: CompleteRequest) -> dict:
    resp = await _pool.complete(req.prompt, req.system_prompt, model=req.model)
    if resp.error:
        raise HTTPException(status_code=500, detail=resp.error)
    return {"content": resp.content}


@app.post("/start_session")
async def start_session(req: SessionRequest) -> dict:
    session_uuid = _to_uuid(req.session_id)
    resp = await _pool.start_session(session_uuid, req.prompt, req.system_prompt, model=req.model)
    if resp.error:
        raise HTTPException(status_code=500, detail=resp.error)
    return {"content": resp.content}


@app.post("/resume_session")
async def resume_session(req: ResumeRequest) -> dict:
    session_uuid = _to_uuid(req.session_id)
    resp = await _pool.resume_session(session_uuid, req.prompt)
    if resp.error:
        raise HTTPException(status_code=500, detail=resp.error)
    return {"content": resp.content}


@app.post("/close_session")
async def close_session(req: CloseRequest) -> dict:
    session_uuid = _to_uuid(req.session_id)
    resp = await _pool.close_session(session_uuid)
    return resp
