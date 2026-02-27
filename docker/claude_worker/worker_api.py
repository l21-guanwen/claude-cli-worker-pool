"""Claude CLI worker API — HTTP wrapper around claude -p subprocess.

Internal-only service: no external ports published. Accessed only from
the pool service container on the docker internal network.

Critical: --tools "" is always passed to prevent built-in tool use,
which would hit --max-turns 1 and cause error_max_turns failures.
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import subprocess

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="ClaudeWorker")
logger = logging.getLogger("worker")

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
CLAUDE_CONFIG_DIR = os.environ.get("CLAUDE_CONFIG_DIR", "")
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "5"))

_semaphore: asyncio.Semaphore | None = None
_active = 0


@app.on_event("startup")
async def startup() -> None:
    global _semaphore
    _semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    logger.info(f"Worker started: max_concurrent={MAX_CONCURRENT}, "
                f"config_dir={CLAUDE_CONFIG_DIR or 'default'}")


# -- Request models --

class CompleteRequest(BaseModel):
    prompt: str
    system_prompt: str = ""
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 8192
    timeout: float = 300.0


class SessionRequest(BaseModel):
    session_id: str
    prompt: str
    system_prompt: str = ""
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 8192
    timeout: float = 300.0


class ResumeRequest(BaseModel):
    session_id: str
    prompt: str
    timeout: float = 300.0


# -- Claude CLI invocation --

def _run_claude(cmd: list[str], prompt: str, timeout: float) -> dict:
    """Run claude -p synchronously (called via run_in_executor).

    Claude CLI always writes JSON to stdout even on failure (is_error: true).
    We parse stdout first; only fall back to stderr if stdout is not valid JSON.
    """
    env = os.environ.copy()
    if CLAUDE_CONFIG_DIR:
        env["CLAUDE_CONFIG_DIR"] = CLAUDE_CONFIG_DIR
    try:
        result = subprocess.run(
            cmd,
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            env=env,
        )
        stdout = result.stdout.decode("utf-8", errors="replace").strip()
        stderr = result.stderr.decode("utf-8", errors="replace").strip()

        # Claude CLI writes JSON to stdout even on errors (is_error: true, exit code 1).
        # Always try to parse stdout first.
        if stdout:
            try:
                data = json.loads(stdout)
                if data.get("is_error"):
                    return {"error": data.get("result", "unknown error"), "content": ""}
                return {"content": data.get("result", data.get("content", "")), "raw": data}
            except json.JSONDecodeError:
                pass

        if result.returncode != 0:
            return {"error": f"exit {result.returncode}: {stderr or stdout}", "content": ""}

        return {"error": "empty response", "content": ""}

    except subprocess.TimeoutExpired:
        return {"error": "timeout", "content": ""}
    except Exception as e:
        return {"error": str(e), "content": ""}


async def _invoke(cmd: list[str], prompt: str, timeout: float) -> dict:
    """Acquire semaphore and run claude in thread pool."""
    global _active
    async with _semaphore:
        _active += 1
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, _run_claude, cmd, prompt, timeout)
        finally:
            _active -= 1


# -- Endpoints --

@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "active": _active,
        "max": MAX_CONCURRENT,
        "config_dir": CLAUDE_CONFIG_DIR or "default",
    }


@app.post("/complete")
async def complete(req: CompleteRequest) -> dict:
    """Stateless single-turn completion."""
    cmd = [
        CLAUDE_BIN, "-p",
        "--output-format", "json",
        "--max-turns", "1",
        "--model", req.model,
        "--no-session-persistence",
        "--tools", "",
    ]
    prompt = req.prompt
    if req.system_prompt:
        prompt = f"<system-context>\n{req.system_prompt}\n</system-context>\n\n{req.prompt}"
    result = await _invoke(cmd, prompt, req.timeout)
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    return {"content": result["content"]}


@app.post("/start_session")
async def start_session(req: SessionRequest) -> dict:
    """Start a named session (session_id persisted by claude CLI locally)."""
    cmd = [
        CLAUDE_BIN, "-p",
        "--output-format", "json",
        "--max-turns", "1",
        "--model", req.model,
        "--session-id", req.session_id,
        "--tools", "",
    ]
    prompt = req.prompt
    if req.system_prompt:
        prompt = f"<system-context>\n{req.system_prompt}\n</system-context>\n\n{req.prompt}"
    result = await _invoke(cmd, prompt, req.timeout)
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    return {"content": result["content"]}


@app.post("/resume_session")
async def resume_session(req: ResumeRequest) -> dict:
    """Resume an existing session. Must be routed to the same worker as start_session."""
    cmd = [
        CLAUDE_BIN, "-p",
        "--output-format", "json",
        "--max-turns", "1",
        "--resume", req.session_id,
        "--tools", "",
    ]
    result = await _invoke(cmd, req.prompt, req.timeout)
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    return {"content": result["content"]}
