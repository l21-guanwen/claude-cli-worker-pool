"""Claude CLI worker API — HTTP wrapper around claude -p subprocess.

Internal-only service: no external ports published. Accessed only from
the pool service container on the docker internal network.

Hybrid execution model:
  /complete        — stateless subprocess.run (one process per call, exits immediately)
  /start_session   — spawns a persistent streaming subprocess (stays alive)
  /resume_session  — reuses the streaming subprocess (zero respawn overhead)
  /close_session   — explicitly kills a streaming subprocess

The streaming subprocess uses `claude -p --input-format stream-json --output-format
stream-json`, which keeps Node.js alive between messages. This eliminates the ~3-4s
subprocess spawn overhead on every /resume_session call.

Critical: --tools "" is always passed to prevent built-in tool use,
which would hit --max-turns 1 and cause error_max_turns failures.
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import subprocess
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="ClaudeWorker")
logger = logging.getLogger("worker")

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
CLAUDE_CONFIG_DIR = os.environ.get("CLAUDE_CONFIG_DIR", "")
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "5"))
SESSION_IDLE_TIMEOUT = float(os.environ.get("SESSION_IDLE_TIMEOUT", "600"))  # 10 min

_semaphore: asyncio.Semaphore | None = None
_active = 0


# ============================================================
# Streaming Session Manager
# ============================================================

class StreamingSession:
    """Persistent Claude CLI streaming subprocess for a session.

    Spawns `claude -p --input-format stream-json --output-format stream-json`
    once and keeps it alive for multiple resume calls. Each user message is
    written to stdin as NDJSON; responses are read from stdout until a
    `result` message arrives.

    Per-message flow:
      stdin  → {"type":"user","message":{"role":"user","content":"..."},...}
      stdout ← {"type":"system",...}      (first message only — session init)
      stdout ← {"type":"assistant",...}   (model response)
      stdout ← {"type":"result",...}      (terminal — response complete)

    Eliminates the ~3-4s subprocess spawn overhead on every resume call.
    """

    def __init__(self, session_id: str, model: str, system_prompt: str = "") -> None:
        self.session_id = session_id
        self._model = model
        self._system_prompt = system_prompt
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()  # one message at a time
        self._cli_session_id: str | None = None  # assigned by CLI system init
        self.last_used = time.monotonic()
        self.total_messages = 0

    @property
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start_and_send(self, prompt: str, timeout: float) -> dict:
        """Spawn the streaming subprocess and send the first user message."""
        async with self._lock:
            cmd = [
                CLAUDE_BIN, "-p",
                "--input-format", "stream-json",
                "--output-format", "stream-json",
                "--verbose",
                "--model", self._model,
                "--tools", "",
            ]
            env = os.environ.copy()
            if CLAUDE_CONFIG_DIR:
                env["CLAUDE_CONFIG_DIR"] = CLAUDE_CONFIG_DIR

            logger.info(
                f"[stream] spawning subprocess for session {self.session_id[:8]}... "
                f"model={self._model}"
            )
            t0 = time.monotonic()
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # Prepend system context to first message (same as /complete)
            full_prompt = prompt
            if self._system_prompt:
                full_prompt = (
                    f"<system-context>\n{self._system_prompt}\n</system-context>"
                    f"\n\n{prompt}"
                )

            result = await self._send_and_read(full_prompt, timeout)
            elapsed = time.monotonic() - t0
            logger.info(
                f"[stream] session {self.session_id[:8]} started in {elapsed:.1f}s "
                f"(cli_session={self._cli_session_id})"
            )
            return result

    async def resume(self, prompt: str, timeout: float) -> dict:
        """Send a follow-up message to the existing streaming subprocess."""
        async with self._lock:
            if not self.is_alive:
                return {"error": "streaming process died", "content": ""}

            t0 = time.monotonic()
            result = await self._send_and_read(prompt, timeout)
            elapsed = time.monotonic() - t0
            logger.info(
                f"[stream] session {self.session_id[:8]} resume #{self.total_messages} "
                f"in {elapsed:.1f}s (no respawn)"
            )
            return result

    async def _send_and_read(self, prompt: str, timeout: float) -> dict:
        """Write user message to stdin and read stdout until result message."""
        self.last_used = time.monotonic()
        self.total_messages += 1

        msg = {
            "type": "user",
            "message": {"role": "user", "content": prompt},
            "session_id": self._cli_session_id or "default",
            "parent_tool_use_id": None,
        }
        line = json.dumps(msg, ensure_ascii=False) + "\n"

        try:
            self._proc.stdin.write(line.encode("utf-8"))
            await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            return {"error": f"stdin write failed: {e}", "content": ""}

        try:
            return await asyncio.wait_for(
                self._read_until_result(), timeout=timeout
            )
        except asyncio.TimeoutError:
            return {"error": "timeout waiting for result", "content": ""}

    async def _read_until_result(self) -> dict:
        """Read NDJSON lines from stdout until a result message arrives."""
        result_content = ""

        while True:
            raw_line = await self._proc.stdout.readline()
            if not raw_line:
                # Process exited or stdout closed
                stderr = ""
                if self._proc.stderr:
                    try:
                        stderr_bytes = await asyncio.wait_for(
                            self._proc.stderr.read(4096), timeout=2
                        )
                        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
                    except (asyncio.TimeoutError, Exception):
                        pass
                return {
                    "error": f"process exited unexpectedly: {stderr or 'no stderr'}",
                    "content": result_content,
                }

            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.debug(f"[stream] non-JSON line: {line[:200]}")
                continue

            msg_type = data.get("type")

            if msg_type == "system":
                # First message only — capture CLI-assigned session ID
                self._cli_session_id = data.get("session_id")
                logger.debug(f"[stream] system init: session_id={self._cli_session_id}")

            elif msg_type == "assistant":
                # Extract text content from the assistant message
                content_blocks = data.get("message", {}).get("content", [])
                for block in content_blocks:
                    if block.get("type") == "text":
                        result_content = block.get("text", "")

            elif msg_type == "result":
                # Terminal message — response complete
                if data.get("is_error"):
                    return {"error": data.get("result", "unknown error"), "content": ""}
                final = data.get("result", result_content)
                return {"content": final, "raw": data}

            elif msg_type == "control_request":
                # Permission request — auto-deny (tools disabled, shouldn't happen)
                req_id = data.get("request_id", "")
                logger.warning(f"[stream] unexpected control_request: {req_id}")
                deny = {
                    "type": "control_response",
                    "response": {
                        "subtype": "success",
                        "request_id": req_id,
                        "response": {"behavior": "deny"},
                    },
                }
                try:
                    self._proc.stdin.write(
                        (json.dumps(deny) + "\n").encode("utf-8")
                    )
                    await self._proc.stdin.drain()
                except Exception:
                    pass

            # Ignore stream_event and other message types silently

    async def close(self) -> None:
        """Kill the streaming subprocess."""
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()
            except Exception:
                pass
        logger.debug(
            f"[stream] closed session {self.session_id[:8]} "
            f"(total_messages={self.total_messages})"
        )


# Session store: session_id -> StreamingSession
_sessions: dict[str, StreamingSession] = {}
_cleanup_task: asyncio.Task | None = None


async def _cleanup_idle_sessions() -> None:
    """Background task: kill sessions idle longer than SESSION_IDLE_TIMEOUT."""
    while True:
        await asyncio.sleep(60)
        now = time.monotonic()
        to_remove = [
            sid for sid, sess in _sessions.items()
            if now - sess.last_used > SESSION_IDLE_TIMEOUT
        ]
        for sid in to_remove:
            sess = _sessions.pop(sid, None)
            if sess:
                await sess.close()
                logger.info(f"[stream] cleaned up idle session {sid[:8]}")


# ============================================================
# Subprocess invocation (stateless /complete — unchanged)
# ============================================================

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


# ============================================================
# FastAPI lifecycle
# ============================================================

@app.on_event("startup")
async def startup() -> None:
    global _semaphore, _cleanup_task
    _semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    _cleanup_task = asyncio.create_task(_cleanup_idle_sessions())
    logger.info(
        f"Worker started: max_concurrent={MAX_CONCURRENT}, "
        f"session_idle_timeout={SESSION_IDLE_TIMEOUT}s, "
        f"config_dir={CLAUDE_CONFIG_DIR or 'default'}"
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    # Kill all streaming sessions
    for sess in _sessions.values():
        await sess.close()
    _sessions.clear()
    if _cleanup_task:
        _cleanup_task.cancel()


# ============================================================
# Request models
# ============================================================

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


class CloseRequest(BaseModel):
    session_id: str


# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
async def health() -> dict:
    streaming_info = {
        sid[:8]: {
            "alive": sess.is_alive,
            "messages": sess.total_messages,
            "idle_s": round(time.monotonic() - sess.last_used, 1),
        }
        for sid, sess in _sessions.items()
    }
    return {
        "status": "ok",
        "active": _active,
        "max": MAX_CONCURRENT,
        "streaming_sessions": len(_sessions),
        "sessions": streaming_info,
        "config_dir": CLAUDE_CONFIG_DIR or "default",
    }


@app.post("/complete")
async def complete(req: CompleteRequest) -> dict:
    """Stateless single-turn completion (subprocess.run — one process per call)."""
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
    """Start a streaming session — spawns a persistent subprocess.

    The subprocess stays alive for subsequent /resume_session calls,
    eliminating the ~3-4s respawn overhead per resume.
    """
    # Close existing session with same ID if any
    old = _sessions.pop(req.session_id, None)
    if old:
        await old.close()

    sess = StreamingSession(req.session_id, req.model, req.system_prompt)
    _sessions[req.session_id] = sess

    result = await sess.start_and_send(req.prompt, req.timeout)
    if result.get("error"):
        # Clean up failed session
        _sessions.pop(req.session_id, None)
        await sess.close()
        raise HTTPException(status_code=500, detail=result["error"])
    return {"content": result["content"]}


@app.post("/resume_session")
async def resume_session(req: ResumeRequest) -> dict:
    """Resume an existing streaming session — no subprocess respawn.

    The message is written directly to the persistent subprocess's stdin.
    This saves ~3-4s per call vs spawning a new claude process.
    """
    sess = _sessions.get(req.session_id)
    if not sess:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown session: {req.session_id} — call /start_session first",
        )
    if not sess.is_alive:
        _sessions.pop(req.session_id, None)
        raise HTTPException(
            status_code=500,
            detail=f"Streaming process for session {req.session_id[:8]} died",
        )

    result = await sess.resume(req.prompt, req.timeout)
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    return {"content": result["content"]}


@app.post("/close_session")
async def close_session(req: CloseRequest) -> dict:
    """Explicitly close a streaming session and kill its subprocess."""
    sess = _sessions.pop(req.session_id, None)
    if sess:
        await sess.close()
        return {"status": "closed", "total_messages": sess.total_messages}
    return {"status": "not_found"}
