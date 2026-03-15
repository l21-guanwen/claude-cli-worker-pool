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
WARM_POOL_MODEL = os.environ.get("WARM_POOL_MODEL", "claude-sonnet-4-6")

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

    async def start_and_send(
        self,
        prompt: str,
        timeout: float,
        warm_proc: asyncio.subprocess.Process | None = None,
    ) -> dict:
        """Start a streaming session and send the first user message.

        If warm_proc is provided (from WarmProcessPool), reuses it (~0s).
        Otherwise spawns a new subprocess on-demand (~3-4s).
        """
        async with self._lock:
            t0 = time.monotonic()

            if warm_proc and warm_proc.returncode is None:
                # Use pre-spawned warm process — skip the ~3-4s spawn
                self._proc = warm_proc
                logger.info(
                    f"[stream] using warm process pid={warm_proc.pid} "
                    f"for session {self.session_id[:8]} model={self._model}"
                )
            else:
                # Spawn on-demand (original behavior)
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
                self._proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    limit=1024 * 1024,  # 1MB — default 64KB too small for large LLM responses
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
            source = "warm" if warm_proc else "spawned"
            logger.info(
                f"[stream] session {self.session_id[:8]} started in {elapsed:.1f}s "
                f"({source}, cli_session={self._cli_session_id})"
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
                try:
                    self._proc.kill()
                except Exception:
                    pass
            except Exception:
                pass
        logger.debug(
            f"[stream] closed session {self.session_id[:8]} "
            f"(total_messages={self.total_messages})"
        )


# Session store: session_id -> StreamingSession
_sessions: dict[str, StreamingSession] = {}
_sessions_lock = asyncio.Lock()
_streaming_count = 0
_cleanup_task: asyncio.Task | None = None
_warm_pool: "WarmProcessPool | None" = None


# ============================================================
# Warm Process Pool
# ============================================================

class WarmProcessPool:
    """Pre-spawned Claude CLI streaming processes for instant session starts.

    Spawns WARM_POOL_SIZE processes on worker startup. Each sits idle waiting
    for stdin input. When /start_session arrives, a pre-spawned process is
    grabbed (~0s) instead of spawning on-demand (~3-4s). A background task
    eagerly replenishes the pool after each acquisition.

    Only serves requests matching the pool's model. Mismatched models
    fall back to on-demand spawning.
    """

    def __init__(self, model: str, pool_size: int) -> None:
        self._model = model
        self._pool_size = pool_size
        self._ready: asyncio.Queue[asyncio.subprocess.Process] = asyncio.Queue()
        self._replenish_task: asyncio.Task | None = None
        self._total_spawned = 0
        self._total_acquired = 0

    async def start(self) -> None:
        """Pre-spawn processes on worker startup."""
        if self._pool_size <= 0:
            logger.info("[warm] warm pool disabled (size=0)")
            return

        logger.info(f"[warm] pre-spawning {self._pool_size} processes (model={self._model})")
        t0 = time.monotonic()

        tasks = [self._spawn() for _ in range(self._pool_size)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, asyncio.subprocess.Process) and result.returncode is None:
                await self._ready.put(result)
            else:
                logger.warning(f"[warm] pre-spawn failed: {result}")

        elapsed = time.monotonic() - t0
        logger.info(
            f"[warm] {self._ready.qsize()}/{self._pool_size} processes ready in {elapsed:.1f}s"
        )

        self._replenish_task = asyncio.create_task(self._replenish_loop())

    async def _spawn(self) -> asyncio.subprocess.Process:
        """Spawn a single warm Claude CLI streaming process."""
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

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            limit=1024 * 1024,  # 1MB — default 64KB too small for large LLM responses
        )
        self._total_spawned += 1
        logger.debug(f"[warm] spawned pid={proc.pid} (total={self._total_spawned})")
        return proc

    async def acquire(self, model: str) -> asyncio.subprocess.Process | None:
        """Get a pre-spawned process if model matches, else None (spawn on-demand)."""
        if model != self._model:
            logger.debug(f"[warm] model mismatch ({model} != {self._model}), skip")
            return None

        while not self._ready.empty():
            try:
                proc = self._ready.get_nowait()
            except asyncio.QueueEmpty:
                break
            if proc.returncode is None:
                self._total_acquired += 1
                logger.info(
                    f"[warm] acquired pid={proc.pid} "
                    f"(remaining={self._ready.qsize()}, acquired={self._total_acquired})"
                )
                return proc
            # Reap zombie process
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await proc.wait()
            except Exception:
                pass
            logger.debug(f"[warm] reaped dead pid={proc.pid}")

        logger.debug("[warm] pool empty, spawn on-demand")
        return None

    async def _replenish_loop(self) -> None:
        """Background task: keep pool at target size, discard dead processes."""
        while True:
            try:
                await asyncio.sleep(3)

                # Drain queue, keep alive processes
                alive = []
                while not self._ready.empty():
                    try:
                        proc = self._ready.get_nowait()
                        if proc.returncode is None:
                            alive.append(proc)
                        else:
                            # Reap zombie process
                            try:
                                proc.kill()
                            except ProcessLookupError:
                                pass
                            try:
                                await proc.wait()
                            except Exception:
                                pass
                            logger.debug(f"[warm] reaped dead pid={proc.pid}")
                    except asyncio.QueueEmpty:
                        break
                for proc in alive:
                    await self._ready.put(proc)

                # Spawn replacements
                deficit = self._pool_size - self._ready.qsize()
                for _ in range(deficit):
                    try:
                        proc = await self._spawn()
                        if proc.returncode is None:
                            await self._ready.put(proc)
                    except Exception as e:
                        logger.warning(f"[warm] replenish failed: {e}")
                        break
            except asyncio.CancelledError:
                raise  # let cancellation propagate
            except Exception as e:
                logger.error(f"[warm] replenish loop error: {e}")
                await asyncio.sleep(5)  # back off before retrying

    async def close(self) -> None:
        """Kill all warm processes and stop replenishment."""
        if self._replenish_task:
            self._replenish_task.cancel()

        killed = 0
        while not self._ready.empty():
            try:
                proc = self._ready.get_nowait()
                if proc.returncode is None:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=3)
                    except asyncio.TimeoutError:
                        proc.kill()
                    killed += 1
            except asyncio.QueueEmpty:
                break

        logger.info(
            f"[warm] closed: killed={killed}, "
            f"total_spawned={self._total_spawned}, total_acquired={self._total_acquired}"
        )

    @property
    def available(self) -> int:
        return self._ready.qsize()


async def _cleanup_idle_sessions() -> None:
    """Background task: kill sessions idle longer than SESSION_IDLE_TIMEOUT."""
    global _streaming_count
    while True:
        await asyncio.sleep(60)
        now = time.monotonic()
        async with _sessions_lock:
            to_remove = [
                sid for sid, sess in _sessions.items()
                if now - sess.last_used > SESSION_IDLE_TIMEOUT
            ]
            removed = {sid: _sessions.pop(sid) for sid in to_remove if sid in _sessions}
        for sid, sess in removed.items():
            _streaming_count = max(0, _streaming_count - 1)
            await sess.close()
            logger.info(f"[stream] cleaned up idle session {sid[:8]}")


# ============================================================
# Error classification
# ============================================================

# Patterns that indicate the account credential is invalid/expired.
# Claude CLI embeds these in the is_error result string or stderr.
_AUTH_ERROR_PATTERNS = (
    "invalid api key",
    "api key expired",
    "authentication",
    "unauthorized",
    "not authenticated",
    "credential",
    "login required",
    "session expired",
    "token expired",
    "account suspended",
    "account disabled",
    "permission denied",
)


def _is_auth_error(error_msg: str) -> bool:
    """Check if an error message indicates an account credential problem."""
    lower = error_msg.lower()
    return any(pat in lower for pat in _AUTH_ERROR_PATTERNS)


# ============================================================
# Subprocess invocation (stateless /complete)
# ============================================================

async def _invoke_warm(prompt: str, model: str, timeout: float) -> dict | None:
    """Try to run a one-shot completion using a warm streaming process.

    Returns the result dict, or None if warm pool is unavailable (model
    mismatch or pool empty) — caller should fall back to subprocess.run.
    The warm process is killed after the single response.
    """
    if not _warm_pool:
        return None
    proc = await _warm_pool.acquire(model)
    if proc is None:
        return None

    # Use a temporary StreamingSession for the one-shot call
    sess = StreamingSession("_complete_oneshot", model)
    try:
        result = await sess.start_and_send(prompt, timeout, warm_proc=proc)
        return result
    finally:
        await sess.close()


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
    global _semaphore, _cleanup_task, _warm_pool
    _semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    _cleanup_task = asyncio.create_task(_cleanup_idle_sessions())

    _warm_pool = WarmProcessPool(WARM_POOL_MODEL, MAX_CONCURRENT)
    await _warm_pool.start()

    logger.info(
        f"Worker started: max_concurrent={MAX_CONCURRENT}, "
        f"session_idle_timeout={SESSION_IDLE_TIMEOUT}s, "
        f"warm_pool={MAX_CONCURRENT} ({WARM_POOL_MODEL}), "
        f"config_dir={CLAUDE_CONFIG_DIR or 'default'}"
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    global _streaming_count
    # Kill all streaming sessions
    async with _sessions_lock:
        all_sessions = list(_sessions.values())
        _sessions.clear()
        _streaming_count = 0
    for sess in all_sessions:
        await sess.close()
    if _warm_pool:
        await _warm_pool.close()
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
    # Snapshot under lock to avoid RuntimeError from concurrent dict mutation
    async with _sessions_lock:
        sessions_snapshot = dict(_sessions)
    streaming_info = {
        sid[:8]: {
            "alive": sess.is_alive,
            "messages": sess.total_messages,
            "idle_s": round(time.monotonic() - sess.last_used, 1),
        }
        for sid, sess in sessions_snapshot.items()
    }
    return {
        "status": "ok",
        "active": _active,
        "max": MAX_CONCURRENT,
        "streaming_sessions": len(_sessions),
        "streaming_count": _streaming_count,
        "sessions": streaming_info,
        "warm_pool": {
            "available": _warm_pool.available if _warm_pool else 0,
            "target": MAX_CONCURRENT,
            "model": WARM_POOL_MODEL,
            "total_spawned": _warm_pool._total_spawned if _warm_pool else 0,
            "total_acquired": _warm_pool._total_acquired if _warm_pool else 0,
        },
        "config_dir": CLAUDE_CONFIG_DIR or "default",
    }


@app.post("/complete")
async def complete(req: CompleteRequest) -> dict:
    """Stateless single-turn completion.

    Tries warm pool first (~0s spawn). Falls back to subprocess.run (~3-4s)
    if warm pool is empty or model doesn't match.
    Both paths share a single semaphore slot for accurate active count.
    """
    global _active
    prompt = req.prompt
    if req.system_prompt:
        prompt = f"<system-context>\n{req.system_prompt}\n</system-context>\n\n{req.prompt}"

    async with _semaphore:
        _active += 1
        try:
            # Try warm pool path (streaming process, one-shot)
            result = await _invoke_warm(prompt, req.model, req.timeout)
            if result is not None:
                if result.get("error"):
                    status = 401 if _is_auth_error(result["error"]) else 500
                    raise HTTPException(status_code=status, detail=result["error"])
                return {"content": result["content"]}

            # Fallback: subprocess.run (cold start) — run in thread pool
            cmd = [
                CLAUDE_BIN, "-p",
                "--output-format", "json",
                "--max-turns", "1",
                "--model", req.model,
                "--no-session-persistence",
                "--tools", "",
            ]
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, _run_claude, cmd, prompt, req.timeout)
            if result.get("error"):
                status = 401 if _is_auth_error(result["error"]) else 500
                raise HTTPException(status_code=status, detail=result["error"])
            return {"content": result["content"]}
        finally:
            _active -= 1


@app.post("/start_session")
async def start_session(req: SessionRequest) -> dict:
    """Start a streaming session — spawns a persistent subprocess.

    The subprocess stays alive for subsequent /resume_session calls,
    eliminating the ~3-4s respawn overhead per resume.
    Capped at MAX_CONCURRENT active streaming sessions per worker.
    """
    global _streaming_count

    # Atomic check-and-increment under lock to prevent TOCTOU race
    async with _sessions_lock:
        if _streaming_count >= MAX_CONCURRENT:
            raise HTTPException(
                status_code=503,
                detail=f"At session capacity ({MAX_CONCURRENT})",
            )
        old = _sessions.pop(req.session_id, None)
        if old:
            _streaming_count = max(0, _streaming_count - 1)
        _streaming_count += 1
        sess = StreamingSession(req.session_id, req.model, req.system_prompt)
        _sessions[req.session_id] = sess

    # Close old session outside lock to avoid holding it during I/O
    if old:
        await old.close()

    # Try warm pool first — saves ~3-4s subprocess spawn
    warm_proc = await _warm_pool.acquire(req.model) if _warm_pool else None
    result = await sess.start_and_send(req.prompt, req.timeout, warm_proc=warm_proc)
    if result.get("error"):
        # Clean up failed session
        async with _sessions_lock:
            _sessions.pop(req.session_id, None)
        _streaming_count = max(0, _streaming_count - 1)
        await sess.close()
        status = 401 if _is_auth_error(result["error"]) else 500
        raise HTTPException(status_code=status, detail=result["error"])
    return {"content": result["content"]}


@app.post("/resume_session")
async def resume_session(req: ResumeRequest) -> dict:
    """Resume an existing streaming session — no subprocess respawn.

    The message is written directly to the persistent subprocess's stdin.
    This saves ~3-4s per call vs spawning a new claude process.
    """
    global _streaming_count

    async with _sessions_lock:
        sess = _sessions.get(req.session_id)
    if not sess:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown session: {req.session_id} — call /start_session first",
        )
    if not sess.is_alive:
        async with _sessions_lock:
            _sessions.pop(req.session_id, None)
        _streaming_count = max(0, _streaming_count - 1)
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
    global _streaming_count

    async with _sessions_lock:
        sess = _sessions.pop(req.session_id, None)
    if sess:
        _streaming_count = max(0, _streaming_count - 1)
        await sess.close()
        return {"status": "closed", "total_messages": sess.total_messages}
    return {"status": "not_found"}
