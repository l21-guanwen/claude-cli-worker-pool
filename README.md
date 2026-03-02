# Claude CLI Worker Pool

A distributed worker pool for Claude CLI that bypasses the 5-hour per-session token limit
by running multiple Claude CLI processes across Docker containers, each backed by an independent
Claude account.

**Single external endpoint** — your app calls one URL. The pool service handles load balancing,
session affinity, and worker health monitoring internally.

---

## Problem

Claude CLI enforces a **5-hour token/message limit per subscription session**. Running concurrent
evaluations with a single CLI instance exhausts this limit mid-run.

This POC distributes work across multiple worker processes. Each Claude account gets its own
independent 5-hour limit. Multiple workers can share one account to increase concurrency within
that account's budget.

---

## Architecture

### Account × Workers Model

```
total_workers = num_accounts × workers_per_account
```

| Dimension | What it buys you |
|-----------|-----------------|
| More accounts | More independent 5-hour token budgets |
| More workers per account | More concurrency within the same budget |

```
account-1 (~/.claude-acct1)          account-2 (~/.claude-acct2)
  ├── worker process :8000              ├── worker process :8000
  ├── worker process :8001              ├── worker process :8001
  └── worker process :8002              └── worker process :8002
          └─────────── 6 worker URLs (flat list) ────────────┘
                                 │
                      Pool Service :8090
                  ┌───────────────┴───────────────┐
                  │  /complete      → round-robin  │
                  │  /start_session → EWMA pick    │
                  │  /resume_session → pinned      │
                  │  /close_session  → pinned      │
                  └────────────────────────────────┘
                                 │
                              Client
```

**Key properties:**

- **Single external endpoint** — clients call `:8090` only; workers have no published ports.
- **Hybrid execution** — `/complete` uses one-shot `subprocess.run`; sessions use persistent
  streaming subprocesses (`--input-format stream-json`) that stay alive between messages.
- **Warm process pool** — pre-spawns streaming processes on startup so `/start_session` grabs
  one instantly (~0s) instead of spawning on-demand (~3-4s). Eagerly replenishes after each use.
- **Round-robin for stateless calls** — `/complete` cycles worker-0 → worker-1 → … → worker-N
  → worker-0, ensuring even distribution across all accounts.
- **EWMA routing for new sessions** — `/start_session` picks the fastest, least-loaded worker
  using `score = (active_calls + 1) × avg_response_ms`.
- **Session affinity** — `session_id → worker_index` map ensures `/resume_session` always returns
  to the same worker process (streaming subprocess lives on that worker).
- **Shared `CLAUDE_CONFIG_DIR`** — workers within the same account container share one auth dir.
  Sessions are stored as UUID-named files, so concurrent workers never conflict.

### Why a custom load balancer?

Standard load balancers (nginx, HAProxy, Traefik, Envoy) can't route on a JSON body field like
`session_id` — they treat the HTTP body as opaque. Session affinity via cookie or IP doesn't work
here because:

1. Each `resume_session` must hit the **same worker process** that ran `start_session` (Claude CLI
   session state is local to that process).
2. The affinity key (`session_id`) lives inside the JSON body, not in headers or URL.

The pool service implements this in ~15 lines: a `dict[session_id → worker_index]` with an
`asyncio.Lock`. The EWMA math is 3 lines. No frameworks needed — just `aiohttp`.

---

## Prerequisites

- **Docker + Docker Compose** — Docker Desktop on Windows/macOS; Docker Engine on Linux
- **Node.js + npm** — required on the host to install and authenticate Claude CLI
- **1–N Claude accounts** — each authenticated before starting the pool

### Install Claude CLI (host only — for `claude login`)

The worker containers install Claude CLI automatically. You only need it on the host to run
`claude login` and create the config directories that get volume-mounted into the containers.

**Linux / macOS:**
```bash
npm install -g @anthropic-ai/claude-code
```

**Windows (PowerShell or cmd):**
```powershell
npm install -g @anthropic-ai/claude-code
```

Verify:
```bash
claude --version
```

---

## Quick Start

### Linux / macOS

#### 1. Authenticate accounts

Each account needs its own config directory authenticated before Docker starts:

```bash
mkdir -p ~/.claude-acct1 ~/.claude-acct2

CLAUDE_CONFIG_DIR=~/.claude-acct1 claude login
CLAUDE_CONFIG_DIR=~/.claude-acct2 claude login
```

For single-account testing (no independent token budgets, but verifies the pool works):

```bash
# Both account containers point at the same auth dir
CLAUDE_CONFIG_DIR_1=~/.claude CLAUDE_CONFIG_DIR_2=~/.claude docker compose up -d
```

#### 2. Start the pool

```bash
cd docker/

CLAUDE_CONFIG_DIR_1=~/.claude-acct1 \
CLAUDE_CONFIG_DIR_2=~/.claude-acct2 \
WORKERS_PER_ACCOUNT=3 \
docker compose up -d
```

This starts: 2 account containers × 3 worker processes each = **6 workers total**.

#### 3. Verify

```bash
curl http://localhost:8090/health
```

---

### Windows

#### 1. Authenticate accounts

Open **PowerShell** and create two separate config directories:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.claude-acct1"
New-Item -ItemType Directory -Force "$env:USERPROFILE\.claude-acct2"
```

Authenticate each account (a browser window will open):

```powershell
$env:CLAUDE_CONFIG_DIR = "$env:USERPROFILE\.claude-acct1"
claude login

$env:CLAUDE_CONFIG_DIR = "$env:USERPROFILE\.claude-acct2"
claude login
```

Verify authentication works for each:

```powershell
$env:CLAUDE_CONFIG_DIR = "$env:USERPROFILE\.claude-acct1"
claude -p --max-turns 1 "say hi"
```

#### 2. Start the pool

```powershell
cd docker/

$env:CLAUDE_CONFIG_DIR_1 = "$env:USERPROFILE\.claude-acct1"
$env:CLAUDE_CONFIG_DIR_2 = "$env:USERPROFILE\.claude-acct2"
$env:WORKERS_PER_ACCOUNT = "3"
docker compose up -d
```

Or as a one-liner:

```powershell
$env:CLAUDE_CONFIG_DIR_1="$env:USERPROFILE\.claude-acct1"; $env:CLAUDE_CONFIG_DIR_2="$env:USERPROFILE\.claude-acct2"; $env:WORKERS_PER_ACCOUNT="3"; docker compose up -d
```

This starts: 2 account containers × 3 worker processes each = **6 workers total**.

> **Note:** Docker Desktop on Windows mounts Windows paths (e.g. `C:\Users\jenny\.claude-acct1`)
> directly — no WSL path conversion needed.

#### 3. Verify

```powershell
curl http://localhost:8090/health
```

Or in PowerShell:

```powershell
Invoke-RestMethod http://localhost:8090/health | ConvertTo-Json -Depth 5
```

---

### Expected health response

```json
{
  "status": "ok",
  "healthy": 6,
  "total": 6,
  "active_calls": 0,
  "workers": [
    {
      "url": "http://account-1:8000",
      "active_calls": 0,
      "total_requests": 0,
      "avg_response_ms": 5000.0,
      "score": 5000.0,
      "is_healthy": true
    },
    { "url": "http://account-1:8001", ... },
    { "url": "http://account-1:8002", ... },
    { "url": "http://account-2:8000", ... },
    { "url": "http://account-2:8001", ... },
    { "url": "http://account-2:8002", ... }
  ]
}
```

`total_requests` counts successful calls per worker since the pool started — useful for
verifying even distribution across accounts.

Workers are not reachable externally — `curl http://localhost:8000/health` should fail or
return an unrelated service (workers only listen on the internal Docker network).

### 4. Run tests

```bash
pip install pytest pytest-asyncio aiohttp
python -m pytest tests/test_pool.py -v
```

---

## HTTP API

All requests go to the pool service. Workers are never called directly.

### `GET /health`

Pool and per-worker status. Returns `total_requests` per worker — check this after a run to
verify both accounts received traffic (should be roughly equal).

### `POST /complete`

Stateless single-turn completion. Routes via **round-robin** across all workers — guarantees
even distribution across accounts over time.

```json
// Request
{"prompt": "What is 2+2?", "system_prompt": "You are a math tutor."}

// Request with per-request model override (faster, see Latency section)
{"prompt": "Score 1-10.", "model": "claude-haiku-4-5-20251001"}

// Response
{"content": "4"}
```

### `POST /start_session`

Start a named multi-turn session. Routes via **EWMA scoring** to the fastest available worker,
then pins the session to that worker.

Session IDs can be **any string** — the pool converts non-UUID strings to a deterministic UUID5
transparently (Claude CLI requires valid UUIDs for `--session-id`).

```json
// Request
{"session_id": "eval-001", "prompt": "Analyze AAPL.", "system_prompt": "You are an analyst."}

// Request with per-request model override
{"session_id": "eval-001", "prompt": "Analyze AAPL.", "model": "claude-haiku-4-5-20251001"}

// Response
{"content": "..."}
```

### `POST /resume_session`

Continue an existing session. Automatically routed to the **pinned worker** for that session ID.
The streaming subprocess is reused — **no Node.js respawn**, just a stdin write + API inference.
Model is carried forward from `start_session` — no override here.

```json
// Request
{"session_id": "eval-001", "prompt": "What is the fair value?"}

// Response
{"content": "Based on the DCF..."}
```

### `POST /close_session`

Explicitly close a streaming session and kill its subprocess. Optional — idle sessions are
automatically cleaned up after `SESSION_IDLE_TIMEOUT` (default: 10 minutes).

```json
// Request
{"session_id": "eval-001"}

// Response
{"status": "closed", "total_messages": 5}
```

---

## Python Usage

```python
import asyncio
import aiohttp

POOL = "http://localhost:8090"

async def run_eval():
    async with aiohttp.ClientSession() as s:
        # Stateless (no session) — round-robin across all workers
        async with s.post(f"{POOL}/complete", json={"prompt": "Say hi."}) as r:
            print((await r.json())["content"])

        # Multi-turn session — pinned to one worker
        session_id = "my-eval-001"
        await s.post(f"{POOL}/start_session", json={
            "session_id": session_id,
            "system_prompt": "You are a judge scoring AI responses.",
            "prompt": "Here is response A: [...]",
        })
        async with s.post(f"{POOL}/resume_session", json={
            "session_id": session_id,
            "prompt": "Score this response 1-10.",
        }) as r:
            print((await r.json())["content"])

asyncio.run(run_eval())
```

---

## Capacity Planning

```
total_workers = num_accounts × workers_per_account
```

| Accounts | Workers/Account | Total Workers | ~Concurrent Questions | Token Budgets |
|:--------:|:--------------:|:-------------:|:---------------------:|:-------------:|
| 1 | 1 | 1 | 1 | 1× |
| 1 | 3 | 3 | 1–2 | 1× (shared) |
| 2 | 3 | 6 | 2–3 | 2× |
| 3 | 5 | 15 | 5–7 | 3× |
| 5 | 5 | 25 | 10–12 | 5× |

**Session mode** (multi-turn): each eval question uses ~3–5 sequential CLI calls on one session
(`start_session` + 2–4 `resume_session` turns). Set `WORKERS_PER_ACCOUNT` high enough that
workers are never all busy at once; set `num_accounts` based on how many independent token
budgets you need.

**Independent mode** (stateless): each checkpoint scores as a standalone `/complete` call;
checkpoints within a question run concurrently. Round-robin distributes them evenly — with
2 accounts and 67 calls you'll see ~33-34 per account regardless of question count.

---

## Routing Strategy

### `/complete` — Round-Robin

Stateless calls cycle through all workers in order (worker-0 → worker-1 → … → worker-N-1 → worker-0).
This guarantees that over any batch of N calls, each worker sees exactly one call — and therefore
each account sees exactly 50% of traffic with 2 accounts, 33% with 3, etc.

EWMA scoring is not used here because it tends to cluster on low-index workers (all workers start
with the same initial score of 5000ms, so the first worker always wins ties), starving
higher-indexed workers — and their accounts — of traffic.

### `/start_session` — EWMA Scoring

New sessions pick the worker with the lowest score:

```
score = (active_calls + 1) × avg_response_ms
```

- Lower score = preferred.
- Unhealthy workers get `score = ∞` and are never picked.
- EWMA (α = 0.2) updates on successful calls only — errors don't artificially lower the score.

```
State                    acct-1:8000  acct-1:8001  acct-1:8002  acct-2:*
Cold start (no data)       5000         5000         5000         5000   → picks worker 0
W0 slow (3s avg)           3000          800          800          800   → picks 1, 2, or acct-2
W1 busy (4 active)          800         4000          800          800   → picks 0, 2, or acct-2
W2 unhealthy                800          800           ∞           800   → never picks W2
```

### `/resume_session` — Pinned

Always routes to the worker that handled `start_session` for that session ID. The streaming
subprocess lives on that worker, so the resume must hit the exact same process.

### `/close_session` — Pinned

Routes to the pinned worker and kills the streaming subprocess. Removes the session from the
affinity map. Optional — idle sessions are auto-cleaned after `SESSION_IDLE_TIMEOUT`.

---

## Latency

### Two optimizations

**1. Streaming mode** — sessions use `--input-format stream-json --output-format stream-json`,
keeping the Node.js subprocess alive between messages. Eliminates ~3-4s subprocess spawn on
every `/resume_session` call.

**2. Warm process pool** — pre-spawns streaming processes at worker startup. When `/start_session`
arrives, it grabs one instantly (~0s) instead of spawning on-demand (~3-4s). Eagerly replenishes
after each acquisition. Falls back to on-demand if the pool is empty.

```
/complete:        subprocess.run per call  → spawns, responds, exits  (~5-7s)
/start_session:   grab warm process (~0s)  → responds, stays alive    (~1-3s with warm pool)
/start_session:   pool empty, spawn        → responds, stays alive    (~5-7s, on-demand fallback)
/resume_session:  write to existing stdin  → responds immediately     (~1-8s, no respawn)
```

### Benchmark: 18-question evaluation

| Metric | Old Session | Streaming (cold) | Streaming + Warm Pool | Independent |
|--------|:-:|:-:|:-:|:-:|
| **Wall-clock** | 4m 34s | 3m 10s | **3m 5s** | 3m 24s |
| Per-question avg | 50.5s | 30.0s | **30.8s** | 35.6s |
| **Start session avg** | ~15.1s | 6.7s | **3.1s (-53%)** | n/a |
| Resume session avg | ~10.3s | 6.3s | 7.4s | n/a |

Warm pool breakdown (18 start_session calls): 12 warm hits at 2.0s avg, 6 cold misses at 5.4s
avg (pool exhausted under burst). Overall start avg: 3.1s — 53% faster than cold streaming,
79% faster than old session.

**When to use each:**
- **Streaming + warm pool** (default): Best for most workloads — lowest start + resume latency
- **Streaming (cold)**: When `WARM_POOL_SIZE=0` or model varies per request
- **Independent**: Best for very large pools (10+ workers) where parallelism dominates
- **Old session**: Deprecated — streaming is strictly better

### Per-request model override

Pass `"model"` in any `/complete` or `/start_session` request to override the pool default:

```bash
# Default model (~5-7s)
curl -X POST http://localhost:8090/complete \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Score this 1-10: [response]"}'

# Haiku override (~3-5s)
curl -X POST http://localhost:8090/complete \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Score this 1-10: [response]", "model": "claude-haiku-4-5-20251001"}'
```

If `model` is empty or omitted, the pool uses the `MODEL` env var (default: `claude-sonnet-4-6`).

---

## Configuration Reference

**Pool service** (`docker-compose.yml` → `pool-service`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ACCOUNT_BASE_URLS` | required | Comma-separated account container base URLs (no port), e.g. `http://account-1,http://account-2` |
| `WORKERS_PER_ACCOUNT` | `1` | Worker processes per account container |
| `BASE_PORT` | `8000` | Starting port; workers listen on `BASE_PORT`, `BASE_PORT+1`, … |
| `MODEL` | `claude-sonnet-4-6` | Default Claude model (overridable per-request via `"model"` field) |
| `TIMEOUT` | `300` | Per-request timeout (seconds) |
| `MAX_TOKENS` | `8192` | Max tokens per response |
| `WORKER_URLS` | — | Override: flat comma-separated list of worker URLs (skips `ACCOUNT_BASE_URLS` expansion) |

**Account containers** (`docker-compose.yml` → `account-1`, `account-2`, …):

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_CONFIG_DIR_N` | `~/.claude` | Host path for account N's auth config (set via env var in docker compose command) |
| `WORKERS_PER_ACCOUNT` | `1` | Must match the value set in pool-service |
| `BASE_PORT` | `8000` | Must match the value set in pool-service |
| `MAX_CONCURRENT` | `1` | Semaphore per worker process (usually leave at 1) |
| `SESSION_IDLE_TIMEOUT` | `600` | Kill idle streaming sessions after N seconds (default: 10 min) |
| `WARM_POOL_SIZE` | `2` | Pre-spawned streaming processes per worker (0 to disable) |
| `WARM_POOL_MODEL` | `claude-sonnet-4-6` | Model for warm processes (must match request model) |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Empty `content` in responses | `--tools ""` not passed to `claude -p` | Already handled in `worker_api.py` — do not remove |
| `Unknown session` 404 error | `resume_session` called before `start_session` | Always call `start_session` first |
| `Streaming process died` 500 | Streaming subprocess crashed between calls | Will need a new `start_session`; check worker logs for cause |
| `Session ID already in use` 500 | Same session ID reused across test runs | Use unique session IDs per run (e.g. add UUID suffix) |
| Workers show `healthy: 0` | Workers still starting up | Wait 5–10s after `docker compose up`, then retry `/health` |
| Worker score stuck at 5000 | Cold start — no successful calls yet | Normal; self-corrects after the first successful call |
| All workers unhealthy | Docker network issue or workers crashed | `docker compose logs account-1` |
| Auth fails in worker | `CLAUDE_CONFIG_DIR` not authenticated | Run `CLAUDE_CONFIG_DIR=~/.claude-acctN claude login` on the host |
| `ACCOUNT_BASE_URLS` not set | Missing env var | Set it in `docker-compose.yml` or via shell env |
| Wrong worker count in health | `WORKERS_PER_ACCOUNT` mismatch | Ensure pool-service and account containers use the same value |
| One account gets all traffic | Old pool version without round-robin | Rebuild pool-service: `docker compose build pool-service && docker compose up --no-deps -d pool-service` |
| `total_requests` all 0 after a run | Pool service not rebuilt after update | See above — rebuild needed to pick up `pool.py` changes |
| **Windows:** `claude login` opens wrong browser | Multiple browsers installed | Claude CLI uses the default browser — set it in Windows settings |
| **Windows:** Volume mount fails | Path contains spaces or backslashes | Use `"$env:USERPROFILE\.claude-acct1"` — PowerShell quotes handle it correctly |
| **Windows:** `docker compose` not found | Docker Desktop CLI plugin not installed | Ensure Docker Desktop is up to date; use `docker compose` (not `docker-compose`) |
| **Windows:** Workers start but `/health` returns 0 healthy | Docker Desktop using WSL 2 backend, volume not accessible | Check Docker Desktop → Settings → Resources → WSL Integration is enabled for your distro |
| **Windows:** `CRLF` script error in container | Shell script has Windows line endings | Already handled in the worker Dockerfile with `sed -i 's/\r//'` — rebuild if you edited `start_workers.sh` on Windows |

---

## File Structure

```
claude-cli-worker-pool/
├── docker/
│   ├── claude_worker/
│   │   ├── Dockerfile           Worker image (Node.js + Claude CLI + Python FastAPI)
│   │   ├── start_workers.sh     Spawns WORKERS_PER_ACCOUNT uvicorn processes on consecutive ports
│   │   ├── worker_api.py        FastAPI wrapper: one-shot subprocess + streaming sessions + warm pool
│   │   └── requirements.txt
│   ├── pool_service/
│   │   ├── Dockerfile           Uses repo-root build context to include claude_cli_pool/
│   │   ├── pool_service.py      FastAPI load balancer — single endpoint, routing, session affinity
│   │   └── requirements.txt
│   └── docker-compose.yml       pool-service (external :8090) + account containers (internal)
├── claude_cli_pool/
│   ├── __init__.py
│   └── pool.py                  WorkerClient (EWMA + request counter) + ClaudeCLIPool (round-robin
│                                  for /complete, EWMA for /start_session, pinned for /resume + /close)
├── tests/
│   └── test_pool.py             Integration tests (5 tests, require docker compose up)
├── requirements.txt             aiohttp>=3.9.0
└── README.md
```
