"""Thin adapter: calls Claude CLI Pool Service HTTP API as a linvest21 BaseLLMProvider.

The pool service is the load balancer. This adapter makes HTTP calls to it.
Copy this file into the linvest21 core repo when ready to integrate:
  → core/sharedinfra/llm/impl/claude_cli_pool_adapter.py

Integration steps:
1. Copy this file into core/sharedinfra/llm/impl/
2. In scoring.py: add ClaudeCLIPoolAdapter to the isinstance check for judge_llm
3. In run_live_eval.py: add --pool-url arg, instantiate ClaudeCLIPoolAdapter when provided
4. Add aiohttp>=3.9.0 to pyproject.toml
"""
from __future__ import annotations
from datetime import datetime, timezone

import aiohttp

# These imports will resolve once this file is in the linvest21 core repo:
from linvest21.product.am.ainvestor.core.sharedinfra.llm.base import BaseLLMProvider
from linvest21.product.am.ainvestor.core.sharedinfra.llm.models import (
    LLMConfig,
    LLMResponse,
    Message,
    ToolDefinition,
)


class ClaudeCLIPoolAdapter(BaseLLMProvider):
    """HTTP client to the Claude CLI Pool Service (load balancer at :8080).

    Clients call this adapter the same way they call any other BaseLLMProvider.
    The pool service handles worker selection, session affinity, and EWMA routing.

    Args:
        pool_url: Pool service base URL, e.g. "http://localhost:8080"
        config: LLMConfig (used for timeout).
    """

    def __init__(self, pool_url: str, config: LLMConfig) -> None:
        self._url = pool_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=config.timeout + 10)

    async def _post(self, endpoint: str, payload: dict) -> LLMResponse:
        async with aiohttp.ClientSession(timeout=self._timeout) as s:
            async with s.post(f"{self._url}{endpoint}", json=payload) as r:
                if r.status != 200:
                    text = await r.text()
                    raise RuntimeError(
                        f"Pool service {endpoint} → {r.status}: {text}"
                    )
                data = await r.json()
                return LLMResponse(
                    content=data["content"],
                    created_at=datetime.now(timezone.utc),
                )

    async def complete(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
        **kwargs,
    ) -> LLMResponse:
        prompt = messages[-1].content if messages else ""
        return await self._post(
            "/complete",
            {"prompt": prompt, "system_prompt": system_prompt or ""},
        )

    async def start_session(
        self,
        session_id: str,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        return await self._post(
            "/start_session",
            {
                "session_id": session_id,
                "prompt": prompt,
                "system_prompt": system_prompt or "",
            },
        )

    async def resume_session(
        self,
        session_id: str,
        prompt: str,
        **kwargs,
    ) -> LLMResponse:
        return await self._post(
            "/resume_session",
            {"session_id": session_id, "prompt": prompt},
        )

    async def health_check(self) -> bool:
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as s:
                async with s.get(f"{self._url}/health") as r:
                    return r.status == 200
        except Exception:
            return False
