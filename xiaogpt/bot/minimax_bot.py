from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, ClassVar

import httpx

from xiaogpt.bot.base_bot import BaseBot, ChatHistoryMixin

if TYPE_CHECKING:
    import openai


@dataclasses.dataclass
class MiniMaxBot(ChatHistoryMixin, BaseBot):
    name: ClassVar[str] = "MiniMax"
    default_options: ClassVar[dict[str, str]] = {"model": "MiniMax-M2.1"}
    minimax_key: str
    proxy: str | None = None
    history: list[tuple[str, str]] = dataclasses.field(default_factory=list, init=False)

    @classmethod
    def from_config(cls, config):
        return cls(
            minimax_key=config.minimax_key,
            proxy=config.proxy,
        )

    async def ask(self, question: str, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.minimax_key}",
            "Content-Type": "application/json",
        }
        
        # Build messages with history
        messages = []
        for q, a in self.history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": question})
        
        payload = {
            "model": kwargs.get("model", "MiniMax-M2.1"),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "https://api.minimax.chat/v1/text/chatcompletion_v2",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            
            # Save to history
            self.history.append((question, answer))
            
            return answer

    async def stream_ask(self, question: str, **kwargs):
        # For streaming - return chunks
        answer = await self.ask(question, **kwargs)
        yield answer
