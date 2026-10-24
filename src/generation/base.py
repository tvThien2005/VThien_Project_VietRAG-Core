from abc import ABC, abstractmethod
from typing import Iterator, Optional


class BaseLLM(ABC):
    """Interface cơ sở cho các LLM Provider (Groq, Ollama)."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Sinh câu trả lời từ prompt và system prompt."""
        pass

    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Sinh câu trả lời dạng luồng streaming từng token."""
        pass

