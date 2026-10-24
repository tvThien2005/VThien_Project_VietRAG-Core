from typing import Iterator, Optional
import groq

from src.config import get_config
from src.generation.base import BaseLLM
from src.logger import get_logger
from src.utils.rate_limit import call_with_rate_limit_retry

logger = get_logger("groq_llm")


class GroqLLM(BaseLLM):
    """Wrapper kết nối tới Groq Cloud API phục vụ sinh câu trả lời với tốc độ cao."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,  
    ):
        cfg = get_config()
        self.model_name = model_name or cfg.llm.groq.model_name
        self.api_key = api_key or cfg.llm.groq.api_key
        self.temperature = temperature if temperature is not None else cfg.llm.groq.temperature
        self.max_tokens = max_tokens or cfg.llm.groq.max_tokens
        self.timeout = timeout or cfg.llm.groq.timeout

        if not self.api_key:
            raise ValueError(
                "Chưa tìm thấy GROQ_API_KEY. Vui lòng thiết lập biến môi trường GROQ_API_KEY trong file .env."
            )

        logger.info(f"Khởi tạo Groq LLM với mô hình: {self.model_name}")
        self.client = groq.Groq(
            api_key=self.api_key,
            timeout=float(self.timeout),
            max_retries=0,
        )

    def _prepare_messages(self, prompt: str, system_prompt: Optional[str] = None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Gửi prompt tới Groq và nhận toàn bộ câu trả lời."""
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens
        messages = self._prepare_messages(prompt, system_prompt)

        try:
            response = call_with_rate_limit_retry(
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                ),
                operation_name="Groq generation",
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Lỗi khi gọi Groq API ({self.model_name}): {e}")
            raise

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Gửi prompt tới Groq và stream từng token khi chúng được sinh ra."""
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens
        messages = self._prepare_messages(prompt, system_prompt)

        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                stream=True,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            logger.error(f"Lỗi khi streaming từ Groq API: {e}")
            raise

