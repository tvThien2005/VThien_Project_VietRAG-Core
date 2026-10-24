import json
from typing import Iterator, Optional
import requests

from src.config import get_config
from src.generation.base import BaseLLM
from src.logger import get_logger

logger = get_logger("ollama_llm")


class OllamaLLM(BaseLLM):
    """Wrapper kết nối tới Ollama cục bộ qua REST API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ):
        cfg = get_config()
        self.base_url = (base_url or cfg.llm.ollama.base_url).rstrip("/")
        self.model_name = model_name or cfg.llm.ollama.model_name
        self.temperature = temperature if temperature is not None else cfg.llm.ollama.temperature
        self.max_tokens = max_tokens or cfg.llm.ollama.max_tokens
        self.timeout = timeout or cfg.llm.ollama.timeout

        logger.info(f"Khởi tạo Ollama LLM tại: {self.base_url} với model: {self.model_name}")

    def _prepare_payload(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            },
        }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Gửi prompt tới Ollama chat API và nhận kết quả."""
        payload = self._prepare_payload(
            prompt, system_prompt, temperature, max_tokens, stream=False
        )
        url = f"{self.base_url}/api/chat"

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Không thể kết nối tới Ollama tại {self.base_url}. "
                "Hãy kiểm tra xem ứng dụng Ollama đã được khởi chạy chưa."
            )
        except Exception as e:
            logger.error(f"Lỗi khi gọi Ollama API ({self.model_name}): {e}")
            raise

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Gửi prompt tới Ollama và stream từng chunk trả về."""
        payload = self._prepare_payload(
            prompt, system_prompt, temperature, max_tokens, stream=True
        )
        url = f"{self.base_url}/api/chat"

        try:
            with requests.post(url, json=payload, stream=True, timeout=self.timeout) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        chunk_data = json.loads(line.decode("utf-8"))
                        content = chunk_data.get("message", {}).get("content", "")
                        if content:
                            yield content
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Không thể kết nối tới Ollama tại {self.base_url}.")
        except Exception as e:
            logger.error(f"Lỗi khi stream từ Ollama API: {e}")
            raise

