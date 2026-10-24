import json
import re
from typing import Dict, Optional, Tuple
import groq

from src.config import get_config
from src.logger import get_logger
from src.utils.rate_limit import call_with_rate_limit_retry

logger = get_logger("llm_judge")

JUDGE_PROMPT_TEMPLATE = """Bạn là một trọng tài AI (LLM-as-a-Judge) công tâm chuyên đánh giá chất lượng câu trả lời của hệ thống RAG (Retrieval-Augmented Generation).

Nhiệm vụ: Đánh giá xem CÂU TRẢ LỜI CỦA MÔ HÌNH có giải quyết ĐÚNG TRỌNG TÂM CÂU HỎI của người dùng hay không (Answer Relevancy / Trọng tâm).

Thông tin đánh giá:
- CÂU HỎI CỦA USER: {query}
- CÂU TRẢ LỜI CỦA MÔ HÌNH: {prediction}
- ĐÁP ÁN CHUẨN THAM KHẢO (Ground Truth): {ground_truth}
- LOẠI CÂU HỎI: {question_type} (Có đáp án trong tài liệu: {is_answerable})

Quy tắc chấm điểm (Thang điểm từ 0.0 đến 1.0):
1. Điểm 1.0: Trả lời trực diện, đúng trọng tâm, khớp với yêu cầu của câu hỏi.
2. Điểm 0.7 - 0.9: Đi đúng hướng nhưng có thông tin giải thích thừa hoặc diễn đạt chưa thật sự trực diện.
3. Điểm 0.4 - 0.6: Chỉ giải quyết được một phần câu hỏi, hoặc có nhiều thông tin lạc đề lan man.
4. Điểm 0.0 - 0.3: Hoàn toàn lạc đề, trả lời sai hoặc không trả lời được câu hỏi.
5. ĐẶC BIỆT: Với câu hỏi KHÔNG CÓ THÔNG TIN trong tài liệu (is_answerable = False), nếu mô hình thẳng thắn từ chối trả lời vì không có thông tin -> Cho 1.0 điểm (đạt chuẩn chống ảo giác). Nếu tự bịa ra đáp án -> Cho 0.0 điểm.

Hãy trả về duy nhất một đối tượng JSON hợp lệ (không kèm theo bất kỳ văn bản nào khác ngoài JSON):
{{
  "score": <điểm số dạng số thực từ 0.0 đến 1.0>,
  "reason": "<nhận xét ngắn gọn 1 câu giải thích lý do cho điểm>"
}}
"""


class LLMJudge:
    """Đánh giá chất lượng câu trả lời RAG bằng mô hình ngôn ngữ lớn (LLM-as-a-Judge)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        cfg = get_config()
        self.provider = provider or cfg.evaluation.judge_llm.provider
        self.model_name = model_name or cfg.evaluation.judge_llm.model_name
        # Ưu tiên api_key truyền vào -> api_key của judge trong config/env -> api_key groq chung
        judge_key = getattr(cfg.evaluation.judge_llm, "api_key", None)
        groq_key = getattr(cfg.llm.groq, "api_key", None) if hasattr(cfg.llm, "groq") else None
        self.api_key = api_key or judge_key or groq_key


        if not self.api_key:
            logger.warning(
                "Chưa tìm thấy API Key cho LLM Judge (EVAL_GROQ_API_KEY hoặc GROQ_API_KEY). "
                "Tính năng LLM Judge sẽ bị tạm vô hiệu hóa."
            )
            self.client = None
        else:
            try:
                self.client = groq.Groq(
                    api_key=self.api_key,
                    timeout=30.0,
                    max_retries=0,
                )
                logger.info(f"Đã khởi tạo LLM Judge với model: {self.model_name}")
            except Exception as e:
                logger.error(f"Lỗi khi khởi tạo Groq Client cho LLM Judge: {e}")
                self.client = None

    def evaluate_relevancy(
        self,
        query: str,
        prediction: str,
        ground_truth: str,
        question_type: str = "direct",
        is_answerable: bool = True,
    ) -> Tuple[float, str]:
        """Đánh giá Answer Relevancy (độ đúng trọng tâm câu hỏi).
        
        Returns:
            Tuple[float, str]: (điểm số từ 0.0 đến 1.0, lý do nhận xét)
        """
        if not self.client:
            return 0.0, "LLM Judge client không khả dụng (thiếu API Key)."

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            query=query.strip(),
            prediction=prediction.strip(),
            ground_truth=ground_truth.strip(),
            question_type=question_type,
            is_answerable=str(is_answerable),
        )

        try:
            response = call_with_rate_limit_retry(
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "Bạn là chuyên gia đánh giá QA và RAG chuẩn xác. Luôn trả về định dạng JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                ),
                operation_name="Groq LLM Judge",
            )

            raw_text = response.choices[0].message.content or "{}"
            data = json.loads(raw_text)
            score = float(data.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            reason = str(data.get("reason", ""))
            return score, reason

        except Exception as e:
            logger.warning(f"Lỗi khi gọi LLM Judge chấm điểm: {e}. Thử regex fallback.")
            try:
                # Fallback parse nếu có lỗi định dạng
                match = re.search(r'"score"\s*:\s*([0-9.]+)', str(e))
                if match:
                    return float(match.group(1)), "Parsed from exception response"
            except Exception:
                pass
            return 0.0, f"Lỗi chấm điểm: {str(e)[:100]}"
