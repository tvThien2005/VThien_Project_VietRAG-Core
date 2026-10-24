import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.prompt_builder import PromptBuilder
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.pipeline import RetrievalPipeline
from src.schemas.response import RetrievedChunk


class TestGenerationAndPipeline(unittest.TestCase):
    def setUp(self):
        self.builder = PromptBuilder()

    def test_format_context(self):
        chunks = [
            RetrievedChunk(
                chunk_id="chunk-1",
                text="Mô hình DenseNet201 đạt kết quả 93%.",
                score=0.9,
                metadata={
                    "source": "paper.docx",
                    "file_name": "paper.docx",
                    "chapter": "Chương 4: Thực nghiệm",
                    "section_title": "Kết luận",
                    "page_number": 12,
                },
            )
        ]
        context = self.builder.format_context(chunks)
        self.assertIn("- Thuộc chương: Chương 4: Thực nghiệm", context)
        self.assertIn("- Thuộc mục: Kết luận", context)
        self.assertIn("- Nội dung: Mô hình DenseNet201 đạt kết quả 93%.", context)
        self.assertIn("- Trang + tên file tài liệu: 12 - paper.docx", context)
        self.assertIn("- Nguồn: paper.docx", context)
        self.assertIn("- Đường dẫn mục: Không xác định", context)

    def test_format_context_uses_fallback_when_metadata_is_missing(self):
        chunk = RetrievedChunk(
            chunk_id="chunk-without-location",
            text="Nội dung không có metadata vị trí.",
            score=0.8,
        )

        context = self.builder.format_context([chunk])

        self.assertIn("- Thuộc chương: Không xác định", context)
        self.assertIn("- Thuộc mục: Không xác định", context)
        self.assertIn("- Nguồn: Không xác định", context)
        self.assertIn(
            "- Trang + tên file tài liệu: Không xác định - Không xác định",
            context,
        )

    def test_format_context_preserves_table_caption_and_identity(self):
        chunk = RetrievedChunk(
            chunk_id="table-chunk",
            text="Caption: Table 2\n| Metric | Value |",
            score=0.8,
            metadata={
                "source": "C:/docs/report.docx",
                "file_name": "report.docx",
                "section_path": ["Results", "Model comparison"],
                "page_start": 12,
                "page_end": 13,
                "chunk_has_table": True,
                "table_id": "table-2",
                "table_caption": "Table 2",
                "table_caption_status": "present",
            },
        )

        context = self.builder.format_context([chunk])

        self.assertIn("- Nguồn: C:/docs/report.docx", context)
        self.assertIn("- Đường dẫn mục: Results > Model comparison", context)
        self.assertIn("- Caption bảng: Table 2", context)
        self.assertIn("- Table ID: table-2", context)
        self.assertIn("- Trạng thái caption: present", context)
        self.assertIn("- Trang + tên file tài liệu: 12-13 - report.docx", context)

    def test_build_prompt(self):
        chunks = [
            RetrievedChunk(
                chunk_id="chunk-1",
                text="Đồ án sử dụng kỹ thuật phân đoạn ảnh.",
                score=0.85,
                metadata={"file_name": "paper.pdf"},
            )
        ]
        sys_prompt, user_prompt = self.builder.build_prompt("Đồ án làm về gì?", chunks)
        self.assertIn("phân đoạn ảnh", user_prompt)
        self.assertIn("Đồ án làm về gì?", user_prompt)
        self.assertIn("tiếng Việt", sys_prompt)


class TestRAGPipelineContextThreshold(unittest.TestCase):
    def setUp(self):
        self.low_score_chunk = RetrievedChunk(
            chunk_id="low-score",
            text="Nội dung không đủ liên quan.",
            score=0.099,
        )
        self.threshold_chunk = RetrievedChunk(
            chunk_id="threshold",
            text="Nội dung đạt ngưỡng.",
            score=0.1,
        )

    def test_query_only_sends_chunks_at_or_above_threshold_to_llm(self):
        retrieval = Mock()
        retrieval.retrieve.return_value = [
            self.low_score_chunk,
            self.threshold_chunk,
        ]
        generator = Mock()
        generator.generate_answer.return_value = "Câu trả lời"
        pipeline = RAGPipeline(retrieval_pipeline=retrieval, generator=generator)

        response = pipeline.query("Câu hỏi")

        generator.generate_answer.assert_called_once_with(
            query="Câu hỏi",
            chunks=[self.threshold_chunk],
            temperature=None,
            max_tokens=None,
        )
        self.assertEqual(response.sources, [self.threshold_chunk])

    def test_query_does_not_call_llm_when_all_chunks_are_below_threshold(self):
        retrieval = Mock()
        retrieval.retrieve.return_value = [self.low_score_chunk]
        generator = Mock()
        pipeline = RAGPipeline(retrieval_pipeline=retrieval, generator=generator)

        response = pipeline.query("Câu hỏi")

        generator.generate_answer.assert_not_called()
        self.assertEqual(response.sources, [])
        self.assertIn("không tìm thấy thông tin", response.answer)

    def test_query_keeps_low_rrf_scores_when_hybrid_search_is_enabled(self):
        retrieval = Mock()
        retrieval.retrieve.return_value = [
            RetrievedChunk(chunk_id="hybrid", text="Context hybrid", score=0.01)
        ]
        generator = Mock()
        generator.generate_answer.return_value = "Câu trả lời"
        pipeline = RAGPipeline(retrieval_pipeline=retrieval, generator=generator)

        response = pipeline.query("Câu hỏi", use_hybrid=True)

        self.assertEqual([chunk.chunk_id for chunk in response.sources], ["hybrid"])
        generator.generate_answer.assert_called_once()

    def test_query_keeps_low_cross_encoder_scores_when_reranking_is_enabled(self):
        retrieval = Mock()
        retrieval.retrieve.return_value = [
            RetrievedChunk(chunk_id="reranked", text="Context reranked", score=-0.2)
        ]
        generator = Mock()
        generator.generate_answer.return_value = "Câu trả lời"
        pipeline = RAGPipeline(retrieval_pipeline=retrieval, generator=generator)

        response = pipeline.query("Câu hỏi", use_rerank=True)

        self.assertEqual([chunk.chunk_id for chunk in response.sources], ["reranked"])
        generator.generate_answer.assert_called_once()

    def test_stream_does_not_call_llm_when_all_chunks_are_below_threshold(self):
        retrieval = Mock()
        retrieval.retrieve.return_value = [self.low_score_chunk]
        generator = Mock()
        pipeline = RAGPipeline(retrieval_pipeline=retrieval, generator=generator)

        chunks, stream = pipeline.stream_query("Câu hỏi")

        generator.stream_answer.assert_not_called()
        self.assertEqual(chunks, [])
        self.assertIn("không tìm thấy thông tin", "".join(stream))


class TestRetrievalPipelineRerankerOverride(unittest.TestCase):
    @patch("src.retrieval.pipeline.get_reranker")
    def test_explicit_request_can_enable_reranker_when_config_default_is_off(self, get_reranker):
        chunk = RetrievedChunk(chunk_id="candidate", text="Nội dung", score=0.8)
        active_reranker = Mock()
        active_reranker.rerank.return_value = [chunk]
        get_reranker.return_value = active_reranker

        pipeline = object.__new__(RetrievalPipeline)
        pipeline.cfg = SimpleNamespace(
            retrieval=SimpleNamespace(
                reranker=SimpleNamespace(enabled=False, top_n=1),
                hybrid_search=SimpleNamespace(enabled=False),
            )
        )
        pipeline.retriever = Mock()
        pipeline.retriever.retrieve.return_value = [chunk]
        pipeline.hybrid_retriever = Mock()
        pipeline.vector_retriever = pipeline.retriever
        pipeline.reranker = None
        pipeline.context_expander = None

        result = pipeline.retrieve(
            query="Câu hỏi",
            top_k=1,
            top_n=1,
            use_rerank=True,
            use_hybrid=False,
        )

        get_reranker.assert_called_once_with(force=True)
        active_reranker.rerank.assert_called_once()
        self.assertEqual(result, [chunk])

if __name__ == "__main__":
    unittest.main()
