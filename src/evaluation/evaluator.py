import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from tqdm import tqdm

from src.config import get_config
from src.evaluation.dataset import EvaluationDataset, EvalSample
from src.evaluation.judge import LLMJudge
from src.evaluation.metrics import (
    check_hallucination_rejection,
    compute_exact_match,
    compute_retrieval_metrics,
    compute_semantic_similarity,
    compute_token_f1,
)
from src.generation.prompt_builder import PromptBuilder
from src.logger import get_logger
from src.pipeline.rag_pipeline import RAGPipeline

logger = get_logger("rag_evaluator")


class EvaluationResult:
    """Chứa kết quả đánh giá chi tiết và các hàm tiện ích báo cáo."""

    def __init__(self, records: List[Dict[str, Any]], summary: Dict[str, Any]):
        self.records = records
        self.summary = summary

    def to_dataframe(self) -> pd.DataFrame:
        """Xuất danh sách kết quả chi tiết từng câu hỏi thành DataFrame."""
        return pd.DataFrame(self.records)

    def to_summary_dataframe(self) -> pd.DataFrame:
        """Xuất bảng tổng hợp các chỉ số thành DataFrame."""
        flat_data = []
        for category, metrics in self.summary.items():
            if isinstance(metrics, dict):
                for metric_name, val in metrics.items():
                    flat_data.append({
                        "Phân nhóm": category,
                        "Chỉ số": metric_name,
                        "Điểm số": round(val, 4) if isinstance(val, float) else val,
                    })
        return pd.DataFrame(flat_data)

    def print_summary(self):
        """In bảng tóm tắt kết quả đẹp mắt ra terminal."""
        print("\n" + "=" * 80)
        print("📊 BÁO CÁO ĐÁNH GIÁ CHẤT LƯỢNG HỆ THỐNG RAG")
        print("=" * 80)

        for section, stats in self.summary.items():
            print(f"\n📌 {section.upper()}:")
            if isinstance(stats, dict):
                for k, v in stats.items():
                    if isinstance(v, float):
                        print(f"  • {k: <34}: {v:.4f} ({v*100:.2f}%)" if v <= 1.0 else f"  • {k: <34}: {v:.4f}")
                    else:
                        print(f"  • {k: <34}: {v}")
            else:
                print(f"  {stats}")
        print("=" * 80 + "\n")

    def save_json(self, output_path: Union[str, Path]):
        """Lưu toàn bộ chi tiết và summary ra file JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.summary,
            "details": self.records,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Đã lưu kết quả đánh giá JSON tại: {path}")

    def save_csv(self, output_path: Union[str, Path]):
        """Lưu bảng kết quả chi tiết từng câu hỏi ra file CSV."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"Đã lưu bảng chi tiết CSV tại: {path}")


class RAGEvaluator:
    """Bộ đánh giá toàn diện End-to-End RAG Pipeline."""

    def __init__(
        self,
        pipeline: Optional[RAGPipeline] = None,
        use_eval_prompt: bool = True,
        prompt_template_path: Optional[Union[str, Path]] = None,
        embedder: Optional[object] = None,
        enable_llm_judge: bool = True,
        judge_api_key: Optional[str] = None,
    ):
        self.cfg = get_config()
        self.pipeline = pipeline or RAGPipeline()

        # 1. Cấu hình Prompt Template cho LLM Generator (mặc định dùng eval_prompt.yaml để trả lời trực diện)
        if prompt_template_path:
            target_prompt_path = Path(prompt_template_path)
        elif use_eval_prompt:
            target_prompt_path = self.cfg.prompts.eval_path
        else:
            target_prompt_path = self.cfg.prompts.qa_path

        if target_prompt_path.exists():
            logger.info(f"RAGEvaluator sử dụng template: {target_prompt_path.name}")
            self.pipeline.generator.prompt_builder = PromptBuilder(template_path=target_prompt_path)
        else:
            logger.warning(f"Không tìm thấy file prompt {target_prompt_path}, dùng prompt mặc định.")

        # 2. Embedder phục vụ tính Semantic Similarity (tái sử dụng từ retriever để tiết kiệm RAM)
        if embedder:
            self.embedder = embedder
        elif (
            hasattr(self.pipeline, "retrieval_pipeline")
            and hasattr(self.pipeline.retrieval_pipeline, "vector_retriever")
            and hasattr(self.pipeline.retrieval_pipeline.vector_retriever, "embedder")
        ):
            self.embedder = self.pipeline.retrieval_pipeline.vector_retriever.embedder
        else:
            from src.ingestion.embedder import BGEEmbedder
            self.embedder = BGEEmbedder()

        # 3. LLM Judge phục vụ chấm Answer Relevancy
        self.enable_llm_judge = enable_llm_judge
        self.judge = LLMJudge(api_key=judge_api_key) if enable_llm_judge else None

    def evaluate(
        self,
        dataset: Union[EvaluationDataset, str, Path],
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        use_rerank: Optional[bool] = None,
        use_hybrid: Optional[bool] = None,
        max_samples: Optional[int] = None,
    ) -> EvaluationResult:
        """Thực hiện chạy toàn bộ tập dữ liệu qua RAG Pipeline và tính toán các chỉ số."""
        if isinstance(dataset, (str, Path)):
            dataset = EvaluationDataset.load_jsonl(dataset)

        samples = dataset.samples
        if max_samples and max_samples > 0:
            samples = samples[:max_samples]

        logger.info(f"Bắt đầu đánh giá RAG trên {len(samples)} câu hỏi...")

        records: List[Dict[str, Any]] = []

        for sample in tqdm(samples, desc="Evaluating RAG", unit="query"):
            start_t = time.time()

            # 1. Gọi RAG Pipeline
            try:
                response = self.pipeline.query(
                    query=sample.query,
                    top_k=top_k,
                    top_n=top_n,
                    use_rerank=use_rerank,
                    use_hybrid=use_hybrid,
                )
                pred_answer = response.answer
                retrieved_chunks = [c.text for c in response.sources]
                latency = response.latency_seconds or (time.time() - start_t)
            except Exception as e:
                logger.error(f"Lỗi khi xử lý câu hỏi '{sample.id}': {e}")
                pred_answer = f"ERROR: {e}"
                retrieved_chunks = []
                latency = time.time() - start_t

            # 2. Đo chỉ số Retrieval (chỉ tính trên các câu hỏi có đáp án trong tài liệu)
            ret_metrics = compute_retrieval_metrics(
                retrieved_texts=retrieved_chunks,
                gold_contexts=sample.ground_truth_contexts,
                k_list=[1, 3, 5],
                ground_truth=sample.ground_truth,
            )

            # 3. Đo chỉ số Generation & Ngữ nghĩa
            if sample.is_answerable:
                f1_res = compute_token_f1(pred_answer, sample.ground_truth)
                em = compute_exact_match(pred_answer, sample.ground_truth)
                rejection_correct = None
                # Semantic Similarity
                sem_sim = compute_semantic_similarity(
                    prediction=pred_answer,
                    ground_truth=sample.ground_truth,
                    embedder=self.embedder,
                )
            else:
                # Câu hỏi không có trong tài liệu -> kiểm tra bot có từ chối chính xác không
                f1_res = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                em = 0.0
                is_rejected = check_hallucination_rejection(pred_answer)
                rejection_correct = 1.0 if is_rejected else 0.0
                # Đối với unanswerable, nếu từ chối đúng thì sem_sim = 1.0
                sem_sim = 1.0 if is_rejected else 0.0

            # 4. LLM Judge: Chấm điểm Answer Relevancy (Trọng tâm câu hỏi)
            relevancy_score = 0.0
            judge_reason = ""
            if self.enable_llm_judge and self.judge and self.judge.client:
                try:
                    relevancy_score, judge_reason = self.judge.evaluate_relevancy(
                        query=sample.query,
                        prediction=pred_answer,
                        ground_truth=sample.ground_truth,
                        question_type=sample.question_type,
                        is_answerable=sample.is_answerable,
                    )
                except Exception as e:
                    logger.warning(f"Lỗi khi gọi LLM Judge cho {sample.id}: {e}")
                    relevancy_score = 0.0
                    judge_reason = str(e)

            record = {
                "id": sample.id,
                "query": sample.query,
                "ground_truth": sample.ground_truth,
                "generated_answer": pred_answer,
                "question_type": sample.question_type,
                "is_answerable": sample.is_answerable,
                "source_doc": sample.source_doc,
                "num_retrieved_chunks": len(retrieved_chunks),
                "latency_seconds": latency,
                # Retrieval metrics
                "hit@1": ret_metrics["hit_rate@1"],
                "hit@3": ret_metrics["hit_rate@3"],
                "hit@5": ret_metrics["hit_rate@5"],
                "mrr": ret_metrics["mrr"],
                "context_recall": ret_metrics["context_recall"],
                # Generation metrics
                "semantic_similarity": sem_sim,
                "answer_relevancy": relevancy_score if self.enable_llm_judge else None,
                "judge_reason": judge_reason if self.enable_llm_judge else "",
                "token_f1": f1_res["f1"],
                "token_precision": f1_res["precision"],
                "token_recall": f1_res["recall"],
                "exact_match": em,
                "rejection_correct": rejection_correct,
            }
            records.append(record)

        # 5. Tính toán tổng hợp Summary
        summary = self._compute_summary(records)

        return EvaluationResult(records=records, summary=summary)

    def _compute_summary(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Tổng hợp các chỉ số trung bình theo toàn bộ và theo từng nhóm câu hỏi."""
        total_queries = len(records)
        if total_queries == 0:
            return {}

        df = pd.DataFrame(records)

        # 1. Retrieval (lọc các câu answerable vì unanswerable không có gold context)
        ans_df = df[df["is_answerable"] == True]
        num_ans = len(ans_df)

        retrieval_summary = {
            "Total Answerable Queries": num_ans,
            "Hit Rate @ 1": float(ans_df["hit@1"].mean()) if num_ans > 0 else 0.0,
            "Hit Rate @ 3": float(ans_df["hit@3"].mean()) if num_ans > 0 else 0.0,
            "Hit Rate @ 5": float(ans_df["hit@5"].mean()) if num_ans > 0 else 0.0,
            "MRR (Mean Reciprocal Rank)": float(ans_df["mrr"].mean()) if num_ans > 0 else 0.0,
            "Context Recall": float(ans_df["context_recall"].mean()) if num_ans > 0 else 0.0,
        }

        # 2. Generation & Semantic Performance (Answerable questions)
        generation_summary = {
            "Mean Semantic Similarity (Cosine)": float(ans_df["semantic_similarity"].mean()) if num_ans > 0 else 0.0,
            "Mean Answer Relevancy (LLM Judge)": float(df["answer_relevancy"].mean()) if (self.enable_llm_judge and "answer_relevancy" in df and df["answer_relevancy"].notna().any()) else 0.0,
            "Mean Token F1-Score": float(ans_df["token_f1"].mean()) if num_ans > 0 else 0.0,
            "Mean Token Precision": float(ans_df["token_precision"].mean()) if num_ans > 0 else 0.0,
            "Mean Token Recall": float(ans_df["token_recall"].mean()) if num_ans > 0 else 0.0,
            "Exact Match Rate": float(ans_df["exact_match"].mean()) if num_ans > 0 else 0.0,
        }

        # 3. Anti-Hallucination (Unanswerable questions)
        unans_df = df[df["is_answerable"] == False]
        num_unans = len(unans_df)
        rejection_rate = float(unans_df["rejection_correct"].mean()) if num_unans > 0 else 1.0

        hallucination_summary = {
            "Total Unanswerable Queries": num_unans,
            "Correct Rejection Rate (Anti-Hallucination)": rejection_rate,
            "Hallucination Rate": (1.0 - rejection_rate) if num_unans > 0 else 0.0,
        }

        # 4. System Performance
        perf_summary = {
            "Total Evaluated Queries": total_queries,
            "Average Latency (seconds)": float(df["latency_seconds"].mean()),
            "Min Latency": float(df["latency_seconds"].min()),
            "Max Latency": float(df["latency_seconds"].max()),
        }

        # 5. Breakdown theo Question Type
        breakdown_by_type = {}
        for q_type, group in df.groupby("question_type"):
            breakdown_by_type[str(q_type)] = {
                "Count": int(len(group)),
                "Avg Hit@3": round(float(group["hit@3"].mean()), 4) if "hit@3" in group else 0.0,
                "Avg Semantic Sim": round(float(group["semantic_similarity"].mean()), 4) if "semantic_similarity" in group else 0.0,
                "Avg Relevancy (Judge)": round(float(group["answer_relevancy"].mean()), 4) if ("answer_relevancy" in group and group["answer_relevancy"].notna().any()) else 0.0,
                "Avg F1": round(float(group["token_f1"].mean()), 4) if "token_f1" in group else 0.0,
                "Avg Latency (s)": round(float(group["latency_seconds"].mean()), 2),
            }

        return {
            "retrieval_performance": retrieval_summary,
            "generation_performance": generation_summary,
            "anti_hallucination": hallucination_summary,
            "system_latency": perf_summary,
            "breakdown_by_question_type": breakdown_by_type,
        }
