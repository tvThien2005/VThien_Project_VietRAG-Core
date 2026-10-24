import argparse
import sys
import time
from pathlib import Path

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.evaluation.dataset import EvaluationDataset
from src.evaluation.evaluator import RAGEvaluator
from src.logger import get_logger
from src.pipeline.rag_pipeline import RAGPipeline

logger = get_logger("run_eval")


def main():
    parser = argparse.ArgumentParser(description="Chương trình Đánh giá Tự động Hệ thống RAG tiếng Việt.")
    parser.add_argument(
        "--testset",
        "-t",
        type=str,
        default="data/eval/testset.jsonl",
        help="Đường dẫn file testset.jsonl cần đánh giá.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/eval/evaluation_results.json",
        help="Đường dẫn lưu kết quả tổng hợp JSON.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/eval/evaluation_details.csv",
        help="Đường dẫn lưu chi tiết từng câu hỏi dưới dạng CSV.",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Số lượng mẫu giới hạn để test nhanh (mặc định đánh giá toàn bộ).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Số lượng chunks truy xuất trước khi Rerank.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Số lượng chunks giữ lại sau khi Rerank đưa vào LLM.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Tắt Reranker, chỉ kiểm thử Vector Retrieval thuần.",
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Tắt Hybrid search, chỉ dùng Dense embedding search.",
    )
    parser.add_argument(
        "--use-qa-prompt",
        action="store_true",
        help="Sử dụng qa_prompt.yaml (chi tiết, trích dẫn nguồn) thay vì mặc định eval_prompt.yaml (trực diện, súc tích).",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Tắt đánh giá bằng LLM Judge (chỉ dùng rule-based và Semantic Similarity để tăng tốc).",
    )
    parser.add_argument(
        "--judge-api-key",
        type=str,
        default=None,
        help="API key riêng cho LLM Judge (nếu muốn tách biệt với key sinh câu trả lời).",
    )

    args = parser.parse_args()
    cfg = get_config()

    testset_path = Path(args.testset)
    if not testset_path.exists():
        print(f"❌ Không tìm thấy file dữ liệu testset tại: {testset_path}")
        sys.exit(1)

    prompt_name = "qa_prompt.yaml (Chi tiết)" if args.use_qa_prompt else "eval_prompt.yaml (Trực diện)"
    print("=" * 80)
    print("🚀 BẮT ĐẦU CHƯƠNG TRÌNH ĐÁNH GIÁ HỆ THỐNG RAG")
    print(f"📄 Tập dữ liệu : {testset_path}")
    print(f"⚙️  Rerank      : {'TẮT (Chỉ Vector Search)' if args.no_rerank else 'BẬT (BGE Reranker)'}")
    print(f"⚙️  Hybrid      : {'TẮT' if args.no_hybrid else 'BẬT'}")
    print(f"📝 Prompt      : {prompt_name}")
    print(f"⚖️  LLM Judge   : {'TẮT' if args.no_judge else f'BẬT ({cfg.evaluation.judge_llm.model_name})'}")
    print(f"🤖 LLM Gen      : {cfg.llm.default_provider} ({cfg.llm.groq.model_name})")
    print("=" * 80)

    # 1. Khởi tạo Pipeline & Evaluator với prompt eval_prompt.yaml
    rag_pipeline = RAGPipeline()
    evaluator = RAGEvaluator(
        pipeline=rag_pipeline,
        use_eval_prompt=not args.use_qa_prompt,
        enable_llm_judge=not args.no_judge,
        judge_api_key=args.judge_api_key,
    )

    # 2. Thực thi đánh giá
    use_rerank = False if args.no_rerank else None
    use_hybrid = False if args.no_hybrid else None

    result = evaluator.evaluate(
        dataset=testset_path,
        top_k=args.top_k,
        top_n=args.top_n,
        use_rerank=use_rerank,
        use_hybrid=use_hybrid,
        max_samples=args.limit,
    )

    # 3. Hiển thị kết quả ra màn hình
    result.print_summary()

    # 4. Xuất file kết quả
    result.save_json(args.output)
    result.save_csv(args.output_csv)

    print(f"✅ Đã lưu kết quả tóm tắt: {args.output}")
    print(f"✅ Đã lưu chi tiết từng câu: {args.output_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
