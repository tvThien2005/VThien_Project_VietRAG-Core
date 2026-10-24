import argparse
import sys
import time
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.logger import get_logger
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.vector_retriever import VectorRetriever

logger = get_logger("run_query")


def main():
    parser = argparse.ArgumentParser(description="Hệ thống hỏi đáp RAG tiếng Việt trên tài liệu.")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Câu hỏi / câu truy vấn tìm kiếm (tiếng Việt).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Số lượng chunks lấy từ Vector DB trước khi Rerank.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Số lượng chunks giữ lại cho LLM trả lời.",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Chỉ truy xuất trong document ID này.",
    )
    parser.add_argument(
        "--file-name",
        type=str,
        default=None,
        help="Chỉ truy xuất trong tên file này.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="Collection Qdrant cần truy vấn (dùng để kiểm thử index versioned mới).",
    )
    parser.add_argument(
        "--dense-vector-name",
        type=str,
        default=None,
        help="Tên dense vector trong collection (mặc định 'dense' khi chỉ định collection).",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Tắt Reranker, chỉ dùng Vector search.",
    )
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Chỉ hiển thị các chunks tìm được, không gọi LLM sinh câu trả lời.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Bật chế độ streaming câu trả lời từ LLM.",
    )

    args = parser.parse_args()
    cfg = get_config()

    query = args.query
    if not query:
        query = input("❓ Nhập câu hỏi tìm kiếm: ").strip()

    if not query:
        print("⚠️ Câu truy vấn rỗng.")
        return

    top_k = args.top_k or cfg.retrieval.top_k
    top_n = args.top_n or cfg.retrieval.reranker.top_n

    print("=" * 80)
    print(f"🔍 CÂU HỎI: '{query}'")
    print(
        f"⚙️  Cấu hình: top_k={top_k}, top_n={top_n}, "
        f"rerank={not args.no_rerank}, LLM={cfg.llm.default_provider} ({cfg.llm.groq.model_name})"
    )
    print("=" * 80)

    if args.collection_name:
        vector_retriever = VectorRetriever(
            collection_name=args.collection_name,
            dense_vector_name=args.dense_vector_name or "dense",
        )
        rag = RAGPipeline(
            retrieval_pipeline=RetrievalPipeline(
                retriever=vector_retriever,
            )
        )
    else:
        rag = RAGPipeline()
    if args.no_rerank:
        rag.retrieval_pipeline.reranker = None

    start_time = time.time()

    # Chế độ chỉ truy xuất
    if args.retrieve_only:
        chunks = rag.retrieval_pipeline.retrieve(
            query=query,
            top_k=top_k,
            top_n=top_n,
            doc_id=args.doc_id,
            file_name=args.file_name,
        )
        latency = time.time() - start_time
        print(f"\n🎯 ĐÃ TÌM THẤY {len(chunks)} CHUNKS LIÊN QUAN ({latency:.2f}s):\n")
        for chunk in chunks:
            meta = chunk.metadata
            file_name = meta.get("file_name", "N/A")
            page = meta.get("page_number")
            section = meta.get("section_title")
            loc_info = f"Trang {page}" if page else (f"Mục: '{section}'" if section else "")

            print(f"┌─ [Hạng #{chunk.rank}] - Điểm: {chunk.score:.4f} ─────────────────────────────")
            print(f"│ 📄 Nguồn   : {file_name} ({loc_info})")
            print(f"│ 📝 Nội dung:\n│")
            for line in chunk.text.split("\n"):
                print(f"│   {line}")
            print("└" + "─" * 70 + "\n")
        return

    # Chế độ Streaming
    if args.stream:
        chunks, stream_iter = rag.stream_query(
            query=query,
            top_k=top_k,
            top_n=top_n,
            doc_id=args.doc_id,
            file_name=args.file_name,
        )
        print(f"\n📚 NGUỒN TÀI LIỆU THAM KHẢO ({len(chunks)} đoạn):")
        for c in chunks:
            sec = c.metadata.get("section_title") or f"Trang {c.metadata.get('page_number', 'N/A')}"
            print(f"  • {c.metadata.get('file_name')} ({sec}) - Điểm: {c.score:.4f}")

        print("\n🤖 CÂU TRẢ LỜI CỦA AI:\n")
        for token in stream_iter:
            print(token, end="", flush=True)
        print("\n\n" + "=" * 80)
        return

    # Chế độ RAG hoàn chỉnh (mặc định)
    try:
        response = rag.query(
            query_text=query,
            top_k=top_k,
            top_n=top_n,
            doc_id=args.doc_id,
            file_name=args.file_name,
        )
    except Exception as e:
        print(f"\n❌ Lỗi khi thực thi RAG Pipeline: {e}")
        return

    print(f"\n🤖 CÂU TRẢ LỜI CỦA AI ({response.latency_seconds:.2f}s):\n")
    print(response.answer)

    print("\n" + "-" * 80)
    print(f"📚 NGUỒN TÀI LIỆU TRÍCH DẪN ({len(response.sources)} đoạn):")
    for chunk in response.sources:
        sec = chunk.metadata.get("section_title") or f"Trang {chunk.metadata.get('page_number', 'N/A')}"
        print(f"  • [Hạng #{chunk.rank}] {chunk.metadata.get('file_name')} ({sec}) | Điểm: {chunk.score:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
