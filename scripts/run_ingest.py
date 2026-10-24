import argparse
import sys
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.ingestion.embedder import BGEEmbedder
from src.ingestion.indexer import QdrantIndexer
from src.ingestion.pipeline import IngestionPipeline
from src.logger import get_logger

logger = get_logger("run_ingest")


def main():
    parser = argparse.ArgumentParser(description="Ingestion Pipeline: Đọc, cắt nhỏ và index tài liệu vào Qdrant.")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Đường dẫn đến file hoặc thư mục tài liệu (mặc định lấy data/raw từ config).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Kích thước chunk (nếu muốn ghi đè cấu hình trong config.yaml).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Độ gối đầu overlap (nếu muốn ghi đè cấu hình trong config.yaml).",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Tạo embedding và index vào collection legacy hiện tại.",
    )
    parser.add_argument(
        "--versioned-index",
        action="store_true",
        help="Tạo collection versioned riêng (dense + BM25), không sửa collection hiện tại.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="Tên collection mới cho --versioned-index; tên đã tồn tại sẽ bị từ chối.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Xóa collection legacy hiện tại và tạo lại khi dùng --index (destructive).",
    )

    args = parser.parse_args()
    cfg = get_config()
    if args.index and args.versioned_index:
        parser.error("Use either --index or --versioned-index, not both.")
    if args.versioned_index and args.recreate:
        parser.error("--recreate cannot be combined with the safe --versioned-index flow.")
    if args.collection_name and not args.versioned_index:
        parser.error("--collection-name requires --versioned-index.")

    print("=" * 70)
    print("🚀 BẮT ĐẦU CHẠY INGESTION PIPELINE")
    print(f"📂 Thư mục dữ liệu: {args.source or cfg.data.raw_path}")
    print(
        f"⚙️  Cấu hình: Chunk Size={args.chunk_size or cfg.chunking.recursive.chunk_size}, "
        f"Overlap={args.chunk_overlap or cfg.chunking.recursive.chunk_overlap}"
    )
    print("=" * 70)

    pipeline = IngestionPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    chunks = pipeline.run(source_path=args.source, save_to_processed=True)

    if not chunks:
        print("⚠️ Không có chunk nào được tạo ra.")
        return

    # Thống kê kết quả
    files_counter = {}
    total_chars = 0
    total_words = 0

    for c in chunks:
        file_name = c.metadata.get("file_name", "unknown")
        files_counter[file_name] = files_counter.get(file_name, 0) + 1
        total_chars += c.char_count
        total_words += c.word_count

    avg_chars = total_chars // len(chunks)
    avg_words = total_words // len(chunks)

    print("\n" + "=" * 70)
    print("🎉 HOÀN TẤT CẮT CHUNKS!")
    print(f"📊 Tổng số Chunks tạo ra : {len(chunks)}")
    print(f"📝 Kích thước trung bình  : {avg_chars} ký tự (~{avg_words} từ / chunk)")
    print("-" * 70)
    print("Chi tiết theo từng file:")
    for fname, count in files_counter.items():
        print(f"  • {fname:<40}: {count:>5} chunks")
    print("=" * 70)

    # Nếu người dùng truyền cờ --index, thực hiện đẩy vào Qdrant
    if args.index or args.versioned_index:
        print("\n" + "=" * 70)
        print("⚡ TIẾN HÀNH EMBED VÀ INDEX VÀO QDRANT...")
        print("=" * 70)

        indexer = QdrantIndexer()
        if not indexer.is_connected():
            print(f"❌ Không thể kết nối tới Qdrant tại {indexer.cfg.vector_store.url}")
            print("👉 Vui lòng mở Docker Desktop và chạy lệnh: docker compose up -d")
            return

        embedder = BGEEmbedder()
        if args.versioned_index:
            collection_name = args.collection_name or indexer.versioned_collection_name(
                chunks,
                embedder,
            )
            indexed_count = indexer.index_chunks_versioned(
                chunks=chunks,
                embedder=embedder,
                collection_name=collection_name,
            )
        else:
            collection_name = indexer.collection_name
            indexed_count = indexer.index_chunks(
                chunks=chunks,
                embedder=embedder,
                recreate_collection=args.recreate,
            )

        info = indexer.get_collection_info(collection_name=collection_name)
        print("\n" + "=" * 70)
        print(f"✅ ĐÃ ĐẨY THÀNH CÔNG {indexed_count} CHUNKS LÊN QDRANT!")
        print(f"📌 Collection : {info.get('name')}")
        print(f"📌 Số vectors  : {info.get('vectors_count')}")
        print(f"📌 Dimension   : {info.get('vector_size')}")
        print(f"📌 Dense names : {info.get('vector_names')}")
        print(f"📌 Sparse names: {info.get('sparse_vector_names')}")
        print(f"📌 Metric      : {info.get('distance')}")
        print(f"🌐 Xem dashboard tại: http://localhost:6333/dashboard")
        print("=" * 70)


if __name__ == "__main__":
    main()
