import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from src.logger import get_logger

logger = get_logger("eval_dataset")


@dataclass
class EvalSample:
    """Đại diện cho 1 mẫu kiểm thử trong bộ dữ liệu đánh giá RAG."""

    id: str
    query: str
    ground_truth: str
    ground_truth_contexts: List[str] = field(default_factory=list)
    question_type: str = "direct"  # direct | reasoning | unanswerable | ...
    is_answerable: bool = True
    source_doc: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvaluationDataset:
    """Quản lý nạp, lọc và xuất tập dữ liệu đánh giá."""

    def __init__(self, samples: Optional[List[EvalSample]] = None):
        self.samples: List[EvalSample] = samples or []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> EvalSample:
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)

    @classmethod
    def load_jsonl(cls, file_path: Union[str, Path]) -> "EvaluationDataset":
        """Nạp dữ liệu từ file JSONL chuẩn."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file dataset: {path}")

        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    sample = EvalSample(
                        id=str(data.get("id", f"q_{line_idx:03d}")),
                        query=data.get("query", data.get("question", "")).strip(),
                        ground_truth=data.get("ground_truth", data.get("answer", "")).strip(),
                        ground_truth_contexts=data.get(
                            "ground_truth_contexts",
                            data.get("contexts", data.get("gold_contexts", [])),
                        ),
                        question_type=data.get("question_type", "direct"),
                        is_answerable=bool(data.get("is_answerable", True)),
                        source_doc=data.get("source_doc"),
                        metadata={k: v for k, v in data.items() if k not in [
                            "id", "query", "question", "ground_truth", "answer",
                            "ground_truth_contexts", "contexts", "gold_contexts",
                            "question_type", "is_answerable", "source_doc"
                        ]},
                    )
                    samples.append(sample)
                except Exception as e:
                    logger.warning(f"Lỗi khi đọc dòng {line_idx} trong {path}: {e}")

        logger.info(f"Đã nạp thành công {len(samples)} mẫu đánh giá từ '{path.name}'")
        return cls(samples)

    def to_pandas(self) -> pd.DataFrame:
        """Chuyển đổi dataset sang Pandas DataFrame."""
        return pd.DataFrame([s.to_dict() for s in self.samples])

    def to_ragas_dataset(self):
        """Chuyển đổi sang định dạng HuggingFace Dataset dùng cho thư viện Ragas (nếu cần)."""
        from datasets import Dataset

        records = {
            "user_input": [s.query for s in self.samples],
            "reference": [s.ground_truth for s in self.samples],
            "reference_contexts": [s.ground_truth_contexts for s in self.samples],
        }
        return Dataset.from_dict(records)

