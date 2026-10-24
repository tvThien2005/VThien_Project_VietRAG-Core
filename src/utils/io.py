import json
from pathlib import Path
from typing import Any, Dict, List, Union


def save_jsonl(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """Lưu danh sách dict vào file JSONL."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Đọc dữ liệu từ file JSONL."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file JSONL: {path}")
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """Lưu dữ liệu vào file JSON."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(file_path: Union[str, Path]) -> Any:
    """Đọc dữ liệu từ file JSON."""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
