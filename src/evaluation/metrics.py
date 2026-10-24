import re
import string
import numpy as np
import unicodedata
from typing import Dict, List, Optional, Set, Tuple
from difflib import SequenceMatcher
from collections import Counter


# ------------------------------------------------------------------------------
# 0. Normalization
# ------------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Chuẩn hóa: NFC, lowercase, bỏ dấu câu, gộp whitespace."""
    if not text:
        return ""
    # Unicode NFC để tránh mismatch dấu tiếng Việt
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"[“”‘’«»–—…]", " ", text)  # bỏ thêm ký tự đặc biệt
    return " ".join(text.split())


def strip_page_prefix(text: str) -> str:
    """FIX #3: Bỏ prefix 'Page N:' khỏi gold context trước khi so khớp."""
    return re.sub(r"^\s*page\s+\d+\s*[:\-]\s*", "", text, flags=re.IGNORECASE)


def get_tokens(text: str) -> List[str]:
    norm = normalize_text(text)
    return norm.split() if norm else []


def extract_numbers(text: str) -> Set[str]:
    return set(re.findall(r"\b\d+(?:[.,]\d+)?\b", text))


# ------------------------------------------------------------------------------
# 1. Generation Metrics
# ------------------------------------------------------------------------------

def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def compute_token_f1(prediction: str, ground_truth: str) -> Dict[str, float]:
    pred_tokens = get_tokens(prediction)
    gold_tokens = get_tokens(ground_truth)
    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    common = set(pred_counter) & set(gold_counter)
    num_same = sum(min(pred_counter[t], gold_counter[t]) for t in common)
    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


REJECTION_KEYWORDS = [
    "không tìm thấy", "không có thông tin", "không đề cập",
    "không được nhắc đến", "không nêu", "tài liệu không chứa",
    "tài liệu không cung cấp", "không thể trả lời", "chưa có thông tin",
    "không thấy thông tin", "không nằm trong","Tài liệu không ghi"
]


def check_hallucination_rejection(prediction: str) -> bool:
    norm = normalize_text(prediction)
    return any(normalize_text(kw) in norm for kw in REJECTION_KEYWORDS)


# ------------------------------------------------------------------------------
# 2. Retrieval Metrics 
# ------------------------------------------------------------------------------

def _answer_in_chunk(ground_truth: str, chunk_norm: str) -> bool:
    """ Check câu trả lời có trong chunk (chuẩn hóa, bỏ dấu câu)."""
    if not ground_truth or not chunk_norm:
        return False

    gt_norm = normalize_text(ground_truth)
    if not gt_norm:
        return False

    # Substring sau normalize (đã bỏ dấu câu)
    if gt_norm in chunk_norm:
        return True

    # Fallback cho câu trả lời rất ngắn ("3", "4")— check theo token
    gt_tokens = gt_norm.split()
    if len(gt_tokens) <= 3:
        chunk_tokens = set(chunk_norm.split())
        # Nếu ≥ 80% token của answer xuất hiện trong chunk
        if sum(1 for t in gt_tokens if t in chunk_tokens) / len(gt_tokens) >= 0.8:
            return True

    # Fallback cho câu trả lời nhiều phần ("K-Means với K=3")
    parts = [p.strip() for p in re.split(r"[;,]| với | and ", gt_norm) if len(p.strip()) > 2]
    if parts and any(p in chunk_norm for p in parts):
        return True

    return False


def _gold_context_matches_chunk(gold: str, chunk_norm: str, threshold: float = 0.5):
    """Bỏ fallback precision_ratio để tránh match lỏng với gold ngắn."""
    gold_clean = strip_page_prefix(gold)
    gold_norm = normalize_text(gold_clean)
    if not gold_norm or not chunk_norm:
        return False

    # Chỉ dùng substring check (2 chiều)
    if gold_norm in chunk_norm or chunk_norm in gold_norm:
        return True

    # Word overlap — chỉ dùng recall, không dùng max
    gold_words = set(gold_norm.split())
    if len(gold_words) < 5:  # gold quá ngắn → dễ false positive
        return False

    chunk_words = set(chunk_norm.split())
    overlap = gold_words & chunk_words
    recall_ratio = len(overlap) / len(gold_words)
    return recall_ratio >= threshold  # 0.5, không dùng precision


def is_chunk_relevant(
    retrieved_text: str,
    gold_contexts: List[str],
    ground_truth: str = "",
    threshold: float = 0.5,
) -> bool:
    """ Check chunk có liên quan không (theo ground_truth và gold_contexts)."""
    chunk_norm = normalize_text(retrieved_text)
    if not chunk_norm:
        return False

    # Ưu tiên 1: câu trả lời có trong chunk không?
    if ground_truth and _answer_in_chunk(ground_truth, chunk_norm):
        return True

    # Ưu tiên 2: chunk có khớp với gold context không?
    for gold in gold_contexts or []:
        if _gold_context_matches_chunk(gold, chunk_norm, threshold):
            return True

    return False


def compute_retrieval_metrics(
    retrieved_texts: List[str],
    gold_contexts: List[str],
    ground_truth: str = "",
    k_list: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Tính Hit@K, MRR, Context Recall. Đã fix để nhận ground_truth."""
    if k_list is None:
        k_list = [1, 3, 5]

    metrics: Dict[str, float] = {}

    if not gold_contexts and not ground_truth:
        for k in k_list:
            metrics[f"hit_rate@{k}"] = 0.0
        metrics["mrr"] = 0.0
        metrics["context_recall"] = 0.0
        return metrics

    relevant_ranks: List[int] = []
    found_gold_indices: Set[int] = set()

    for rank, chunk_text in enumerate(retrieved_texts, start=1):
        chunk_norm = normalize_text(chunk_text)

        # 1. Check theo ground_truth answer
        if ground_truth and _answer_in_chunk(ground_truth, chunk_norm):
            relevant_ranks.append(rank)

        # 2. Check theo gold contexts
        for gold_idx, gold in enumerate(gold_contexts or []):
            if _gold_context_matches_chunk(gold, chunk_norm):
                found_gold_indices.add(gold_idx)
                if rank not in relevant_ranks:
                    relevant_ranks.append(rank)

    # Hit Rate @ K
    for k in k_list:
        metrics[f"hit_rate@{k}"] = 1.0 if any(r <= k for r in relevant_ranks) else 0.0

    # MRR
    metrics["mrr"] = 1.0 / min(relevant_ranks) if relevant_ranks else 0.0

    # Context Recall — cải tiến: dùng max(answer_found, gold_ratio)
    gold_ratio = len(found_gold_indices) / len(gold_contexts) if gold_contexts else 0.0
    answer_found = 1.0 if (ground_truth and any(
        _answer_in_chunk(ground_truth, normalize_text(c)) for c in retrieved_texts
    )) else 0.0
    metrics["context_recall"] = max(gold_ratio, answer_found) if ground_truth else gold_ratio

    return metrics


# ------------------------------------------------------------------------------
# 3. Semantic Similarity 
# ------------------------------------------------------------------------------

def compute_semantic_similarity(
    prediction: str,
    ground_truth: str,
    embedder: Optional[object] = None,
) -> float:
    if not prediction or not ground_truth:
        return 0.0

    if embedder is None:
        t1 = set(get_tokens(prediction))
        t2 = set(get_tokens(ground_truth))
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / len(t1 | t2)

    
    try:
        if hasattr(embedder, "embed_query"):
            vec1 = np.array(embedder.embed_query(prediction.strip()), dtype=np.float32)
            vec2 = np.array(embedder.embed_query(ground_truth.strip()), dtype=np.float32)
        elif hasattr(embedder, "encode"):
            vec1 = np.array(embedder.encode(prediction.strip(), normalize_embeddings=True), dtype=np.float32)
            vec2 = np.array(embedder.encode(ground_truth.strip(), normalize_embeddings=True), dtype=np.float32)
        else:
            return 0.0

        n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return max(0.0, min(1.0, float(np.dot(vec1, vec2) / (n1 * n2))))
    except Exception:
        return 0.0