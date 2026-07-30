import sys
import json
import csv
import math
import re
import logging
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.pipeline.graph import build_rag_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def tokenize(text: Any) -> List[str]:
    """Tokenizes text into lowercase word tokens, safely handling str or list of content blocks."""
    if isinstance(text, list):
        parts = []
        for item in text:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        text = " ".join(parts)
    elif not isinstance(text, str):
        text = str(text) if text is not None else ""

    return re.findall(r'\w+', text.lower())


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extracts n-grams from a list of tokens."""
    if len(tokens) < n:
        return Counter()
    return Counter([tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)])


def compute_rouge_n(candidate_tokens: List[str], reference_tokens: List[str], n: int) -> Dict[str, float]:
    """Computes ROUGE-N Precision, Recall, and F1-score."""
    cand_ngrams = get_ngrams(candidate_tokens, n)
    ref_ngrams = get_ngrams(reference_tokens, n)

    if not ref_ngrams or not cand_ngrams:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = sum((cand_ngrams & ref_ngrams).values())
    total_ref = sum(ref_ngrams.values())
    total_cand = sum(cand_ngrams.values())

    recall = overlap / total_ref if total_ref > 0 else 0.0
    precision = overlap / total_cand if total_cand > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def compute_lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Computes the length of Longest Common Subsequence (LCS) between two sequences."""
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge_l(candidate_tokens: List[str], reference_tokens: List[str]) -> Dict[str, float]:
    """Computes ROUGE-L Precision, Recall, and F1 based on LCS."""
    if not candidate_tokens or not reference_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs_len = compute_lcs_length(candidate_tokens, reference_tokens)
    recall = lcs_len / len(reference_tokens) if len(reference_tokens) > 0 else 0.0
    precision = lcs_len / len(candidate_tokens) if len(candidate_tokens) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def compute_bleu(candidate_tokens: List[str], reference_tokens: List[str], max_order: int = 4) -> float:
    """Computes sentence-level BLEU score with brevity penalty."""
    if not candidate_tokens or not reference_tokens:
        return 0.0

    precisions = []
    for i in range(1, max_order + 1):
        cand_ngrams = get_ngrams(candidate_tokens, i)
        ref_ngrams = get_ngrams(reference_tokens, i)
        total = sum(cand_ngrams.values())
        if total == 0:
            precisions.append(0.0)
            continue
        overlap = sum((cand_ngrams & ref_ngrams).values())
        precisions.append(overlap / total)

    valid_p = [p for p in precisions if p > 0]
    if not valid_p:
        return 0.0

    p_val = math.exp(sum(math.log(p) for p in valid_p) / max_order)

    # Brevity Penalty
    c = len(candidate_tokens)
    r = len(reference_tokens)
    bp = 1.0 if c > r else math.exp(1 - r / c) if c > 0 else 0.0
    return round(bp * p_val, 4)


def load_eval_dataset() -> List[Dict[str, Any]]:
    """Loads evaluation dataset from CSV or JSON file."""
    dataset = []
    if config.EVAL_DATASET_JSON.exists():
        logger.info(f"Loading evaluation dataset from JSON: {config.EVAL_DATASET_JSON}")
        with open(config.EVAL_DATASET_JSON, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    elif config.EVAL_DATASET_CSV.exists():
        logger.info(f"Loading evaluation dataset from CSV: {config.EVAL_DATASET_CSV}")
        with open(config.EVAL_DATASET_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset.append({
                    "id": row.get("#"),
                    "question": row.get("Question"),
                    "ground_truth": row.get("Correct Answer (Summary)"),
                    "reference": row.get("Page & Paragraph Reference")
                })
    else:
        raise FileNotFoundError("Evaluation dataset not found in CSV or JSON format.")
    return dataset


def compute_bert_score(candidate: str, reference: str, embed_model: Any = None) -> Dict[str, float]:
    """
    Computes BERTScore (Precision, Recall, F1) using BGE-M3 embedding cosine similarity
    leveraging the already-cached safetensors model (requires 0 extra disk space).
    """
    try:
        if not candidate or not reference:
            return {"bert_score_precision": 0.0, "bert_score_recall": 0.0, "bert_score_f1": 0.0}

        if embed_model is None:
            from src.ingestion.indexer import initialize_embeddings
            embed_model = initialize_embeddings()

        vec1 = embed_model.get_text_embedding(candidate)
        vec2 = embed_model.get_text_embedding(reference)

        import numpy as np
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 > 0 and norm2 > 0:
            sim = float(np.dot(v1, v2) / (norm1 * norm2))
        else:
            sim = 0.0
        sim = max(0.0, min(1.0, sim))
        return {
            "bert_score_precision": round(sim, 4),
            "bert_score_recall": round(sim, 4),
            "bert_score_f1": round(sim, 4)
        }
    except Exception as e:
        logger.warning(f"BERTScore calculation notice: {e}")
        return {"bert_score_precision": 0.0, "bert_score_recall": 0.0, "bert_score_f1": 0.0}


def evaluate_answer_vs_ground_truth(generated_answer: str, ground_truth: str, embed_model: Any = None) -> Dict[str, float]:
    """
    Computes quantitative metrics comparing Generated Answer against Dataset Ground Truth Answer:
    - ROUGE-1 (Precision, Recall, F1)
    - ROUGE-2 (Precision, Recall, F1)
    - ROUGE-L (Precision, Recall, F1)
    - BLEU-4
    - BERTScore (Precision, Recall, F1)
    """
    ans_tokens = tokenize(generated_answer)
    gt_tokens = tokenize(ground_truth)

    r1 = compute_rouge_n(ans_tokens, gt_tokens, 1)
    r2 = compute_rouge_n(ans_tokens, gt_tokens, 2)
    rl = compute_rouge_l(ans_tokens, gt_tokens)
    bleu = compute_bleu(ans_tokens, gt_tokens, max_order=4)
    bert = compute_bert_score(generated_answer, ground_truth, embed_model=embed_model)

    return {
        "rouge1_f1": r1["f1"],
        "rouge1_precision": r1["precision"],
        "rouge1_recall": r1["recall"],
        "rouge2_f1": r2["f1"],
        "rouge2_precision": r2["precision"],
        "rouge2_recall": r2["recall"],
        "rougel_f1": rl["f1"],
        "rougel_precision": rl["precision"],
        "rougel_recall": rl["recall"],
        "bleu_score": bleu,
        "bert_score_precision": bert["bert_score_precision"],
        "bert_score_recall": bert["bert_score_recall"],
        "bert_score_f1": bert["bert_score_f1"]
    }


def main():
    print("=" * 60)
    print("Step 3: Quantitative RAG System Evaluation")
    print("(100% Backend Aligned: ROUGE, BLEU & BERTScore)")
    print("=" * 60)

    dataset = load_eval_dataset()
    logger.info(f"Loaded {len(dataset)} evaluation questions.")

    # Locate default LightRAG.pdf Collection ID (100% aligned with backend upload hash)
    pdf_path = config.RAW_DATA_DIR / "2410.05779v3.pdf"
    if not pdf_path.exists():
        pdf_path = config.RAW_DATA_DIR / "LightRAG.pdf"

    doc_id = config.COLLECTION_NAME
    if pdf_path.exists():
        import hashlib
        with open(pdf_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:8]
            doc_id = f"doc_{file_hash}"
    logger.info(f"Evaluating against ChromaDB Collection ID: '{doc_id}'")

    # Import backend map_reduce summary runner for 100% logic alignment
    try:
        from UI.backend.main import handle_map_reduce_summary
    except Exception:
        handle_map_reduce_summary = None

    from src.ingestion.indexer import get_retriever, initialize_embeddings
    from src.llm_factory import get_llm
    
    embed_model = initialize_embeddings()
    retriever = get_retriever(top_k=config.TOP_K, collection_name=doc_id)
    llm = get_llm()
    rag_graph = build_rag_graph(retriever_instance=retriever, llm_instance=llm)

    results = []
    totals = {
        "rouge1_f1": 0.0, "rouge1_precision": 0.0, "rouge1_recall": 0.0,
        "rouge2_f1": 0.0, "rouge2_precision": 0.0, "rouge2_recall": 0.0,
        "rougel_f1": 0.0, "rougel_precision": 0.0, "rougel_recall": 0.0,
        "bleu_score": 0.0,
        "bert_score_precision": 0.0, "bert_score_recall": 0.0, "bert_score_f1": 0.0
    }

    for idx, item in enumerate(dataset):
        q_id = item.get("id", idx + 1)
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")

        logger.info(f"Evaluating Question [{q_id}/{len(dataset)}]: '{question[:50]}...'")

        q_lower = question.lower().strip()
        # 100% Backend Aligned execution logic
        if any(k in q_lower for k in ["summary", "summarize", "摘要", "總結"]) and handle_map_reduce_summary:
            res_dict = handle_map_reduce_summary(doc_id)
            ans_raw = res_dict.get("answer", "")
            if isinstance(ans_raw, list):
                generated_answer = "".join(item.get("text", "") for item in ans_raw if isinstance(item, dict))
            else:
                generated_answer = str(ans_raw)
            token_usage = res_dict.get("token_usage", {})
        else:
            state = rag_graph.invoke({"query": question, "context": [], "answer": "", "token_usage": {}})
            generated_answer = state.get("answer", "")
            token_usage = state.get("token_usage", {})

        # Compute quantitative metrics against Dataset Ground Truth Answer (0 extra disk space needed!)
        metrics = evaluate_answer_vs_ground_truth(generated_answer, ground_truth, embed_model=embed_model)

        for k, v in metrics.items():
            totals[k] += v

        result_row = {
            "id": q_id,
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "input_tokens": token_usage.get("input_tokens", 0),
            "output_tokens": token_usage.get("output_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0),
            "rouge1_f1": metrics["rouge1_f1"],
            "rouge1_precision": metrics["rouge1_precision"],
            "rouge1_recall": metrics["rouge1_recall"],
            "rouge2_f1": metrics["rouge2_f1"],
            "rouge2_precision": metrics["rouge2_precision"],
            "rouge2_recall": metrics["rouge2_recall"],
            "rougel_f1": metrics["rougel_f1"],
            "rougel_precision": metrics["rougel_precision"],
            "rougel_recall": metrics["rougel_recall"],
            "bleu_score": metrics["bleu_score"],
            "bert_score_f1": metrics["bert_score_f1"],
            "bert_score_precision": metrics["bert_score_precision"],
            "bert_score_recall": metrics["bert_score_recall"]
        }
        results.append(result_row)

    num_items = len(dataset)
    means = {k: round(v / num_items, 4) for k, v in totals.items()}

    # Save to CSV and JSON report
    report_csv = config.EVAL_REPORTS_DIR / "eval_report.csv"
    report_json = config.EVAL_REPORTS_DIR / "eval_report.json"

    # CSV output
    fieldnames = [
        "id", "question", "ground_truth", "generated_answer",
        "input_tokens", "output_tokens", "total_tokens",
        "rouge1_f1", "rouge1_precision", "rouge1_recall",
        "rouge2_f1", "rouge2_precision", "rouge2_recall",
        "rougel_f1", "rougel_precision", "rougel_recall",
        "bleu_score", "bert_score_f1", "bert_score_precision", "bert_score_recall"
    ]
    with open(report_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        # Summary row
        summary_row = {
            "id": "AVERAGE",
            "question": "SUMMARY AVERAGE SCORES",
            "ground_truth": "-",
            "generated_answer": "-",
            "input_tokens": "-",
            "output_tokens": "-",
            "total_tokens": "-"
        }
        summary_row.update(means)
        writer.writerow(summary_row)

    # JSON output
    report_data = {
        "summary_averages": means,
        "itemized_results": results
    }
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("📊 Evaluation Summary Report (100% Backend Aligned)")
    print("Target: RAG Generated Answer vs. Dataset Ground Truth Answer")
    print("=" * 60)
    print(f"Total Questions Evaluated: {num_items}")
    print(f"\n--- [ROUGE Metrics] ---")
    print(f"  • ROUGE-1 F1 Score : {means['rouge1_f1']:.4f}  (Precision: {means['rouge1_precision']:.4f}, Recall: {means['rouge1_recall']:.4f})")
    print(f"  • ROUGE-2 F1 Score : {means['rouge2_f1']:.4f}  (Precision: {means['rouge2_precision']:.4f}, Recall: {means['rouge2_recall']:.4f})")
    print(f"  • ROUGE-L F1 Score : {means['rougel_f1']:.4f}  (Precision: {means['rougel_precision']:.4f}, Recall: {means['rougel_recall']:.4f})")
    print(f"\n--- [BLEU Metric] ---")
    print(f"  • BLEU Score       : {means['bleu_score']:.4f}")
    print(f"\n--- [BERTScore Metric] ---")
    print(f"  • BERTScore F1     : {means['bert_score_f1']:.4f}  (Precision: {means['bert_score_precision']:.4f}, Recall: {means['bert_score_recall']:.4f})")
    print(f"\nFull CSV Report : {report_csv}")
    print(f"Full JSON Report: {report_json}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
