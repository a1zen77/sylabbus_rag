import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from retriever import retrieve_context_reranked

def load_eval_dataset(path: str = "eval_dataset.json") -> list[dict]:
    eval_path = os.path.join(os.path.dirname(__file__), path)
    with open(eval_path, 'r') as f:
        return json.load(f)

def evaluate_single_retrieval(item: dict, top_k: int = 5, candidate_pool: int = 15) -> dict:
    question = item["question"]
    expected_filename = item.get("expected_filename")
    expected_page = item.get("expected_page")

    if expected_filename is None:
        return {"id": item["id"], "question": question, "skip_retrieval_eval": True}

    results = retrieve_context_reranked(question, top_k=top_k, candidate_pool=candidate_pool)

    rank = None
    for i, result in enumerate(results, start=1):
        meta = result["metadata"]
        if meta["filename"] == expected_filename and meta.get("page_number") == expected_page:
            rank = i
            break

    hit = rank is not None

    return {
        "id": item["id"],
        "question": question,
        "hit": hit,
        "rank": rank,
        "reciprocal_rank": (1 / rank) if hit else 0
    }

def run_retrieval_eval(top_k: int = 5, candidate_pool: int = 15) -> dict:
    dataset = load_eval_dataset()
    results = [evaluate_single_retrieval(item, top_k=top_k, candidate_pool=candidate_pool) for item in dataset]
    scored_results = [r for r in results if not r.get("skip_retrieval_eval")]

    total = len(scored_results)
    hits = sum(1 for r in scored_results if r["hit"])
    recall_at_k = hits / total if total > 0 else 0
    mrr = sum(r["reciprocal_rank"] for r in scored_results) / total if total > 0 else 0

    return {
        "top_k": top_k,
        "total_questions": total,
        "hits": hits,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "per_question_results": results
    }

def print_report(eval_results: dict):
    print("=" * 60)
    print(f"RERANKED RETRIEVAL EVALUATION REPORT (top_k={eval_results['top_k']})")
    print("=" * 60)
    print(f"Recall@{eval_results['top_k']}: {eval_results['recall_at_k']:.2%}")
    print(f"MRR: {eval_results['mrr']:.3f}")
    print()
    for r in eval_results['per_question_results']:
        if r.get("skip_retrieval_eval"):
            continue
        status = "✅ HIT" if r["hit"] else "❌ MISS"
        rank_str = f"rank={r['rank']}" if r["hit"] else "not found"
        print(f"[{status}] {r['question'][:50]}... ({rank_str})")

if __name__ == "__main__":
    results = run_retrieval_eval(top_k=5, candidate_pool=15)
    print_report(results)