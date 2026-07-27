import json
import os
import sys

# Add parent directory to path so we can import retriever
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from retriever import retrieve_context

def load_eval_dataset(path: str = "eval_dataset.json") -> list[dict]:
    """Load the evaluation dataset."""
    eval_path = os.path.join(os.path.dirname(__file__), path)
    with open(eval_path, 'r') as f:
        return json.load(f)

def evaluate_single_retrieval(item: dict, top_k: int = 5) -> dict:
    """
    Evaluate retrieval for a single question.
    Returns hit/miss info and rank of correct chunk.
    """
    question = item["question"]
    expected_filename = item.get("expected_filename")
    expected_page = item.get("expected_page")
    
    # Skip retrieval scoring for "should refuse" questions
    if expected_filename is None:
        return {
            "id": item["id"],
            "question": question,
            "skip_retrieval_eval": True
        }
    
    # Run retrieval
    results = retrieve_context(question, top_k=top_k)
    
    # Find rank of the first chunk matching expected filename + page
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
        "expected_filename": expected_filename,
        "expected_page": expected_page,
        "hit": hit,
        "rank": rank,  # None if not found in top_k
        "reciprocal_rank": (1 / rank) if hit else 0,
        "top_result_distance": results[0]["distance"] if results else None
    }

def run_retrieval_eval(top_k: int = 5) -> dict:
    """Run retrieval evaluation across the whole dataset."""
    dataset = load_eval_dataset()
    
    results = []
    for item in dataset:
        result = evaluate_single_retrieval(item, top_k=top_k)
        results.append(result)
    
    # Filter to only questions we scored (exclude "should refuse" ones)
    scored_results = [r for r in results if not r.get("skip_retrieval_eval")]
    
    # Compute aggregate metrics
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
    """Pretty-print the evaluation results."""
    print("=" * 60)
    print(f"RETRIEVAL EVALUATION REPORT (top_k={eval_results['top_k']})")
    print("=" * 60)
    print(f"Total questions evaluated: {eval_results['total_questions']}")
    print(f"Hits: {eval_results['hits']}/{eval_results['total_questions']}")
    print(f"Recall@{eval_results['top_k']}: {eval_results['recall_at_k']:.2%}")
    print(f"MRR: {eval_results['mrr']:.3f}")
    print()
    print("Per-question breakdown:")
    print("-" * 60)
    
    for r in eval_results['per_question_results']:
        if r.get("skip_retrieval_eval"):
            print(f"[SKIP] {r['question'][:50]}... (no-answer question)")
            continue
        
        status = "✅ HIT" if r["hit"] else "❌ MISS"
        rank_str = f"rank={r['rank']}" if r["hit"] else "not found"
        print(f"[{status}] {r['question'][:50]}... ({rank_str})")

if __name__ == "__main__":
    results = run_retrieval_eval(top_k=5)
    print_report(results)