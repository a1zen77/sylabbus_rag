import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from retriever import retrieve_context, check_confidence
from generator import generate_answer

def load_eval_dataset(path: str = "eval_dataset.json") -> list[dict]:
    """Load the evaluation dataset."""
    eval_path = os.path.join(os.path.dirname(__file__), path)
    with open(eval_path, 'r') as f:
        return json.load(f)

def check_answer_contains_keywords(answer: str, expected_keywords: list[str]) -> dict:
    """
    Check how many of the expected keywords appear in the answer.
    Case-insensitive substring match.
    """
    answer_lower = answer.lower()
    matched = []
    missing = []
    
    for keyword in expected_keywords:
        if keyword.lower() in answer_lower:
            matched.append(keyword)
        else:
            missing.append(keyword)
    
    return {
        "matched": matched,
        "missing": missing,
        "match_ratio": len(matched) / len(expected_keywords) if expected_keywords else 0
    }

def evaluate_single_generation(item: dict, top_k: int = 5, threshold: float = 0.5) -> dict:
    """Evaluate generation quality for a single question."""
    question = item["question"]
    expected_keywords = item["expected_answer_contains"]
    is_out_of_scope = item.get("expected_filename") is None
    
    # Run full pipeline: retrieve -> check confidence -> generate
    contexts = retrieve_context(question, top_k=top_k)
    is_confident = check_confidence(contexts, threshold)
    
    if is_confident:
        answer = generate_answer(question, contexts)
    else:
        answer = "I don't have enough information in the provided syllabus to answer this question confidently."
    
    keyword_check = check_answer_contains_keywords(answer, expected_keywords)
    
    result = {
        "id": item["id"],
        "question": question,
        "category": item.get("category", "unknown"),
        "answer": answer,
        "is_out_of_scope": is_out_of_scope,
        "system_was_confident": is_confident,
        "keyword_match_ratio": keyword_check["match_ratio"],
        "matched_keywords": keyword_check["matched"],
        "missing_keywords": keyword_check["missing"]
    }
    
    if is_out_of_scope:
        # "Correct" means the FINAL ANSWER indicates refusal — regardless of
        # whether that refusal came from the distance threshold or the LLM's
        # own instruction-following. Check for common refusal phrasing.
        refusal_phrases = [
            "don't have enough information",
            "not enough information",
            "not explicitly mentioned",
            "not mentioned",
            "not stated",
            "not specified",
            "no information",
            "does not contain"
        ]
        answer_lower = answer.lower()
        correctly_refused = any(phrase in answer_lower for phrase in refusal_phrases)
        
        result["retrieval_was_confident"] = is_confident  # keep for diagnostics
        result["correct_refusal"] = correctly_refused
        result["passed"] = correctly_refused
    else:
        # For answerable questions, "correct" means high keyword match AND system was confident
        result["passed"] = is_confident and keyword_check["match_ratio"] >= 0.5
    
    return result

def run_generation_eval(top_k: int = 5, threshold: float = 0.5) -> dict:
    """Run generation evaluation across the whole dataset."""
    dataset = load_eval_dataset()
    
    results = []
    for item in dataset:
        result = evaluate_single_generation(item, top_k=top_k, threshold=threshold)
        results.append(result)
    
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    
    answerable = [r for r in results if not r["is_out_of_scope"]]
    out_of_scope = [r for r in results if r["is_out_of_scope"]]
    
    answerable_pass_rate = (
        sum(1 for r in answerable if r["passed"]) / len(answerable) if answerable else 0
    )
    refusal_accuracy = (
        sum(1 for r in out_of_scope if r["passed"]) / len(out_of_scope) if out_of_scope else 0
    )
    
    avg_keyword_match = sum(r["keyword_match_ratio"] for r in answerable) / len(answerable) if answerable else 0
    
    return {
        "total_questions": total,
        "overall_pass_rate": passed / total if total else 0,
        "answerable_pass_rate": answerable_pass_rate,
        "refusal_accuracy": refusal_accuracy,
        "avg_keyword_match_ratio": avg_keyword_match,
        "per_question_results": results
    }

def print_report(eval_results: dict):
    """Pretty-print the evaluation results."""
    print("=" * 60)
    print("GENERATION / FAITHFULNESS EVALUATION REPORT")
    print("=" * 60)
    print(f"Total questions: {eval_results['total_questions']}")
    print(f"Overall pass rate: {eval_results['overall_pass_rate']:.2%}")
    print(f"Answerable question pass rate: {eval_results['answerable_pass_rate']:.2%}")
    print(f"Refusal accuracy (out-of-scope): {eval_results['refusal_accuracy']:.2%}")
    print(f"Avg keyword match ratio: {eval_results['avg_keyword_match_ratio']:.2%}")
    print()
    print("Per-question breakdown:")
    print("-" * 60)
    
    for r in eval_results['per_question_results']:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        tag = "[OUT-OF-SCOPE]" if r["is_out_of_scope"] else f"[{r['keyword_match_ratio']:.0%} keywords]"
        print(f"[{status}] {tag} {r['question'][:55]}...")
        
        if not r["passed"]:
            if r["is_out_of_scope"]:
                print(f"         → System answered instead of refusing: \"{r['answer'][:80]}...\"")
            else:
                print(f"         → Missing keywords: {r['missing_keywords']}")
                print(f"         → Answer: \"{r['answer'][:100]}...\"")

if __name__ == "__main__":
    results = run_generation_eval(top_k=5, threshold=0.5)
    print_report(results)
    
    # Save detailed results to file for later comparison
    output_path = os.path.join(os.path.dirname(__file__), "last_generation_eval.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")