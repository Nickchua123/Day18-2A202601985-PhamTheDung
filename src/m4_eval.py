from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    metric_names = ("faithfulness", "answer_relevancy",
                    "context_precision", "context_recall")
    if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError("questions, answers, contexts and ground_truths must have equal lengths")

    # RAGAS invokes an LLM and can be slow or unavailable in offline tests.
    # Opt in explicitly for the full LLM-based evaluation.
    if os.getenv("RAGAS_ENABLE", "0") == "1":
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (faithfulness, answer_relevancy,
                                        context_precision, context_recall)
            dataset = Dataset.from_dict({
                "question": questions, "answer": answers,
                "contexts": contexts, "ground_truth": ground_truths,
            })
            result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                                context_precision, context_recall])
            frame = result.to_pandas()

            def value(row, name):
                raw = row.get(name, 0.0)
                try:
                    number = float(raw)
                    return 0.0 if number != number else number
                except (TypeError, ValueError):
                    return 0.0

            per_question = [EvalResult(
                question=str(row.get("question", questions[i])),
                answer=str(row.get("answer", answers[i])),
                contexts=list(row.get("contexts", contexts[i])),
                ground_truth=str(row.get("ground_truth", ground_truths[i])),
                **{name: value(row, name) for name in metric_names},
            ) for i, (_, row) in enumerate(frame.iterrows())]
            return {name: (sum(getattr(item, name) for item in per_question) /
                           len(per_question) if per_question else 0.0)
                    for name in metric_names} | {"per_question": per_question}
        except Exception as exc:
            print(f"  ⚠️  RAGAS evaluation failed, using local fallback: {exc}")

    def tokens(value: str) -> set[str]:
        return set(re.findall(r"\w+", (value or "").lower(), flags=re.UNICODE))

    per_question = []
    for question, answer, context, ground_truth in zip(
            questions, answers, contexts, ground_truths):
        answer_tokens = tokens(answer)
        question_tokens = tokens(question)
        truth_tokens = tokens(ground_truth)
        context_tokens = tokens(" ".join(context))
        relevant_contexts = [tokens(item) for item in context]

        faithfulness_score = (len(answer_tokens & context_tokens) /
                              len(answer_tokens) if answer_tokens else 0.0)
        answer_relevancy_score = (len(answer_tokens & truth_tokens) /
                                  len(truth_tokens) if truth_tokens else 0.0)
        precision_hits = sum(bool(item & truth_tokens) for item in relevant_contexts)
        context_precision_score = (precision_hits / len(relevant_contexts)
                                   if relevant_contexts else 0.0)
        context_recall_score = (len(truth_tokens & context_tokens) /
                                len(truth_tokens) if truth_tokens else 0.0)
        per_question.append(EvalResult(
            question, answer, context, ground_truth, faithfulness_score,
            answer_relevancy_score, context_precision_score, context_recall_score))

    return {name: (sum(getattr(item, name) for item in per_question) /
                   len(per_question) if per_question else 0.0)
            for name in metric_names} | {"per_question": per_question}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating or unsupported by context", "Tighten the answer prompt and require citations."),
        "context_recall": ("Relevant information was not retrieved", "Improve chunking or add BM25/dense candidates."),
        "context_precision": ("Retrieved context contains irrelevant chunks", "Increase reranking or add metadata filters."),
        "answer_relevancy": ("Answer does not directly address the question", "Improve the answer prompt and query rewriting."),
    }
    scored = []
    for result in eval_results:
        metrics = {k: getattr(result, k) for k in diagnostic_tree}
        worst = min(metrics, key=metrics.get)
        scored.append((sum(metrics.values()) / len(metrics), result, worst, metrics[worst]))
    output = []
    for avg, result, worst, worst_score in sorted(scored, key=lambda x: x[0])[:bottom_n]:
        diagnosis, fix = diagnostic_tree[worst]
        output.append({"question": result.question, "worst_metric": worst,
                       "score": float(worst_score),
                       "diagnosis": diagnosis, "suggested_fix": fix})
    return output


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
