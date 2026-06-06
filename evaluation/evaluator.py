import time
import re
from statistics import mean

from langchain_openai import ChatOpenAI

from app.llm.rag_chain import get_rag_chain
from app.retrieval.hybrid_retriever import (
    HybridRetriever
)

from evaluation.eval_data import eval_data

from evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate,
    reciprocal_rank,
    average_precision,
    ndcg
)


rag_chain = get_rag_chain()

retriever = HybridRetriever(
    dense_k=10,
    sparse_k=10,
    rerank_top_k=5
)

judge_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


def judge(
    query,
    ground_truth,
    answer
):

    try:

        prompt = f"""
You are an evaluator.

Question:
{query}

Ground Truth:
{ground_truth}

Model Answer:
{answer}

Score from 1 to 5.

1 = completely wrong
5 = fully correct

Return ONLY a number.
"""

        response = judge_llm.invoke(
            prompt
        )

        text = str(response)

        match = re.search(
            r"\d+(\.\d+)?",
            text
        )

        if match:
            return float(
                match.group()
            )

        return 0.0

    except Exception:
        return 0.0


def run_evaluation():

    results = []

    for item in eval_data:

        query = item["question"]

        ground_truth = item[
            "ground_truth"
        ]

        keywords = item[
            "expected_doc_keywords"
        ]

        print(
            f"\nEvaluating: {query}"
        )

        docs = retriever.invoke(
            query
        )

        precision = precision_at_k(
            docs,
            keywords,
            k=5
        )

        recall = recall_at_k(
            docs,
            keywords,
            k=5
        )

        hit = hit_rate(
            docs,
            keywords
        )

        mrr = reciprocal_rank(
            docs,
            keywords
        )

        ap = average_precision(
            docs,
            keywords,
            k=5
        )

        ndcg_score = ndcg(
            docs,
            keywords,
            k=5
        )

        start = time.time()

        answer = rag_chain.invoke(
            query
        )

        latency = (
            time.time() - start
        )

        score = judge(
            query,
            ground_truth,
            answer
        )

        results.append(
            {
                "query": query,
                "precision": precision,
                "recall": recall,
                "hit_rate": hit,
                "mrr": mrr,
                "map": ap,
                "ndcg": ndcg_score,
                "answer_score": score,
                "latency": latency
            }
        )

    return results


def summarize(results):

    print("\n===== FINAL REPORT =====")

    print(
        f"Precision@5: "
        f"{mean([r['precision'] for r in results]):.4f}"
    )

    print(
        f"Recall@5: "
        f"{mean([r['recall'] for r in results]):.4f}"
    )

    print(
        f"Hit Rate: "
        f"{mean([r['hit_rate'] for r in results]):.4f}"
    )

    print(
        f"MRR: "
        f"{mean([r['mrr'] for r in results]):.4f}"
    )

    print(
        f"MAP: "
        f"{mean([r['map'] for r in results]):.4f}"
    )

    print(
        f"NDCG: "
        f"{mean([r['ndcg'] for r in results]):.4f}"
    )

    print(
        f"Answer Score: "
        f"{mean([r['answer_score'] for r in results]):.2f}/5"
    )

    print(
        f"Latency: "
        f"{mean([r['latency'] for r in results]):.2f}s"
    )


if __name__ == "__main__":

    results = run_evaluation()

    summarize(results)