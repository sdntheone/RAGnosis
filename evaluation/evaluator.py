import time
import re
from statistics import mean

from app.llm.rag_chain import get_rag_chain
from app.retrieval.vector_store import get_retriever
from evaluation.eval_data import eval_data
from langchain_openai import ChatOpenAI


# ===== Initialize once =====
rag_chain = get_rag_chain()
retriever = get_retriever(k=2)

judge_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# ===== Retrieval Evaluation =====
def retrieval_hit_rate(retrieved_docs, expected_keywords):
    try:
        text = " ".join([doc.page_content.lower() for doc in retrieved_docs])
        return any(keyword.lower() in text for keyword in expected_keywords)
    except Exception:
        return False


# ===== LLM-as-a-Judge =====
def judge(query, ground_truth, answer):
    try:
        prompt = f"""
You are an evaluator.

Question: {query}
Ground Truth: {ground_truth}
Model Answer: {answer}

Score the answer from 1 to 5:
1 = incorrect
5 = fully correct and complete

Return ONLY a number (e.g., 4 or 5). No explanation.
"""

        response = judge_llm.invoke(prompt)

        # Convert response to string
        text = str(response)

        # Extract numeric score using regex
        match = re.search(r"\d+(\.\d+)?", text)

        if match:
            return float(match.group())

        return 0.0

    except Exception as e:
        print(f"Judge error: {e}")
        return 0.0


# ===== Main Evaluation =====
def run_evaluation():
    results = []

    for item in eval_data:
        query = item["question"]
        ground_truth = item["ground_truth"]
        keywords = item["expected_doc_keywords"]

        print(f"\n🔍 Evaluating: {query}")

        # ----- Retrieval -----
        retrieved_docs = retriever.invoke(query)
        hit = retrieval_hit_rate(retrieved_docs, keywords)

        # ----- Generation + Latency -----
        start_time = time.time()
        answer = rag_chain.invoke(query)
        latency = time.time() - start_time

        # ----- Answer Quality -----
        score = judge(query, ground_truth, answer)

        results.append({
            "query": query,
            "retrieval_hit": hit,
            "answer_score": score,
            "latency": latency
        })

        print(f"Hit: {hit} | Score: {score} | Latency: {round(latency,2)}s")

    return results


# ===== Summary Metrics =====
def summarize(results):
    hit_rate = mean([1 if r["retrieval_hit"] else 0 for r in results])
    avg_score = mean([r["answer_score"] for r in results])
    avg_latency = mean([r["latency"] for r in results])

    print("\n===== FINAL METRICS =====")
    print(f"Retrieval Hit Rate: {round(hit_rate * 100, 2)}%")
    print(f"Average Answer Score: {round(avg_score, 2)} / 5")
    print(f"Average Latency: {round(avg_latency, 2)} sec")


# ===== Run =====
if __name__ == "__main__":
    results = run_evaluation()
    summarize(results)