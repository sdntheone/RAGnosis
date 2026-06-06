from datasets import Dataset

from ragas import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from app.llm.rag_chain import get_rag_chain
from app.retrieval.hybrid_retriever import (
    HybridRetriever
)

from evaluation.eval_data import eval_data


rag_chain = get_rag_chain()

retriever = HybridRetriever(
    dense_k=10,
    sparse_k=10,
    rerank_top_k=5
)


def build_dataset():

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in eval_data:

        question = item["question"]

        docs = retriever.invoke(
            question
        )

        context = [
            doc.page_content
            for doc in docs
        ]

        answer = rag_chain.invoke(
            question
        )

        questions.append(
            question
        )

        answers.append(
            answer
        )

        contexts.append(
            context
        )

        ground_truths.append(
            item["ground_truth"]
        )

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
    )

    return dataset


def run_ragas():

    dataset = build_dataset()

    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
    )

    return results


if __name__ == "__main__":

    result = run_ragas()

    print("\n===== RAGAS REPORT =====\n")

    print(result)