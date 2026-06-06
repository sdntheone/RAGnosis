from statistics import mean

from deepeval import evaluate
from deepeval.metrics import (
    HallucinationMetric,
    AnswerRelevancyMetric
)
from deepeval.test_case import LLMTestCase

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


def build_test_cases():

    test_cases = []

    for item in eval_data:

        query = item["question"]

        docs = retriever.invoke(
            query
        )

        context = "\n".join(
            doc.page_content
            for doc in docs
        )

        answer = rag_chain.invoke(
            query
        )

        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            expected_output=item["ground_truth"],
            context=[context]
        )

        test_cases.append(
            test_case
        )

    return test_cases


def run_hallucination_eval(
    test_cases
):

    metric = HallucinationMetric(
        threshold=0.7,
        model="gpt-4o-mini"
    )

    scores = []

    for case in test_cases:

        metric.measure(case)

        scores.append(
            metric.score
        )

    return mean(scores)


def run_relevancy_eval(
    test_cases
):

    metric = AnswerRelevancyMetric(
        threshold=0.7,
        model="gpt-4o-mini"
    )

    scores = []

    for case in test_cases:

        metric.measure(case)

        scores.append(
            metric.score
        )

    return mean(scores)


if __name__ == "__main__":

    print(
        "\nBuilding DeepEval test cases...\n"
    )

    test_cases = (
        build_test_cases()
    )

    print(
        "\nRunning Hallucination Evaluation...\n"
    )

    hallucination_score = (
        run_hallucination_eval(
            test_cases
        )
    )

    print(
        "\nRunning Answer Relevancy Evaluation...\n"
    )

    relevancy_score = (
        run_relevancy_eval(
            test_cases
        )
    )

    print(
        "\n===== DEEPEVAL REPORT =====\n"
    )

    print(
        f"Hallucination Score: "
        f"{hallucination_score:.4f}"
    )

    print(
        f"Answer Relevancy Score: "
        f"{relevancy_score:.4f}"
    )