import math


def precision_at_k(retrieved_docs, expected_keywords, k):
    relevant = 0

    for doc in retrieved_docs[:k]:
        text = doc.page_content.lower()

        if any(
            keyword.lower() in text
            for keyword in expected_keywords
        ):
            relevant += 1

    return relevant / k if k > 0 else 0


def recall_at_k(retrieved_docs, expected_keywords, k):
    found_keywords = set()

    for doc in retrieved_docs[:k]:
        text = doc.page_content.lower()

        for keyword in expected_keywords:
            if keyword.lower() in text:
                found_keywords.add(
                    keyword.lower()
                )

    if len(expected_keywords) == 0:
        return 0

    return len(found_keywords) / len(expected_keywords)


def hit_rate(retrieved_docs, expected_keywords):

    for doc in retrieved_docs:

        text = doc.page_content.lower()

        if any(
            keyword.lower() in text
            for keyword in expected_keywords
        ):
            return 1

    return 0


def reciprocal_rank(retrieved_docs, expected_keywords):

    for rank, doc in enumerate(
        retrieved_docs,
        start=1
    ):

        text = doc.page_content.lower()

        if any(
            keyword.lower() in text
            for keyword in expected_keywords
        ):
            return 1 / rank

    return 0


def average_precision(
    retrieved_docs,
    expected_keywords,
    k
):

    precisions = []
    relevant_found = 0

    for idx, doc in enumerate(
        retrieved_docs[:k],
        start=1
    ):

        text = doc.page_content.lower()

        if any(
            keyword.lower() in text
            for keyword in expected_keywords
        ):
            relevant_found += 1
            precisions.append(
                relevant_found / idx
            )

    if len(precisions) == 0:
        return 0

    return sum(precisions) / len(precisions)


def dcg(relevances):

    score = 0

    for idx, rel in enumerate(relevances):
        score += rel / math.log2(idx + 2)

    return score


def ndcg(
    retrieved_docs,
    expected_keywords,
    k
):

    relevances = []

    for doc in retrieved_docs[:k]:

        text = doc.page_content.lower()

        rel = 1 if any(
            keyword.lower() in text
            for keyword in expected_keywords
        ) else 0

        relevances.append(rel)

    ideal_relevances = sorted(
        relevances,
        reverse=True
    )

    dcg_score = dcg(relevances)
    idcg_score = dcg(ideal_relevances)

    if idcg_score == 0:
        return 0

    return dcg_score / idcg_score