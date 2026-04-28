from evaluation.evaluator import run_evaluation, summarize


def main():
    print("Starting RAG Evaluation...\n")

    results = run_evaluation()

    summarize(results)


if __name__ == "__main__":
    main()