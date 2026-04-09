import json
from pipeline import Pipeline
from utils import normalise

def load_json(path: str):
    """
    Utility function to read JSON files.
    """

    with open(path, "r") as f:
        return json.load(f)
    

def evaluate(word_list: list[str], eval_data: list[dict]) -> None:
    """
    Evaluates the pipeline on a set of fuzzy query/expected match pairs.

    Builds the index from word_list, queries each entry in eval_data,
    and reports the exact match (top_k=1) accuracy: the fraction of queries
    where the correct answer is ranked first.
    """

    p = Pipeline()
    p.build(word_list)

    hits = 0
    misses = 0
    no_results = 0

    for entry in eval_data:
        query = entry["query"]
        expected = entry["expected"]

        results = p.search(query, top_k=1)

        if not results:
            no_results += 1
            misses += 1
        elif normalise(results[0].word) == normalise(expected):
            hits += 1
        else:
            misses += 1

    total = len(eval_data)
    accuracy = hits / total if total else 0.0

    print("\n=== Evaluation Summary ===")
    print(f"Total queries: {total}")
    print(f"Correct (hits): {hits}")
    print(f"Misses: {misses}")
    print(f"No results: {no_results}")
    print(f"Exact Match Accuracy: {accuracy:.2%}")
    

if __name__ == "__main__":
    word_list = load_json("word_list.json")
    eval_data = load_json("eval_data.json")

    evaluate(word_list, eval_data)