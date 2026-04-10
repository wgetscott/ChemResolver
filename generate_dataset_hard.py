from generate_dataset import fuzz, generate_eval_data
from utils import load_json, save_json
import random

def fuzz_hard(word: str, transforms: int = 2) -> str:
    """
    Applies multiple random transforms to a word to simulate a harder misspelling.
    Calls fuzz() repeatedly, chaining transforms.

    Args:
        word: input word to fuzz
        transforms: number of transforms to apply (default 2)
    Returns:
        str: the transformed word
    """
    
    result = word
    for _ in range(transforms):
        result = fuzz(result)
    return result


if __name__ == "__main__":
    word_list = load_json("word_list.json")
    eval_data = generate_eval_data(word_list, fuzz_fn=fuzz_hard)
    save_json("eval_data_hard.json", eval_data)
    print(f"Saved {len(eval_data)} eval pairs from {len(word_list)} words")

    # Sanity check: print some samples
    print("\nExample eval pairs:")
    for entry in random.sample(eval_data, min(5, len(eval_data))):
        print(f"{entry['query']} -> {entry['expected']}")
