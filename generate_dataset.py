import random
import string
from utils import load_json, save_json

PHONETIC_MAP = {
    # Common letter-sound confusions
    "ph": "f",
    "gh": "f",
    "ck": "k",
    "qu": "kw",

    # Vowel confusions
    "ae": "e",
    "oe": "e",
    "ou": "u",

    # Consonant confusions
    "x": "ks",
    "z": "s",
    "c": "k",

    # Common chemical/pharmaceutical suffixes
    "ine": "in",
    "amide": "amid",
    "azole": "azal",
    "mab": "maph",
    "pril": "pryl",
    "statin": "staten",
    "olol": "alol",
    "prazole": "prazol",
    "tidine": "tidin",
    "mycin": "misin",
    "cillin": "silin",
    "cycline": "sicline",

    # Pharmaceutical prefixes
    "cef": "sef",
    "ceph": "sef",
    "sulfa": "zulfa",
    "sulph": "sulf",
    "acetyl": "asetyl",
    "acet": "aset",
    "meth": "met",
    "phen": "fen",
    "chlor": "klor",
    "nitro": "nitra",
    "hydro": "hidro",
    "fluor": "flor",
}

def fuzz(word: str) -> str:
    """
    Applies a single random transform to a word to simulate a realistic misspelling.
    Transforms: deletion, transposition, substitution, insertion or phonetic swap.
    Phonetic swap is only offered if a matching pattern exists in the word.

    Returns:
        str: the transformed word (or the original unchanged if shorter than 4 characters)
    """
    
    if len(word) < 4:
        return word
    
    transforms = ["delete", "transpose", "substitute", "insert"]

    new_word = word
    phonetic_matches = [k for k in PHONETIC_MAP if k in word]
    if phonetic_matches:
        transforms.append("phonetic")

    choice = random.choice(transforms)

    if choice == "delete" and len(new_word) > 0:
        i = random.randint(0, len(new_word) - 1)
        new_word = new_word[:i] + new_word[i+1:]

    elif choice == "transpose" and len(new_word) > 1:
        i = random.randint(0, len(new_word) - 2)
        lst = list(new_word)
        lst[i], lst[i+1] = lst[i+1], lst[i]
        new_word = "".join(lst)

    elif choice == "substitute" and len(new_word) > 0:
        i = random.randint(0, len(new_word) - 1)
        new_char = random.choice([c for c in string.ascii_lowercase if c != word[i]])
        new_word = (
            new_word[:i] +
            new_char +
            new_word[i+1:]
        )

    elif choice == "insert":
        i = random.randint(0, len(new_word))
        new_word = (
            new_word[:i] +
            random.choice(string.ascii_lowercase) +
            new_word[i:]
        )

    elif choice == "phonetic":
        k = random.choice(phonetic_matches)
        new_word = word.replace(k, PHONETIC_MAP[k], 1)
    
    return new_word


def generate_eval_data(word_list: list[str]) -> list[dict]:
    """
    Generates fuzzy query/expected pairs from a list of chemical names.
    Each word is fuzzed once using a random transform to simulate a misspelling.
    Entries where fuzz produced no change are excluded.
    Args:
        word_list: List of drug names to generate eval pairs from

    Returns:
        list[dict]: keys 'query' and 'expected'
    """
    
    eval_data: list[dict] = []

    for word in word_list:
        query = fuzz(word)
        if query != word:
            eval_data.append({"query": query, "expected": word})

    return eval_data


if __name__ == "__main__":
    word_list = load_json("word_list.json")
    eval_data = generate_eval_data(word_list)
    save_json("eval_data.json", eval_data)
    print(f"Saved {len(eval_data)} eval pairs from {len(word_list)} words")

    # Sanity check: print some samples
    print("\nExample eval pairs:")
    for entry in random.sample(eval_data, min(5, len(eval_data))):
        print(f"{entry['query']} -> {entry['expected']}")