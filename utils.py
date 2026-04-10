import re
import json

def normalise(text: str) -> str:
    """Utility function to normalise text input."""

    text = text.lower().replace(" ", "")
    text = re.sub(r"[^a-z0-9]", "", text) # remove puncuation and symbols
    return text


def load_json(path: str):
    """Utility function to read JSON files."""

    with open(path, "r") as f:
        return json.load(f)
    

def save_json(path: str, data):
    """Utility function to write JSON files."""

    with open(path, "w") as f:
        json.dump(data, f, indent=2)