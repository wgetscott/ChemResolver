import re

def normalise(text: str) -> str:
    text = text.lower().replace(" ", "")
    text = re.sub(r"[^a-z0-9]", "", text) # remove puncuation and symbols
    return text
