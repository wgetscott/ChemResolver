# Ranking Score Function for Chemical Entities

from similarity import jaccard_similarity, levenshtein_similarity

def score(a: str, b: str, overlap: float = 0.0) -> float:
    """
    Combined similarity score for chemical entity matching.

    Combines:
        - Jaccard similarity (character overlap)
        - Levenshtein similarity (edit distance-based similarity)
        - Prefix similarity (chemical naming bias)
        - N-gram overlap score (retrieval strength from inverted index)
        
    Returns:
        float: final score in [0, 1]
    """
    
    a, b = a.lower(), b.lower() # normalise

    ls = levenshtein_similarity(a, b)
    js = jaccard_similarity(a, b)
    pb = prefix_bonus(a, b)

    # Weighted linear combination
    final_score = 0.35*js + 0.35*ls + 0.15*pb + 0.15*overlap
    
    return final_score

def prefix_bonus(a: str, b: str, k: int = 3) -> float:
    """
    Measures how similar the prefixes of two strings are.

    Returns a value in [0, 1], where:
        - 1.0 -> full match in first k characters
        - 0.0 -> no prefix match

    Returns:
        float: prefix_bonus in [0, 1]
    """

    a, b = a.lower(), b.lower() # normalise

    # Compare up to k characters or shortest string length
    max_prefix = min(len(a), len(b), k)

    match_len = 0
    for i in range(max_prefix):
        if a[i] == b[i]:
            match_len += 1
        else:
            break

    # Return normalised prefix bonus
    return match_len / max_prefix if max_prefix > 0 else 0.0
    