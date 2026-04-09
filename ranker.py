# Ranking Score Function for Chemical Entities

from similarity import jaccard_similarity, levenshtein_similarity
from utils import normalise

def _weighted_sum(js: float, ls: float, pb: float, overlap: float) -> float:
    return 0.35*js + 0.35*ls + 0.15*pb + 0.15*overlap


def prefix_bonus(a: str, b: str, k: int = 3) -> float:
    """
    Measures how similar the prefixes of two strings are.

    Returns a value in [0, 1], where:
        - 1.0 -> full match in first k characters
        - 0.0 -> no prefix match

    Returns:
        float: prefix_bonus in [0, 1]
    """

    a, b = normalise(a), normalise(b) # normalise

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
    
    a, b = normalise(a), normalise(b)

    js = jaccard_similarity(a, b)
    ls = levenshtein_similarity(a, b)
    pb = prefix_bonus(a, b)

    # Weighted linear combination
    final_score = _weighted_sum(js, ls, pb, overlap)

    return final_score
    

def breakdown(a: str, b: str, overlap: float = 0.0) -> dict:
    """
    Returns the per-component score breakdown for a query/candidate pair.

    Same calculation as score(), but exposes the individual signals along
    with the final weighted score. Used for debug mode.

    Returns:
        dict: {
            "jaccard": Jaccard character similarity, 
            "levenshtein": normalised edit-distance similarity, 
            "prefix": prefix match bonus, 
            "overlap": combined Dice + TF-IDF signal, 
            "final": weighted combination of all component signals
        }
    """
    
    a, b = normalise(a), normalise(b)
    
    js = jaccard_similarity(a, b)
    ls = levenshtein_similarity(a, b)
    pb = prefix_bonus(a, b)

    # Weighted linear combination
    final_score = _weighted_sum(js, ls, pb, overlap)

    return {
        "jaccard": js,
        "levenshtein": ls,
        "prefix": pb,
        "overlap": overlap,
        "final": final_score
    }
