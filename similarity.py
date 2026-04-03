# String Similarity Measures

def jaccard_similarity(a: str, b: str) -> float:
    """
    Computes Jaccard similarity between two strings based on character sets.

    Jaccard similarity is defined as: |A ∩ B| / |A ∪ B| 

    Returns:
        float: similarity score in [0, 1]
            - 1.0 -> identical character sets
            - 0.0 -> no shared characters
    """
    
    union_len = len(set(a).union(b))
    intersection_len = len(set(a).intersection(b))

    if union_len == 0:
        return 0.0

    return intersection_len / union_len

def levenshtein_similarity(a: str, b: str) -> float:
    """
    Computes similarity between two strings using Levenshtein edit distance.

    Levenshtein edit distance measures the minimal number of edits required to
    transform one string into another using insertion, deletion and substitution.

    The distance is then converted into a normalised similarity score.

    Returns:
        float: normalised similarity score in [0, 1]
            - 1.0 -> identical strings
            - 0.0 -> completely different strings
    """
    
    m = len(a)
    n = len(b)

    # Edge case: both strings are empty
    if m == 0 and n == 0:
        return 1.0

    # dp[i][j] = min number of edits to convert a[:i] -> b[:j]
    dp = [[0] * (n+1) for _ in range(m+1)] # 2D matrix with m+1 rows and n+1 columns

    # === Base cases ===

    # If b is empty, we must delete all characters from a 
    for i in range(0, m+1):
        dp[i][0] = i # i deletions

    # If a is empty, we must insert all characters of b
    for j in range(0, n+1):
        dp[0][j] = j # j insertions

    # === Fill DP table ===
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                        # Deletion (remove a[i-1] and solve smaller problem)
                        dp[i-1][j] + 1,

                        # Insertion (insert b[j-1] into a)
                        dp[i][j-1] + 1,     

                        # Substitution (replace a[i-1] with b[j-1])
                        dp[i-1][j-1] + cost
                        )
            
    ld = dp[m][n] # Levenshtein distance

    # Normalise distance to similarity
    return 1 - (ld / max(m, n))
