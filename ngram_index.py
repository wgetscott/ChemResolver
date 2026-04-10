from ranker import score, breakdown
from utils import normalise
from typing import Set
from dataclasses import dataclass
import math

@dataclass
class Result:
    """
    Represents a single ranked match result from the n-gram search system.

    Attributes:
        word (str):
            The matched candidate string (e.g., a chemical name or entity).

        score (float):
            The final similarity score combining:
            - Jaccard similarity (character overlap)
            - Levenshtein similarity (edit-distance similarity)
            - Prefix similarity (structural naming bias)
            - N-gram overlap (retrieval-stage signal)

            Higher values indicate stronger matches.

        overlap (float):
            Combination of Dice coefficient + TF-IDF weighted overlap.

        dice (float | None):
            Ratio of shared n-grams to total n-grams across query and 
            candidate. Only populated when debug=True.

        tfidf (float | None):
            TF-IDF weighted n-gram overlap. Downweights common n-grams
            due to their lesser specificity. Only populated when debug=True.

        jaccard (float | None):
            Character-level Jaccard similarity between query and candidate.
            Only populated when debug=True.

        levenshtein (float | None):
            Normalised edit-distance similarity between query and candidate.
            Only populated when debug=True.

        prefix (float | None):
            Prefix match bonus. Rewards shared leading characters.
            Only populated when debug=True.
    """
    
    word: str
    score: float
    overlap: float

    # Debug fields: only populated when debug=True
    dice: float | None = None
    tfidf: float | None = None
    jaccard: float | None = None
    levenshtein: float | None = None
    prefix: float | None = None 


class NGramIndex:
    """
    Simple inverted index for fast string matching using character n-grams.

    Stores a mapping:
        n-gram -> set of words containing that n-gram

    Allows fast candidate retrieval for similarity search.
    """

    def __init__(self, n: int = 3):
        """
        Initialises the index.

        Args:
            n (int): size of character n-grams (default = 3)
        """
        
        self.n = n
        self.index: dict[str, set[str]] = {} # stores words that contain a given n-gram
        self.n_gram_df: dict[str, int] = {} # document frequency of each n-gram


    def get_ngrams(self, text: str, n: int | None = None) -> Set[str]:
        """
        Generates character-level n-grams from a string.

        An n-gram is a contiguous substring of length n.

        Example:
            "ethanol", n=3 ->
            {"eth", "tha", "han", "ano", "nol"}

        Returns:
            set: unique n-grams
        """
        
        if n is None:
            n = self.n

        # Handle negative/invalid n-grams
        if n <= 0:
            raise ValueError("n must be positive")

        # Handle empty string
        if len(text) == 0:
            return set()
        
        # If string is shorter than n, treat whole string as a single gram (fallback)
        if len(text) < n:
            return {text}
        
        ngrams = set()

        # Slide a window of size n across the string
        for i in range(0, (len(text)-n) + 1):
            substring = text[i:i+n]
            ngrams.add(substring)

        return ngrams


    def add(self, word: str) -> None:
        """
        Adds a word to the n-gram inverted index.

        Each word is broken into character n-grams, and each n-gram
        is mapped to all words that contain it.

        Enables fast retrieval of candidate matches during search.
        """
        
        # Normalise
        clean_word = normalise(word)

        # Generate all n-grams for word
        grams = self.get_ngrams(clean_word)

        # Add the word to each n-gram bucket in the index
        for gram in grams:
            # New n-gram -> initialise it
            if not gram in self.index:
                self.index[gram] = set()

            # Add word to the set of words containing this n-gram
            self.index[gram].add(clean_word)
            
            # Update document frequency: counts how many distinct words
            # contain this n-gram
            if gram in self.n_gram_df:
                self.n_gram_df[gram] += 1
            else:
                self.n_gram_df[gram] = 1


    def add_many(self, words: list[str]) -> None:
        for word in words:
            self.add(word)


    def query(self, text: str, top_k: int | None = None, min_shared_ngrams: int = 2, debug: bool = False) -> list[Result]:
        """
        Queries the n-gram index to find and rank candidate matches.
        
        Can optionally be set to return only the top k candidates.
        
        Pipeline:
            1. Normalise input text
            2. Generate n-grams from normalised input
            3. Retrieve candidate words from inverted index
            4. Count n-gram overlaps per candidate
            5. Score each candidate using similarity metrics
            6. Return ranked results
        """

        # Normalise
        clean_text = normalise(text)

        # Generate n-grams from query
        query_grams = self.get_ngrams(clean_text)

        # Set of candidates
        candidates = set()

        # Track how many query n-grams each candidate matches
        shared_counts: dict[str, int] = {}

        # Retrieve candidate words that share at least one n-gram with the query
        for gram in query_grams:
            if gram in self.index:
                words = self.index[gram]
                for word in words:
                    candidates.add(word)

                    if word in shared_counts:
                        shared_counts[word] += 1
                    else:
                        shared_counts[word] = 1

        # Filter weak candidates (require at least 2 shared n-grams)
        filtered_candidates = [
            c for c in candidates 
            if shared_counts.get(c, 0) >= min_shared_ngrams
        ]

        results: list[Result] = []

        for candidate in filtered_candidates:
            candidate_grams = self.get_ngrams(candidate)

            intersection = query_grams & candidate_grams

            # Dice coefficient: ratio of shared n-grams to total n-grams across both strings.
            # Captures breadth of overlap: how much of the query is covered by the candidate.
            denom = len(query_grams) + len(candidate_grams)
            dice = (2 * len(intersection)) / denom if denom else 0.0
            
            # TF-IDF weighted overlap: each shared n-gram contributes 1/log(df+2) rather than 1.
            # Common n-grams (high df) are down-weighted; rarer n-grams contribute more due to their
            # specificity.
            tfidf_score = sum(
                1 / math.log(self.n_gram_df.get(g, 1) + 2) 
                for g in intersection
            )
            tfidf_normalised = (2 * tfidf_score) / denom if denom else 0.0

            # Blend both signals equally
            alpha = 0.5
            overlap = alpha * dice + (1 - alpha) * tfidf_normalised

            if debug:
                # Debug mode: include score breakdown in each Result object
                candidate_dict = breakdown(clean_text, candidate, overlap)

                results.append(Result(
                    word=candidate,
                    score=candidate_dict["final"],
                    overlap=overlap,
                    dice=dice,
                    tfidf=tfidf_normalised,
                    jaccard=candidate_dict["jaccard"],
                    levenshtein=candidate_dict["levenshtein"],
                    prefix=candidate_dict["prefix"]
                ))
            else:
                # Compare query string vs candidate string
                final_score = score(clean_text, candidate, overlap)

                results.append(Result(
                    word=candidate,
                    score=final_score,
                    overlap=overlap
                ))

        # Sort by similarity score (descending)
        ranked = sorted(results, key=lambda x: x.score, reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k] # return top k candidates

        return ranked
    