# Build n-gram Inverted Index

from ranker import score
from utils import normalise
from typing import Set
from dataclasses import dataclass

@dataclass
class Result:
    """
    Represents a single ranked match result from the n-gram search system.

    Attributes:
        word (str):
            The matched candidate string (e.g., checmical name or entity).

        score (float):
            The final similarity score combining:
            - Jaccard similarity (character overlap)
            - Levenshtein similarity (edit-distance similarity)
            - Prefix similarity (structural naming bias)
            - N-gram overlap (retrieval-stage signal)

            Higher values indicate stronger matches.

        overlap (float):
            Dice coefficient based on n-gram intersection between the
            query and candidate strings.
    """
    
    word: str
    score: float
    overlap: float


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


    def add(self, word: str):
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

    def add_many(self, words: list[str]):
        for word in words:
            self.add(word)


    def query(self, text: str, top_k: int | None = None, min_shared_ngrams: int = 2) -> list[Result]:
        """
        Queries the n-gram index to find and rank candidate matches.
        
        Can optionally be set to return only the top k candidates.
        
        Pipeline:
            1. Normalise input text
            2. Generate n-grams from normalised input
            3. Retrieve candidate words from inverted index
            4. Count n-gram overlaps per candidate
            5. Score each candidate using similarity metrics
            5. Return ranked results
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

            # Overlap as pairwise Dice coefficient between query and candidate n-grams.
            # Uses query grams and candidate grams only, ensuring similarity is independent of
            # retrieval set size.
            intersection = query_grams & candidate_grams
            denom = len(query_grams) + len(candidate_grams)

            overlap = (2 * len(intersection)) / denom if denom else 0.0 

            # Compare query string vs candidate string
            final_score = score(clean_text, candidate, overlap=overlap)

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
    