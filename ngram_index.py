# Build n-gram Inverted Index

from ranker import score
from utils import normalise
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


    def get_ngrams(self, text: str, n: int = 3) -> set:
        """
        Generates character-level n-grams from a string.

        An n-gram is a contiguous substring of length n.

        Example:
            "ethanol", n=3 ->
            {"eth", "tha", "han", "ano", "nol"}

        Returns:
            set: unique n-grams
        """

        # Handle negative/invalid n-grams
        if n <= 0:
            raise ValueError("n must be positive")

        # Handle empty string
        if len(text) == 0:
            return set()
        
        # Handle short strings (below n-gram requirement)
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


    def add_many(self, words: list[str]):
        for word in words:
            self.add(word)


    def query(self, text: str, top_k: int | None = None) -> list:
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
        grams = self.get_ngrams(clean_text)

        # Set of candidates
        candidates = set()

        # Tracks how many n-grams each candidate shares with query
        counts: dict[str, int] = {}

        for gram in grams:
            if gram in self.index:
                words = self.index[gram]
                for word in words:
                    candidates.add(word)

                    # Count how many times this word appears across n-grams
                    if word in counts:
                        counts[word] += 1
                    else:
                        counts[word] = 1

        candidates_list = list(candidates)

        # Score tracker
        scores = {}

        for candidate in candidates_list:
            candidate_grams = self.get_ngrams(candidate)

            # Overlap as pairwise Dice coefficient between query and candidate n-grams.
            # Uses query grams and candidate grams only, ensuring similarity is independent of
            # retrieval set size.
            denom = len(grams) + len(candidate_grams)
            overlap = (2 * counts[candidate]) / denom if denom else 0.0 

            # Compare query string vs candidate string
            scores[candidate] = score(clean_text, candidate, overlap=overlap)

        # Sort by similarity score (descending)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if top_k is not None:
            top_k = max(0, top_k)
            ranked = ranked[:top_k] # return top k candidates

        return ranked
    
