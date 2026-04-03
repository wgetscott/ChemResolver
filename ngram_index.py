# Build n-gram Inverted Index

from ranker import score

class NGramIndex:
    """
    Simple inverted index for fast string matching using character n-grams.

    Stores a mapping:
        n-gram -> set of words containing that n-gram

    Allows fast candidate retrieval for similarity search.
    """

    def __init__(self, n:int = 3):
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

        # Normalise input
        clean_text = text.lower().replace(" ", "")

        ngrams = set()

        # Slide a window of size n across the string
        for i in range(0, (len(clean_text)-n) + 1):
            substring = clean_text[i:i+n]
            ngrams.add(substring)

        return ngrams


    def add(self, word: str):
        """
        Adds a word to the n-gram inverted index.

        Each word is broken into character n-grams, and each n-gram
        is mapped to all words that contain it.

        Enables fast retrieval of candidate matches during search.
        """
        
        # Generate all n-grams for word
        grams = self.get_ngrams(word)

        # Add the word to each n-gram bucket in the index
        for gram in grams:
            # New n-gram -> initialise it
            if not gram in self.index:
                self.index[gram] = set()

            # Add word to the set of words containing this n-gram
            self.index[gram].add(word)

    def add_many(self, words: list[str]):
        for word in words:
            self.add(word)

    def query(self, text: str, top_k: int | None = None) -> list:
        """
        Queries the n-gram index to find and rank candidate matches.
        
        Can optionally be set to return only the top k candidates.
        
        Pipeline:
            1. Normalise input
            2. Generate n-grams
            3. Retrieve candidate words from inverted index
            4. Count n-gram overlaps per candidate
            5. Score each candidate using similarity metrics
            6. Return ranked results
        """


        # Normalise text
        clean_text = text.lower().replace(" ", "")

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
            # Overlap
            overlap = counts[candidate] / len(grams) if grams else 0.0

            # Compare query string vs candidate string
            scores[candidate] = score(clean_text, candidate, overlap=overlap)

        # Sort by similarity score (descending)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if top_k is not None:
            top_k = max(0, top_k)
            ranked = ranked[:top_k]

        return ranked
