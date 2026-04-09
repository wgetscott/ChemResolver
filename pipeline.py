from ngram_index import NGramIndex, Result

class Pipeline:
    def __init__(self, n: int = 3):
        self._index = NGramIndex(n=n)


    def build(self, words: list[str]) -> None:
        """
        Builds the n-gram index from a list of words.

        Args:
            words: candidate words to index (e.g., chemical names)
        """

        self._index.add_many(words)

    
    def search(self, query: str, top_k: int | None = None, min_shared_ngrams: int = 2, debug: bool = False) -> list[Result]:
        """
        Searches the index for the closest matches to a query string.

        Args:
            query: input string to match against the index
            top_k: if set, return only the top k results
            min_shared_ngrams: minimum n-gram overlap to consider a candidate
            debug: if true, populate per-component score fields on each Result
            
        Returns:
            Ranked list of Result objects, or empty list if no matches found.
        """

        results = self._index.query(
            query,
            top_k=top_k,
            min_shared_ngrams=min_shared_ngrams,
            debug=debug
        )

        return results
    