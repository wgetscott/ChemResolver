import time
import statistics
from pipeline import Pipeline
from utils import load_json

def benchmark(word_list: list[str], bench_data: list[dict], repeat: int = 3, top_k: int = 5) -> dict:
    """
    Benchmarks the pipeline on index build time and query performance.
    Builds the index from word_list, then runs each query in bench_data
    repeat times, reporting build time in seconds and average, median, and p99 query
    latency in milliseconds.
    """

    p = Pipeline()

    # === Index ===
    start = time.perf_counter()
    p.build(word_list)
    index_time = time.perf_counter() - start

    print("\n=== Index Benchmark ===")
    print(f"Words indexed: {len(word_list)}")
    print(f"Index build time: {index_time:.6f}s ({index_time *1000:.3f} ms)")

    # === Queries ===
    query_times: list[float] = []
    
    for _ in range(repeat):
        for entry in bench_data:
            start = time.perf_counter()
            p.search(entry["query"], top_k=top_k)
            query_times.append(time.perf_counter() - start)

    total_queries = len(bench_data) * repeat
    avg_ms = statistics.mean(query_times) * 1000
    median_ms = statistics.median(query_times) * 1000
    p99_ms = statistics.quantiles(query_times, n=100)[98] * 1000

    print("\n=== Query Benchmark ===")
    print(f"Total queries: {total_queries}")
    print(f"Average query time: {avg_ms:.3f} ms")
    print(f"Median query time: {median_ms:.3f} ms")
    print(f"p99 query time: {p99_ms:.3f} ms")

    return {
        "index_time_s": index_time,
        "query_times": query_times,
        "avg_query_ms": avg_ms,
        "median_query_ms": median_ms,
        "p99_query_ms": p99_ms,
    }


if __name__ == "__main__":
    word_list = load_json("word_list.json")
    bench_data = load_json("eval_data.json")
    benchmark(word_list, bench_data)
