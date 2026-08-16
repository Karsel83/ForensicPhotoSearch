import csv
import gc
import os
import time
from pathlib import Path

import faiss
import numpy as np


ROOT = Path(__file__).resolve().parent

EMBEDDINGS_FILE = (
    ROOT
    / "data"
    / "embeddings"
    / "embeddings.npy"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "ann_benchmark"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CSV_FILE = (
    RESULT_DIR
    / "benchmark_results.csv"
)


# =========================================================
# Configuration
# =========================================================

DATASET_SIZES = [
    10_000,
    100_000,
    1_000_000
]

TOP_K_VALUES = [
    1,
    10,
    100
]

QUERY_COUNT = 10

RANDOM_SEED = 42

# HNSW
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128

# IVF
IVF_NLIST = 4000

IVF_FLAT_NPROBE = [
    128,
    512,
    1024,
    2048,
    3072,
    4000
]

IVF_PQ_NPROBE = [
    32,
    128,
    512
]

PQ_M = 64
PQ_NBITS = 8


# =========================================================
# Helpers
# =========================================================

def memory_mb():

    try:
        import psutil

        process = psutil.Process(
            os.getpid()
        )

        return (
            process.memory_info().rss
            / (1024 ** 2)
        )

    except ImportError:

        return float("nan")


def l2_normalize(
    vectors
):

    vectors = np.asarray(
        vectors,
        dtype=np.float32
    )

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    return vectors / np.maximum(
        norms,
        1e-12
    )


def create_dataset(
    base_embeddings,
    size
):

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    base_count = len(
        base_embeddings
    )

    if size <= base_count:

        return np.ascontiguousarray(
            base_embeddings[
                :size
            ],
            dtype=np.float32
        )

    # IMPORTANT:
    # This creates synthetic scale by repeating the
    # available embeddings with tiny deterministic noise.
    # It is suitable for infrastructure/scaling tests,
    # NOT for final retrieval-quality research results.

    repeats = (
        size // base_count
    )

    remainder = (
        size % base_count
    )

    parts = []

    for _ in range(repeats):

        parts.append(
            base_embeddings
        )

    if remainder:

        parts.append(
            base_embeddings[
                :remainder
            ]
        )

    vectors = np.concatenate(
        parts,
        axis=0
    )

    noise = (
        rng.normal(
            0,
            1e-6,
            size=vectors.shape
        )
        .astype(np.float32)
    )

    vectors += noise

    return l2_normalize(
        vectors
    )


def select_queries(
    embeddings,
    query_count
):

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    query_count = min(
        query_count,
        len(embeddings)
    )

    indices = rng.choice(
        len(embeddings),
        size=query_count,
        replace=False
    )

    return (
        embeddings[indices],
        indices
    )


# =========================================================
# Exact Ground Truth
# =========================================================

def exact_search(
    embeddings,
    queries,
    top_k
):

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    _, indices = index.search(
        queries,
        top_k
    )

    return indices


# =========================================================
# Recall
# =========================================================

def recall_at_k(
    predicted,
    ground_truth,
    k
):

    total = 0
    correct = 0

    for pred, truth in zip(
        predicted,
        ground_truth
    ):

        truth_set = set(
            map(
                int,
                truth[:k]
            )
        )

        pred_set = set(
            map(
                int,
                pred[:k]
            )
        )

        correct += len(
            truth_set
            & pred_set
        )

        total += len(
            truth_set
        )

    if total == 0:
        return 0.0

    return correct / total


# =========================================================
# Index Builders
# =========================================================

def build_flat(
    embeddings
):

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    return index


def build_hnsw(
    embeddings
):

    index = faiss.IndexHNSWFlat(
        embeddings.shape[1],
        HNSW_M,
        faiss.METRIC_INNER_PRODUCT
    )

    index.hnsw.efConstruction = (
        HNSW_EF_CONSTRUCTION
    )

    index.hnsw.efSearch = (
        HNSW_EF_SEARCH
    )

    index.add(
        embeddings
    )

    return index


def build_ivf_flat(
    embeddings
):

    quantizer = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index = faiss.IndexIVFFlat(
        quantizer,
        embeddings.shape[1],
        IVF_NLIST,
        faiss.METRIC_INNER_PRODUCT
    )

    train_count = min(
        len(embeddings),
        100_000
    )

    index.train(
        embeddings[:train_count]
    )

    index.add(
        embeddings
    )

    return index


def build_ivf_pq(
    embeddings
):

    quantizer = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index = faiss.IndexIVFPQ(
        quantizer,
        embeddings.shape[1],
        IVF_NLIST,
        PQ_M,
        PQ_NBITS,
        faiss.METRIC_INNER_PRODUCT
    )

    train_count = min(
        len(embeddings),
        100_000
    )

    index.train(
        embeddings[:train_count]
    )

    index.add(
        embeddings
    )

    return index


# =========================================================
# Benchmark One Index
# =========================================================

def benchmark_index(
    index,
    index_name,
    embeddings,
    queries,
    ground_truth,
    size,
    nprobe=None
):

    if (
        nprobe is not None
        and hasattr(index, "nprobe")
    ):

        index.nprobe = nprobe

    gc.collect()

    memory_before = (
        memory_mb()
    )

    build_started = (
        time.perf_counter()
    )

    # Index is already built before entering
    # this function. Build timing is therefore
    # measured outside.

    build_time = 0.0

    memory_after = (
        memory_mb()
    )

    search_times = []

    predicted = None

    for query in queries:

        query = np.ascontiguousarray(
            query.reshape(1, -1),
            dtype=np.float32
        )

        start = (
            time.perf_counter()
        )

        _, ids = index.search(
            query,
            100
        )

        elapsed = (
            time.perf_counter()
            - start
        ) * 1000.0

        search_times.append(
            elapsed
        )

        if predicted is None:

            predicted = []

        predicted.append(
            ids[0]
        )

    predicted = np.asarray(
        predicted,
        dtype=np.int64
    )

    result = {

        "dataset_size":
            size,

        "index":
            index_name,

        "nprobe":
            nprobe,

        "build_ms":
            build_time,

        "search_ms_mean":
            float(
                np.mean(
                    search_times
                )
            ),

        "search_ms_p95":
            float(
                np.percentile(
                    search_times,
                    95
                )
            ),

        "memory_delta_mb":
            float(
                memory_after
                - memory_before
            )
    }

    for k in TOP_K_VALUES:

        result[
            f"recall@{k}"
        ] = recall_at_k(
            predicted,
            ground_truth,
            k
        )

    return result


# =========================================================
# Build + Benchmark
# =========================================================

def run_index_test(
    embeddings,
    queries,
    ground_truth,
    size,
    index_name,
    builder,
    nprobe=None
):

    gc.collect()

    memory_before = (
        memory_mb()
    )

    started = (
        time.perf_counter()
    )

    index = builder(
        embeddings
    )

    build_ms = (
        time.perf_counter()
        - started
    ) * 1000.0

    memory_after = (
        memory_mb()
    )

    if (
        nprobe is not None
        and hasattr(index, "nprobe")
    ):

        index.nprobe = nprobe

    search_times = []

    predicted = []

    for query in queries:

        query = np.ascontiguousarray(
            query.reshape(1, -1),
            dtype=np.float32
        )

        started = (
            time.perf_counter()
        )

        _, ids = index.search(
            query,
            100
        )

        elapsed = (
            time.perf_counter()
            - started
        ) * 1000.0

        search_times.append(
            elapsed
        )

        predicted.append(
            ids[0]
        )

    predicted = np.asarray(
        predicted,
        dtype=np.int64
    )

    result = {

        "dataset_size":
            size,

        "index":
            index_name,

        "nprobe":
            nprobe,

        "build_ms":
            round(
                build_ms,
                6
            ),

        "search_ms_mean":
            round(
                float(
                    np.mean(
                        search_times
                    )
                ),
                6
            ),

        "search_ms_p95":
            round(
                float(
                    np.percentile(
                        search_times,
                        95
                    )
                ),
                6
            ),

        "memory_delta_mb":
            round(
                float(
                    memory_after
                    - memory_before
                ),
                6
            )
    }

    for k in TOP_K_VALUES:

        result[
            f"recall@{k}"
        ] = round(
            recall_at_k(
                predicted,
                ground_truth,
                k
            ),
            6
        )

    del index
    gc.collect()

    return result


# =========================================================
# Main
# =========================================================

def main():

    if not EMBEDDINGS_FILE.exists():

        raise FileNotFoundError(
            f"Embedding file not found: "
            f"{EMBEDDINGS_FILE}"
        )

    print(
        "[*] Loading embeddings..."
    )

    base_embeddings = np.load(
        EMBEDDINGS_FILE
    )

    base_embeddings = l2_normalize(
        base_embeddings
    )

    print(
        f"[*] Base embeddings: "
        f"{base_embeddings.shape}"
    )

    rows = []

    for size in DATASET_SIZES:

        print()
        print(
            "=" * 70
        )
        print(
            f"DATASET SIZE: {size:,}"
        )
        print(
            "=" * 70
        )

        embeddings = create_dataset(
            base_embeddings,
            size
        )

        queries, _ = select_queries(
            embeddings,
            QUERY_COUNT
        )

        ground_truth = exact_search(
            embeddings,
            queries,
            100
        )

        # -------------------------------------------------
        # Exact Flat
        # -------------------------------------------------

        print(
            "[*] IndexFlatIP"
        )

        rows.append(
            run_index_test(
                embeddings,
                queries,
                ground_truth,
                size,
                "IndexFlatIP",
                build_flat
            )
        )

        # -------------------------------------------------
        # HNSW
        # -------------------------------------------------

        print(
            "[*] HNSW"
        )

        rows.append(
            run_index_test(
                embeddings,
                queries,
                ground_truth,
                size,
                "HNSW",
                build_hnsw
            )
        )

        # -------------------------------------------------
        # IVF-Flat
        # -------------------------------------------------

        for nprobe in IVF_FLAT_NPROBE:

            print(
                f"[*] IVF-Flat "
                f"nprobe={nprobe}"
            )

            rows.append(
                run_index_test(
                    embeddings,
                    queries,
                    ground_truth,
                    size,
                    "IVF-Flat",
                    build_ivf_flat,
                    nprobe=nprobe
                )
            )

        # -------------------------------------------------
        # IVF-PQ
        # -------------------------------------------------

        for nprobe in IVF_PQ_NPROBE:

            print(
                f"[*] IVF-PQ "
                f"nprobe={nprobe}"
            )

            rows.append(
                run_index_test(
                    embeddings,
                    queries,
                    ground_truth,
                    size,
                    "IVF-PQ",
                    build_ivf_pq,
                    nprobe=nprobe
                )
            )

        del embeddings
        del queries
        del ground_truth

        gc.collect()

    # =====================================================
    # Save CSV
    # =====================================================

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
    )

    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        "=" * 70
    )
    print(
        "[*] BENCHMARK COMPLETE"
    )
    print(
        f"[*] Results: {CSV_FILE}"
    )
    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()