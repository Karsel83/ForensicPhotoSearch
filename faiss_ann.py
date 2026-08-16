"""FAISS ANN helper for ForensicPhotoSearch.

Run from the ForensicPhotoSearch project root. This file does not modify the
existing unified-search code; it builds and evaluates a FAISS index from the
already-created data/embeddings files.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
EMBEDDINGS = ROOT / "data" / "embeddings" / "embeddings.npy"
METADATA = ROOT / "data" / "embeddings" / "metadata.json"
FAISS_DIR = ROOT / "data" / "faiss"
INDEX_FILE = FAISS_DIR / "image.index"
CONFIG_FILE = FAISS_DIR / "index_config.json"
BENCHMARK_FILE = ROOT / "results" / "faiss_benchmark.json"


def get_faiss():
    try:
        import faiss
        return faiss
    except ImportError as error:
        raise SystemExit(
            "FAISS is not installed. Run: python -m pip install faiss-cpu==1.15.0"
        ) from error


def normalize(vectors):
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def index_type_for(count):
    if count < 10_000:
        return "flat"       # Exact baseline: ANN has no value for 9 vectors.
    if count < 500_000:
        return "hnsw"
    return "ivf_pq"


def build_index(vectors, index_type="auto", nprobe=32):
    faiss = get_faiss()
    vectors = normalize(vectors)
    count, dimension = vectors.shape
    index_type = index_type_for(count) if index_type == "auto" else index_type

    if index_type == "flat":
        index = faiss.IndexFlatIP(dimension)
        config = {"index_type": "flat"}

    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 512
        config = {
            "index_type": "hnsw",
            "hnsw_m": 32,
            "ef_construction": 200,
            "ef_search": 64,
        }

    elif index_type in {"ivf_flat", "ivf_pq"}:
        nlist = max(1, min(int(4 * math.sqrt(count)), count))
        quantizer = faiss.IndexFlatIP(dimension)
        if index_type == "ivf_flat":
            index = faiss.IndexIVFFlat(
                quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT
            )
        else:
            # 512 dimensions / 32 subquantizers = 16 dimensions per PQ chunk.
            index = faiss.IndexIVFPQ(
                quantizer, dimension, nlist, 32, 8, faiss.METRIC_INNER_PRODUCT
            )
        train_count = min(count, 200_000)
        train_ids = np.linspace(0, count - 1, train_count, dtype=np.int64)
        index.train(vectors[train_ids])
        index.nprobe = min(nprobe, nlist)
        config = {
            "index_type": index_type,
            "nlist": nlist,
            "nprobe": index.nprobe,
        }

    else:
        raise ValueError(f"Unknown index type: {index_type}")

    index.add(vectors)
    config.update({"vector_count": count, "dimension": dimension})
    return index, config


def build_existing_index(args):
    faiss = get_faiss()
    if not EMBEDDINGS.exists() or not METADATA.exists():
        raise SystemExit("Run build_database.py and build_embeddings.py first.")

    vectors = np.load(EMBEDDINGS)
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if len(vectors) != len(metadata):
        raise SystemExit("embeddings.npy and metadata.json have different lengths.")

    started = time.perf_counter()
    index, config = build_index(vectors, args.index_type, args.nprobe)
    config["build_seconds"] = round(time.perf_counter() - started, 6)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print(f"[*] Saved: {INDEX_FILE}")


def search_existing_index(args):
    faiss = get_faiss()
    if not INDEX_FILE.exists():
        raise SystemExit("Index missing. Run: python faiss_ann.py build")

    from reid_model import PersonReID

    index = faiss.read_index(str(INDEX_FILE))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    query_vector = PersonReID().extract(args.query)
    scores, ids = index.search(normalize(query_vector.reshape(1, -1)), args.top_k)

    for rank, (score, vector_id) in enumerate(zip(scores[0], ids[0]), start=1):
        if vector_id < 0:
            continue
        item = metadata[int(vector_id)]
        print(f"[{rank}] {item['filename']} score={score:.4f}")
        print(f"    crop={item['crop']}")


def mean_search_ms(index, queries, top_k, repeats):
    started = time.perf_counter()
    for _ in range(repeats):
        index.search(queries, top_k)
    return (time.perf_counter() - started) * 1000 / (len(queries) * repeats)


def recall_at_k(exact_ids, ann_ids):
    return float(np.mean([
        len(set(exact).intersection(approx)) / len(exact)
        for exact, approx in zip(exact_ids, ann_ids)
    ]))


def benchmark_one(size, queries, top_k, seed, nprobe, ann_index):
    """Synthetic vectors measure index scalability, not OSNet model accuracy."""
    faiss = get_faiss()
    try:
        import psutil
    except ImportError as error:
        raise SystemExit("Install psutil from requirements.txt first.") from error

    rng = np.random.default_rng(seed)
    vectors = normalize(rng.standard_normal((size, 512), dtype=np.float32))
    query_vectors = normalize(rng.standard_normal((queries, 512), dtype=np.float32))

    exact = faiss.IndexFlatIP(512)
    exact.add(vectors)
    _, exact_ids = exact.search(query_vectors, top_k)
    exact_ms = mean_search_ms(exact, query_vectors, top_k, 3)

    ann_type = ann_index or ("hnsw" if size < 500_000 else "ivf_pq")
    memory_before = psutil.Process(os.getpid()).memory_info().rss
    started = time.perf_counter()
    ann, config = build_index(vectors, ann_type, nprobe=nprobe)
    build_seconds = time.perf_counter() - started
    memory_after = psutil.Process(os.getpid()).memory_info().rss
    _, ann_ids = ann.search(query_vectors, top_k)

    return {
        "dataset": "synthetic_normalized_float32",
        "vector_count": size,
        "dimension": 512,
        "top_k": top_k,
        "exact_search_ms_per_query": round(exact_ms, 6),
        "ann_index": config,
        "ann_build_seconds": round(build_seconds, 6),
        "ann_search_ms_per_query": round(mean_search_ms(ann, query_vectors, top_k, 10), 6),
        "recall_at_k": round(recall_at_k(exact_ids, ann_ids), 6),
        "ann_memory_increase_bytes": memory_after - memory_before,
    }


def benchmark(args):
    results = []
    for size in args.sizes:
        print(f"[*] Benchmarking {size:,} vectors")
        result = benchmark_one(size, args.queries, args.top_k, args.seed, args.nprobe, args.ann_index)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    BENCHMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[*] Saved: {BENCHMARK_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Build, search, and benchmark FAISS ANN indexes.")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build")
    build.add_argument("--index-type", choices=("auto", "flat", "hnsw", "ivf_flat", "ivf_pq"), default="auto")
    build.add_argument("--nprobe", type=int, default=32)
    build.set_defaults(func=build_existing_index)

    search = commands.add_parser("search")
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=100)
    search.set_defaults(func=search_existing_index)

    test = commands.add_parser("benchmark")
    test.add_argument("--sizes", nargs="+", type=int, default=[10_000, 100_000, 1_000_000])
    test.add_argument("--queries", type=int, default=100)
    test.add_argument("--top-k", type=int, default=100)
    test.add_argument("--seed", type=int, default=42)
    test.add_argument("--nprobe", type=int, default=32)
    test.add_argument("--ann-index", choices=("hnsw", "ivf_flat", "ivf_pq"), default=None)
    test.set_defaults(func=benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()




